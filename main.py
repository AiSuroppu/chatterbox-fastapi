import logging
import json
from dataclasses import dataclass
from typing import Type, Optional
from pydantic import BaseModel, ValidationError
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Body, File, UploadFile, Form, Depends
from fastapi.responses import StreamingResponse, JSONResponse

from tts_api.core.config import settings
from tts_api.core.models import BaseTTSRequest, ChatterboxTTSRequest, ExportFormat
from tts_api.tts_engines.base import AbstractTTSEngine
from tts_api.tts_engines.chatterbox_engine import chatterbox_engine
from tts_api.services.tts_service import generate_speech_from_request

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class EngineDefinition:
    """A container for an engine's implementation, request model, and parameter key."""
    instance: AbstractTTSEngine
    request_model: Type[BaseTTSRequest]
    param_field_name: str

# A central registry for all available TTS engine definitions
ENGINE_REGISTRY: dict[str, EngineDefinition] = {
    "chatterbox": EngineDefinition(
        instance=chatterbox_engine,
        request_model=ChatterboxTTSRequest,
        param_field_name="chatterbox_params"  # The name of the field in ChatterboxTTSRequest
    ),
    # To add a new engine, you would just add another entry here:
    # "new_engine": EngineDefinition(
    #     instance=new_engine_instance,
    #     request_model=NewEngineTTSRequest,
    #     param_field_name="new_engine_params"
    # ),
}

MEDIA_TYPES = {
    ExportFormat.MP3: "audio/mpeg",
    ExportFormat.WAV: "audio/wav",
    ExportFormat.FLAC: "audio/flac",
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Application startup...")
    # Selectively load models based on the ENABLED_MODELS setting
    enabled_engine_count = 0
    for engine_name in settings.ENABLED_MODELS:
        if engine_name in ENGINE_REGISTRY:
            try:
                logging.info(f"Loading '{engine_name}' engine...")
                ENGINE_REGISTRY[engine_name].instance.load_model() 
                enabled_engine_count += 1
            except Exception as e:
                logging.error(f"Failed to load '{engine_name}' engine: {e}", exc_info=True)
        else:
            logging.warning(f"Configuration requests to enable '{engine_name}', but no such engine is registered.")
    
    logging.info(f"Startup complete. {enabled_engine_count} TTS engine(s) loaded.")
    yield
    logging.info("Application shutdown.")

app = FastAPI(
    title="Pluggable TTS API",
    description="A modular API for multiple Text-to-Speech engines.",
    version="2.2.0",
    lifespan=lifespan
)

def dynamic_request_parser():
    """
    A dependency that dynamically parses a JSON string from a form field ('req_json')
    against the correct Pydantic model based on the `engine_name` path parameter.
    """
    async def dependency(engine_name: str, req_json: str = Form(..., alias="req_json")) -> BaseTTSRequest:
        if engine_name not in ENGINE_REGISTRY:
            # This check is technically redundant if the endpoint does it first,
            # but it's good practice for a self-contained dependency.
            raise HTTPException(status_code=404, detail=f"Engine '{engine_name}' not found.")
        
        # Look up the correct Pydantic model from our central registry
        model_class = ENGINE_REGISTRY[engine_name].request_model
        
        try:
            model_data = json.loads(req_json)
            return model_class.model_validate(model_data)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=422, 
                detail="Invalid JSON format in the 'req_json' form field."
            )
        except ValidationError as e:
            raise HTTPException(
                status_code=422,
                detail=e.errors() 
            )
    return dependency

@app.get(
    "/engines/{engine_name}/config",
    response_model=BaseTTSRequest, # Use a base model for docs, but will return specific
    summary="Get Default Configuration for an Engine",
    tags=["Management"]
)
async def get_engine_config(engine_name: str):
    """
    Returns a complete, default JSON request payload for the specified engine.
    This is useful for discovering all available parameters.
    """
    if engine_name not in ENGINE_REGISTRY:
        raise HTTPException(
            status_code=404, 
            detail=f"Engine '{engine_name}' not found. Available engines: {list(ENGINE_REGISTRY.keys())}"
        )

    engine_def = ENGINE_REGISTRY[engine_name]
    
    # Create a model instance without validation. This is the correct
    # tool to create a template object. It will have all default fields populated.
    default_payload_obj = engine_def.request_model.model_construct(
        # Set the required 'text' field so the output is a valid template.
        text="Your text to synthesize goes here."
        )
    
    # Dump the model to a dictionary, making sure to include unset fields.
    payload_dict = default_payload_obj.model_dump(exclude_unset=False)

    # Return a final JSONResponse object or FastAPI will remove engine-specific parameters!
    return JSONResponse(content=payload_dict)

@app.post(
    "/{engine_name}/generate", 
    summary="Generate Speech with a Specific Engine",
    tags=["Generation"]
)
async def generate_speech(
    engine_name: str,
    req: BaseTTSRequest = Depends(dynamic_request_parser()),
    ref_audio: Optional[UploadFile] = File(None, description="Reference audio file (e.g., WAV/MP3), required by some engines like 'chatterbox'.")
):
    """
    Generate speech from text using the specified TTS engine.
    
    - **engine_name**: The name of the engine to use (e.g., 'chatterbox').
    - **req_json**: A form field containing the JSON payload for the request. 
      Use the `/engines/{engine_name}/config` endpoint to get a template.
    - **ref_audio**: A file upload, required for voice cloning engines.
    """
    if engine_name not in settings.ENABLED_MODELS:
        raise HTTPException(
            status_code=503, 
            detail=f"The '{engine_name}' engine is not enabled on this server."
        )

    try:
        engine_def = ENGINE_REGISTRY[engine_name]
        
        # Dynamically get the engine-specific parameters object using the field name
        # from our registry. This is the key to making the service call generic.
        engine_params = getattr(req, engine_def.param_field_name)

        ref_audio_data = await ref_audio.read() if ref_audio else None

        # The tts_service is already generic, so we can call it directly.
        audio_buffer = generate_speech_from_request(
            req=req, 
            engine=engine_def.instance, 
            engine_params=engine_params,
            ref_audio_data=ref_audio_data
        )
        
        media_type = MEDIA_TYPES.get(
            req.post_processing.export_format, 
            "application/octet-stream"
        )
        
        return StreamingResponse(audio_buffer, media_type=media_type)

    except ValueError as ve:
        logging.warning(f"Bad request for engine '{engine_name}': {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logging.error(f"An unexpected error occurred with engine '{engine_name}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred during generation.")

@app.get("/health", summary="Check API Health", tags=["Management"])
async def health_check():
    return {"status": "ok", "enabled_models": settings.ENABLED_MODELS}