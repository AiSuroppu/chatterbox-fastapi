import logging
import json
import importlib
import base64
from dataclasses import dataclass
from typing import Type, Optional
from pydantic import BaseModel, ValidationError
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Body, File, UploadFile, Form, Depends, Request
from fastapi.responses import StreamingResponse, JSONResponse

from tts_api.core.config import settings
from tts_api.core.exceptions import (
    ClientRequestError,
    EngineExecutionError,
    EngineLoadError,
    InvalidVoiceTokenException
)
from tts_api.core.models import BaseTTSRequest, ExportFormat, TTSResponseWithTimestamps
from tts_api.tts_engines.base import AbstractTTSEngine
from tts_api.services.tts_service import generate_speech_from_request

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass
class EngineDefinition:
    """A container for an engine's implementation, request model, and parameter key."""
    instance: AbstractTTSEngine
    request_model: Type[BaseTTSRequest]
    param_field_name: str

# Define the metadata for all SUPPORTED engines.
SUPPORTED_ENGINES = {
    "chatterbox": {
        "engine_path": "tts_api.tts_engines.chatterbox_engine.chatterbox_engine",
        "model_path": "tts_api.core.models.ChatterboxTTSRequest",
        "param_field_name": "chatterbox_params"
    },
    "fish_speech": {
        "engine_path": "tts_api.tts_engines.fish_speech_engine.fish_speech_engine",
        "model_path": "tts_api.core.models.FishSpeechTTSRequest",
        "param_field_name": "fish_speech_params"
    },
    # To add a new engine, you would just add another entry here:
    # "new_engine": {
    #     "engine_path": "path.to.new_engine.instance",
    #     "model_path": "path.to.new_engine.NewEngineTTSRequest",
    #     "param_field_name": "new_engine_params"
    # }
}

def _import_from_string(path: str):
    """Dynamically imports an object from a string path like 'module.submodule.object'."""
    module_path, object_name = path.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, object_name)

# Dynamically build the ENGINE_REGISTRY. This registry will only contain engines that were successfully imported.
ENGINE_REGISTRY: dict[str, EngineDefinition] = {}

logging.info("Discovering and registering available TTS engines...")
for name, config in SUPPORTED_ENGINES.items():
    try:
        # Attempt to import the engine instance and its request model
        engine_instance = _import_from_string(config["engine_path"])
        request_model = _import_from_string(config["model_path"])

        # If both imports succeed, add the engine to our live registry
        ENGINE_REGISTRY[name] = EngineDefinition(
            instance=engine_instance,
            request_model=request_model,
            param_field_name=config["param_field_name"]
        )
        logging.info(f"Successfully registered engine: '{name}'")
    except (ImportError, AttributeError) as e:
        # This is the graceful guard. If a module or object doesn't exist, we log
        # it and move on without crashing the application.
        logging.warning(f"Could not register engine '{name}'. It will be unavailable. Reason: {e}")

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
    available_engines = list(ENGINE_REGISTRY.keys())
    logging.info(f"Discovered engines available for loading: {available_engines}")

    for engine_name in settings.ENABLED_MODELS:
        if engine_name in ENGINE_REGISTRY:
            try:
                logging.info(f"Loading '{engine_name}' engine...")
                ENGINE_REGISTRY[engine_name].instance.load_model()
                enabled_engine_count += 1
            except EngineLoadError as e:
                logging.critical(f"FATAL: Failed to load '{engine_name}' engine: {e}", exc_info=True)
            except Exception as e:
                logging.critical(f"An unexpected error occurred while loading '{engine_name}': {e}", exc_info=True)
        else:
            logging.warning(f"Configuration requests to enable '{engine_name}', but no such engine is registered or available.")

    logging.info(f"Startup complete. {enabled_engine_count} TTS engine(s) loaded.")
    yield
    logging.info("Application shutdown.")

app = FastAPI(
    title="Pluggable TTS API",
    description="A modular API for multiple Text-to-Speech engines.",
    version="2.2.0",
    lifespan=lifespan
)

@app.exception_handler(ClientRequestError)
async def client_request_error_handler(request: Request, exc: ClientRequestError):
    logging.warning(f"Client request error for '{request.url.path}': {exc.message}")
    return JSONResponse(
        status_code=400,
        content={"detail": exc.message},
    )

@app.exception_handler(InvalidVoiceTokenException)
async def invalid_voice_token_handler(request: Request, exc: InvalidVoiceTokenException):
    logging.warning(f"Invalid voice token used for '{request.url.path}': {exc.message}")
    return JSONResponse(
        status_code=409,  # 409 Conflict is more appropriate for an invalid state token
        content={"detail": exc.message},
    )

@app.exception_handler(EngineExecutionError)
async def engine_execution_error_handler(request: Request, exc: EngineExecutionError):
    # Log with full traceback because this is a server-side failure
    logging.error(f"Engine execution error for '{request.url.path}': {exc.message}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal error occurred during generation. Check server logs for details."},
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    # This is a final safety net for any unhandled exceptions.
    logging.error(f"An unexpected unhandled exception occurred for '{request.url.path}': {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An unexpected internal server error occurred."},
    )

def dynamic_request_parser():
    """
    A dependency that dynamically parses a JSON string from a form field ('req_json')
    against the correct Pydantic model based on the `engine_name` path parameter.
    """
    async def dependency(engine_name: str, req_json: str = Form(..., alias="req_json")) -> BaseTTSRequest:
        if engine_name not in ENGINE_REGISTRY:
            raise HTTPException(
                status_code=404,
                detail=f"Engine '{engine_name}' not found. Available engines: {list(ENGINE_REGISTRY.keys())}"
            )

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
    tags=["Generation"],
    responses={
        200: {
            "description": "Successful audio generation. The content type will be audio/mpeg, audio/wav, or audio/flac unless 'return_timestamps' is true.",
            "content": {
                "application/json": {
                    "schema": {
                        "$ref": "#/components/schemas/TTSResponseWithTimestamps"
                    }
                },
                "audio/mpeg": {},
                "audio/wav": {},
                "audio/flac": {}
            }
        }
    }
)
async def generate_speech(
    engine_name: str,
    req: BaseTTSRequest = Depends(dynamic_request_parser()),
    ref_audio: Optional[UploadFile] = File(None, description="Reference audio file (e.g., WAV/MP3), required by some engines.")
):
    """
    Generate speech from text using the specified TTS engine.

    - **engine_name**: The name of the engine to use (e.g., 'chatterbox').
    - **req_json**: A form field containing the JSON payload for the request.
      Use the `/engines/{engine_name}/config` endpoint to get a template.
    - **ref_audio**: A file upload, required for voice cloning engines.

    Some engines may return engine-specific data in the `X-Generation-Metadata`
    response header as a JSON string.
    """
    if engine_name not in ENGINE_REGISTRY:
         raise HTTPException(
            status_code=404,
            detail=f"Engine '{engine_name}' not found. Available engines: {list(ENGINE_REGISTRY.keys())}"
        )

    if engine_name not in settings.ENABLED_MODELS:
        raise HTTPException(
            status_code=503,
            detail=f"The '{engine_name}' engine is installed but not enabled on this server. Enabled engines: {settings.ENABLED_MODELS}"
        )

    engine_def = ENGINE_REGISTRY[engine_name]
    engine_params = getattr(req, engine_def.param_field_name)

    ref_audio_data = await ref_audio.read() if ref_audio else None

    synthesis_result = generate_speech_from_request(
        req=req,
        engine=engine_def.instance,
        engine_params=engine_params,
        ref_audio_data=ref_audio_data
    )

    if req.return_timestamps:
        audio_bytes = synthesis_result.audio_buffer.getvalue()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        response_data = TTSResponseWithTimestamps(
            audio_content=audio_base64,
            word_segments=synthesis_result.word_segments or []
        )
        return response_data
    else:
        media_type = MEDIA_TYPES.get(
            req.post_processing.export_format,
            "application/octet-stream"
        )

        response = StreamingResponse(synthesis_result.audio_buffer, media_type=media_type)
        if synthesis_result.metadata:
            metadata_json = json.dumps(synthesis_result.metadata)
            response.headers["X-Generation-Metadata"] = metadata_json
            logging.info(f"Returning generation metadata in header: {metadata_json}")

        return response

@app.get("/health", summary="Check API Health", tags=["Management"])
async def health_check():
    return {
        "status": "ok",
        "registered_engines": list(ENGINE_REGISTRY.keys()),
        "enabled_models": settings.ENABLED_MODELS
    }