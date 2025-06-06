import logging
import json
from typing import Type
from pydantic import BaseModel, ValidationError
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Body, File, UploadFile, Form, Depends
from fastapi.responses import StreamingResponse

from tts_api.core.config import settings
from tts_api.core.models import ChatterboxTTSRequest, ExportFormat
from tts_api.tts_engines.chatterbox_engine import chatterbox_engine
from tts_api.services.tts_service import generate_speech_from_request

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# A central registry for all available TTS engine implementations
ENGINE_REGISTRY = {
    "chatterbox": chatterbox_engine,
    # Future engines would be added here:
    # "another_engine": another_engine_instance,
}


def model_from_form_json(model_class: Type[BaseModel]):
    """
    A dependency that parses a JSON string from a form field and validates
    it against a Pydantic model.
    """
    def dependency(json_string: str = Form(..., alias="req_json")) -> BaseModel:
        try:
            model_data = json.loads(json_string)
            return model_class.model_validate(model_data)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=422, 
                detail="Invalid JSON format in the 'req_json' form field."
            )
        except ValidationError as e:
            raise HTTPException(
                status_code=422,
                # Use FastAPI's built-in error formatting for a consistent feel
                detail=e.errors() 
            )
    return dependency

@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Application startup...")
    # Selectively load models based on the ENABLED_MODELS setting
    enabled_engine_count = 0
    for engine_name in settings.ENABLED_MODELS:
        if engine_name in ENGINE_REGISTRY:
            try:
                logging.info(f"Loading '{engine_name}' engine...")
                ENGINE_REGISTRY[engine_name].load_model()
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

MEDIA_TYPES = {
    ExportFormat.MP3: "audio/mpeg",
    ExportFormat.WAV: "audio/wav",
    ExportFormat.FLAC: "audio/flac",
}

ChatterboxRequestFromForm = model_from_form_json(ChatterboxTTSRequest)

@app.post("/chatterbox/generate", 
          summary="Generate Speech using Chatterbox (Voice Cloning)",
          tags=["Generation"])
async def generate_chatterbox_speech(
    req: ChatterboxTTSRequest = Depends(ChatterboxRequestFromForm),
    ref_audio: UploadFile = File(..., description="A required WAV/MP3 file for voice cloning.")
):
    """
    Generate speech from text using the Chatterbox engine. The audio
    is streamed directly in the response body.
    """
    # Check if the requested engine is enabled in the current server configuration
    if "chatterbox" not in settings.ENABLED_MODELS:
        raise HTTPException(
            status_code=503, 
            detail="The 'chatterbox' engine is not enabled on this server."
        )

    try:
        ref_audio_data = await ref_audio.read() if ref_audio else None

        audio_buffer = generate_speech_from_request(
            req=req, 
            engine=chatterbox_engine, 
            engine_params=req.chatterbox_params,
            ref_audio_data=ref_audio_data
        )
        media_type = MEDIA_TYPES.get(
            req.post_processing.export_format, 
            "application/octet-stream"
        )
        
        return StreamingResponse(audio_buffer, media_type=media_type)

    except ValueError as ve:
        logging.warning(f"Bad request during TTS generation: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="An internal error occurred.")

@app.get("/health", summary="Check API Health", tags=["Management"])
async def health_check():
    return {"status": "ok", "enabled_models": settings.ENABLED_MODELS}