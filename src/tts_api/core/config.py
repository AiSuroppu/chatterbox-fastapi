import torch
from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Generic model path dictionary for multiple engines
    # In .env file, you can set it like:
    # MODEL_PATHS='{"chatterbox": "/path/to/your/chatterbox_model"}'
    MODEL_PATHS: dict[str, str] = {}

    # List of engine names to load on startup.
    # In .env file: ENABLED_MODELS='["chatterbox"]'
    ENABLED_MODELS: list[str] = ["chatterbox"]

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()