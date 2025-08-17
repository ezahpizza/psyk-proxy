"""
Configuration management using Pydantic BaseSettings.
"""
import os
from typing import List
from pydantic import Field, ConfigDict
from pydantic_settings  import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8"
    )
    
    # API Configuration
    DEBUG: bool = False
    
    # CORS Configuration
    CORS_ORIGINS: str = Field(
        default="http://localhost:8080",
        json_schema_extra={"env": "CORS_ORIGINS"}
    )
    
    # MongoDB Configuration
    MONGODB_URL: str = Field(..., json_schema_extra={"env": "MONGODB_URL"})
    DATABASE_NAME: str = Field(..., json_schema_extra={"env": "DATABASE_NAME"})
    
    # Model Configuration
    MODEL_PATH: str = os.path.abspath("app/mL")
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"


settings = Settings()
