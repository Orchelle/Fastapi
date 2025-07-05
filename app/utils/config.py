import os
from pathlib import Path
from typing import Optional, Tuple
from pydantic_settings import BaseSettings
from pydantic import field_validator

class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # Server Configuration
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = True
    
    # Model Configuration
    model_path: str = "data/models/signsafe_model.h5"
    labels_path: str = "data/labels/class_index.json"  # Changed to match your file
    
    # Video Processing Configuration
    max_video_size: str = "50MB"
    max_frames: int = 16  # NUM_FRAMES from your training
    target_size: Tuple[int, int] = (128, 128)  # IMG_SIZE from your training (width, height)
    
    # API Configuration
    api_title: str = "SignSafe API"
    api_description: str = "Sign Language Recognition API"
    api_version: str = "1.0.0"
    
    # CORS Settings
    allowed_origins: list = ["*"]
    allowed_methods: list = ["*"]
    allowed_headers: list = ["*"]
    
    @field_validator('target_size', mode='before')
    @classmethod
    def parse_target_size(cls, v):
        """Parse target_size from various formats to tuple"""
        if isinstance(v, str):
            # Handle parentheses format: "(128, 128)"
            if v.startswith('(') and v.endswith(')'):
                v = v.strip('()')
                parts = [int(x.strip()) for x in v.split(',')]
                if len(parts) == 2:
                    return tuple(parts)
                else:
                    raise ValueError(f"target_size must be a tuple of 2 integers, got: {v}")
        elif isinstance(v, list) and len(v) == 2:
            # Handle list format: [128, 128]
            return tuple(v)
        return v

    class Config:
        env_file = ".env"
        case_sensitive = False

    @property
    def model_exists(self) -> bool:
        """Check if model file exists"""
        return Path(self.model_path).exists()
    
    @property
    def labels_exist(self) -> bool:
        """Check if labels file exists"""
        return Path(self.labels_path).exists()
    
    def get_max_video_size_bytes(self) -> int:
        """Convert max_video_size string to bytes"""
        size_str = self.max_video_size.upper()
        if size_str.endswith('MB'):
            return int(size_str[:-2]) * 1024 * 1024
        elif size_str.endswith('KB'):
            return int(size_str[:-2]) * 1024
        elif size_str.endswith('GB'):
            return int(size_str[:-2]) * 1024 * 1024 * 1024
        else:
            return int(size_str)

# Global settings instance
settings = Settings()