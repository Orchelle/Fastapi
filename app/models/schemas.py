from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class PredictionAlternative(BaseModel):
    """Alternative prediction with confidence score"""
    word: str = Field(..., description="Predicted sign word")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score between 0 and 1")

class PredictionResponse(BaseModel):
    """Response model for sign language prediction"""
    predicted_word: str = Field(..., description="Most likely predicted sign")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score for main prediction")
    alternatives: List[PredictionAlternative] = Field(default=[], description="Alternative predictions")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    timestamp: datetime = Field(default_factory=datetime.now, description="Prediction timestamp")

class HealthResponse(BaseModel):
    """Health check response model"""
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the ML model is loaded")
    server_time: datetime = Field(default_factory=datetime.now, description="Current server time")
    version: str = Field(default="1.0.0", description="API version")

class ClassesResponse(BaseModel):
    """Response model for supported classes"""
    classes: List[str] = Field(..., description="List of supported sign language classes")
    total_classes: int = Field(..., description="Total number of supported classes")

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")

class VideoInfo(BaseModel):
    """Video information model"""
    filename: str = Field(..., description="Original filename")
    size: int = Field(..., description="File size in bytes")
    duration: Optional[float] = Field(None, description="Video duration in seconds")
    frames_count: Optional[int] = Field(None, description="Total number of frames")
    fps: Optional[float] = Field(None, description="Frames per second")
    resolution: Optional[str] = Field(None, description="Video resolution (e.g., '640x480')")

class PredictionRequest(BaseModel):
    """Internal model for prediction processing"""
    video_info: VideoInfo
    frames: List[List[List[List[float]]]]  # 4D array: [frames, height, width, channels]
    preprocessing_time: float