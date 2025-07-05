import time
import logging
from fastapi import APIRouter, File, UploadFile, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi import WebSocket, WebSocketDisconnect
from app.models.schemas import (
    PredictionResponse, 
    HealthResponse, 
    ClassesResponse, 
    ErrorResponse
)
from app.services.ml_service import ml_service
from app.services.video_service import video_service
from app.utils.config import settings
import base64
import numpy as np
import cv2
import json

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint
    Returns the current status of the API and model
    """
    return HealthResponse(
        status="healthy" if ml_service.is_model_loaded() else "unhealthy",
        model_loaded=ml_service.is_model_loaded(),
        version=settings.api_version
    )

@router.post("/predict", response_model=PredictionResponse)
async def predict_sign_language(
    file: UploadFile = File(..., description="Video file containing sign language gestures")
):
    """
    Predict sign language from uploaded video
    
    - **file**: Video file (mp4, avi, mov, mkv, wmv, flv, webm)
    - Returns the predicted sign with confidence score and alternatives
    """
    
    # Check if model is loaded
    if not ml_service.is_model_loaded():
        raise HTTPException(
            status_code=503, 
            detail="Model not available. Please check server configuration."
        )
    
    start_time = time.time()
    
    try:
        # Process video file
        logger.info(f"Processing video file: {file.filename}")
        video_frames, video_info = await video_service.process_uploaded_video(file)
        
        processing_time = time.time() - start_time
        logger.info(f"Video processing completed in {processing_time:.2f} seconds")
        
        # Log shape of video_frames for debugging
        logger.info(f"Video frames shape before prediction: {video_frames.shape}")
        
        # Make prediction
        prediction_start = time.time()
        result = ml_service.predict(video_frames)
        prediction_time = time.time() - prediction_start
        
        # Add processing time to result
        result.processing_time = time.time() - start_time
        
        logger.info(f"Prediction completed: '{result.predicted_word}' with confidence {result.confidence:.3f}")
        
        return result
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@router.websocket("/ws/real_time_predict")
async def websocket_real_time_predict(websocket: WebSocket):
    """
    WebSocket endpoint for real-time sign language prediction.
    Expects base64 encoded frames from client.
    Sends back JSON prediction responses.
    """
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            # Expecting JSON with base64 frame string
            message = json.loads(data)
            frame_b64 = message.get("frame")
            if not frame_b64:
                await websocket.send_text(json.dumps({"error": "No frame data received"}))
                continue
            
            # Decode base64 to bytes
            frame_bytes = base64.b64decode(frame_b64)
            # Convert bytes to numpy array
            np_arr = np.frombuffer(frame_bytes, np.uint8)
            # Decode image
            frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            if frame is None:
                await websocket.send_text(json.dumps({"error": "Invalid frame data"}))
                continue
            
            # Preprocess frame
            preprocessed_frame = video_service._preprocess_frame(frame)

            # Check if there's meaningful sign activity in the frame
            if _is_sign_detected(frame):
                # Predict on single frame (expand dims to match model input)
                prediction = ml_service.predict_single_frame(preprocessed_frame)

                # Only send predictions with reasonable confidence
                if prediction.confidence >= 0.4:  # Higher threshold for actual signs
                    await websocket.send_text(prediction.json())
                else:
                    # Send empty response for low confidence
                    await websocket.send_text(json.dumps({"predicted_word": "", "confidence": 0.0}))
            else:
                # No sign detected, send empty response
                await websocket.send_text(json.dumps({"predicted_word": "", "confidence": 0.0}))
            
    except WebSocketDisconnect:
        logger.info("WebSocket disconnected")
    except Exception as e:
        logger.error(f"Error in WebSocket real-time prediction: {e}")
        await websocket.close(code=1011)


def _is_sign_detected(frame: np.ndarray) -> bool:
    """
    Detect if there's meaningful sign language activity in the frame.
    Uses motion detection and hand presence analysis.
    """
    try:
        # Convert to grayscale for analysis
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Check for sufficient contrast and detail (not just blank/static frame)
        # Calculate image variance - low variance indicates static/blank image
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()

        # If variance is too low, likely no meaningful content
        if variance < 50:  # Threshold for detecting static/blank frames
            return False

        # Check for hand-like regions using skin color detection
        # Convert to HSV for better skin detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define skin color range in HSV
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)

        # Create mask for skin-colored regions
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)

        # Calculate percentage of skin-colored pixels
        skin_pixels = cv2.countNonZero(skin_mask)
        total_pixels = frame.shape[0] * frame.shape[1]
        skin_percentage = skin_pixels / total_pixels

        # If there's a reasonable amount of skin-colored regions, likely hands are present
        # But not too much (which might indicate face/full body)
        if 0.05 <= skin_percentage <= 0.4:  # 5% to 40% of frame
            return True

        return False

    except Exception as e:
        logger.warning(f"Error in sign detection: {e}")
        # If detection fails, assume sign is present to avoid blocking predictions
        return True
