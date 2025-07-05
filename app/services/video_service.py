import cv2
import numpy as np
import tempfile
import os
from typing import Tuple
from fastapi import UploadFile, HTTPException
import logging
from app.utils.config import settings
from app.models.schemas import VideoInfo

logger = logging.getLogger(__name__)

class VideoProcessingService:
    """Service for video processing operations"""
    
    def __init__(self):
        self.max_file_size = settings.get_max_video_size_bytes()
        # Use settings for target frames and size
        self.target_frames = settings.max_frames
        self.target_size = settings.target_size
    
    async def process_uploaded_video(self, file: UploadFile) -> Tuple[np.ndarray, VideoInfo]:
        """
        Process uploaded video file and return preprocessed frames
        
        Args:
            file: Uploaded video file
            
        Returns:
            Tuple of (preprocessed_frames, video_info)
        """
        # Validate file
        await self._validate_video_file(file)
        
        # Save to temporary file
        temp_path = await self._save_temp_file(file)
        
        try:
            # Extract video info
            filename = file.filename or "unknown_video.mp4"
            video_info = self._extract_video_info(temp_path, filename)
            
            # Process video frames
            frames = self._extract_and_preprocess_frames(temp_path)
            
            # Log frames shape for debugging
            logger.info(f"[VideoService] Extracted frames shape: {frames.shape}")
            
            return frames, video_info
            
        finally:
            # Clean up temporary file
            self._cleanup_temp_file(temp_path)
    
    async def _validate_video_file(self, file: UploadFile) -> None:
        """Validate uploaded video file"""
        
        # Check content type
        if not file.content_type or not file.content_type.startswith('video/'):
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file type. Expected video file, got {file.content_type}"
            )
        
        # Check file size
        file.file.seek(0, 2)  # Seek to end
        file_size = file.file.tell()
        file.file.seek(0)  # Reset to beginning
        
        if file_size > self.max_file_size:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {settings.max_video_size}"
            )
        
        # Check filename
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")
        
        # Check file extension
        allowed_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        file_ext = os.path.splitext(file.filename)[1].lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file extension: {file_ext}. Allowed: {', '.join(allowed_extensions)}"
            )
    
    async def _save_temp_file(self, file: UploadFile) -> str:
        """Save uploaded file to temporary location"""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
                content = await file.read()
                tmp_file.write(content)
                return tmp_file.name
                
        except Exception as e:
            logger.error(f"Error saving temporary file: {e}")
            raise HTTPException(status_code=500, detail="Failed to save uploaded file")
    
    def _extract_video_info(self, video_path: str, filename: str) -> VideoInfo:
        """Extract video metadata"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise HTTPException(status_code=400, detail="Cannot open video file")
            
            # Get video properties
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else None
            
            cap.release()
            
            # Get file size
            file_size = os.path.getsize(video_path)
            
            return VideoInfo(
                filename=filename,
                size=file_size,
                duration=duration,
                frames_count=frame_count,
                fps=fps,
                resolution=f"{width}x{height}"
            )
            
        except Exception as e:
            logger.error(f"Error extracting video info: {e}")
            raise HTTPException(status_code=500, detail="Failed to extract video information")
    
    def _extract_and_preprocess_frames(self, video_path: str) -> np.ndarray:
        """
        Extract and preprocess frames from video
        
        Args:
            video_path: Path to video file
            
        Returns:
            Preprocessed frames as numpy array
        """
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise HTTPException(status_code=400, detail="Cannot open video file")
            
            frames = []
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            # Read all frames first
            all_frames = []
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                all_frames.append(frame)
            cap.release()
            
            logger.info(f"[VideoService] Total frames extracted: {len(all_frames)}")
            
            # Sample frames uniformly to target_frames
            if len(all_frames) == 0:
                raise HTTPException(status_code=400, detail="No frames extracted from video")
            
            if len(all_frames) < self.target_frames:
                # Pad frames by repeating last frame
                while len(all_frames) < self.target_frames:
                    all_frames.append(all_frames[-1])
            
            indices = np.linspace(0, len(all_frames) - 1, self.target_frames, dtype=int)
            sampled_frames = [all_frames[i] for i in indices]
            
            logger.info(f"[VideoService] Sampled frames count: {len(sampled_frames)}")
            
            # Resize and preprocess frames
            processed_frames = []
            for idx, frame in enumerate(sampled_frames):
                # Resize frame explicitly to target_size
                resized_frame = cv2.resize(frame, self.target_size)
                # Convert BGR to RGB
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                # Normalize to [0,1]
                normalized_frame = rgb_frame.astype(np.float32) / 255.0
                processed_frames.append(normalized_frame)
                logger.debug(f"[VideoService] Processed frame {idx} shape: {normalized_frame.shape}")
            
            return np.array(processed_frames)
            
        except Exception as e:
            logger.error(f"Error processing video frames: {e}")
            raise HTTPException(status_code=500, detail="Failed to process video frames")
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess individual frame for model input
        
        Args:
            frame: Raw frame from video
            
        Returns:
            Preprocessed frame
        """
        try:
            # Resize frame
            frame = cv2.resize(frame, self.target_size)
            
            # Convert BGR to RGB (OpenCV uses BGR by default)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Normalize pixel values to [0, 1]
            frame = frame.astype(np.float32) / 255.0
            
            return frame
            
        except Exception as e:
            logger.error(f"Error preprocessing frame: {e}")
            raise HTTPException(status_code=500, detail="Failed to preprocess frame")
    
    def _cleanup_temp_file(self, temp_path: str) -> None:
        """Clean up temporary file"""
        try:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        except Exception as e:
            logger.warning(f"Failed to cleanup temporary file {temp_path}: {e}")

# Global service instance
video_service = VideoProcessingService()