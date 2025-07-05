import tensorflow as tf
import numpy as np
import json
import logging
import time
from typing import List, Optional, Any
from app.utils.config import settings
from app.models.schemas import PredictionResponse, PredictionAlternative

logger = logging.getLogger(__name__)

class MLModelService:
    """Service for machine learning model operations"""
    
    def __init__(self):
        self.model: Optional[Any] = None
        self.class_labels: List[str] = []
        self.model_loaded = False
        
    def load_model(self) -> bool:
        """
        Load the trained model and class labels
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        try:
            # Check if model file exists
            if not settings.model_exists:
                logger.error(f"Model file not found: {settings.model_path}")
                return False
            
            # Load the model
            logger.info(f"Loading model from: {settings.model_path}")
            # Load the model using TensorFlow's load_model function
            self.model = tf.saved_model.load(settings.model_path) if settings.model_path.endswith('.pb') else tf.keras.models.load_model(settings.model_path)  # type: ignore
            logger.info("Model loaded successfully")
            
            # Load class labels
            if not self._load_class_labels():
                logger.error("Failed to load class labels")
                return False
            
            # Verify model input shape
            self._verify_model_input_shape()
            
            self.model_loaded = True
            logger.info(f"ML Service initialized with {len(self.class_labels)} classes")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            self.model_loaded = False
            return False
    
    def _load_class_labels(self) -> bool:
        """Load class labels from JSON file"""
        try:
            if not settings.labels_exist:
                logger.error(f"Labels file not found: {settings.labels_path}")
                return False
            
            with open(settings.labels_path, 'r') as f:
                class2idx_loaded = json.load(f)
            
            if not class2idx_loaded:
                logger.error("Class index dictionary is empty")
                return False
            
            # Convert class2idx to list of class names
            # Sort by index to maintain order
            temp_labels = [""] * len(class2idx_loaded)
            for class_name, idx in class2idx_loaded.items():
                idx = int(idx)  # Cast idx to int to avoid type errors
                if 0 <= idx < len(class2idx_loaded):
                    temp_labels[idx] = class_name
            self.class_labels = temp_labels
            
            # Remove any None values (shouldn't happen with proper data)
            self.class_labels = [label for label in self.class_labels if label is not None]
            
            if not self.class_labels:
                logger.error("No valid class labels found")
                return False
            
            logger.info(f"Loaded {len(self.class_labels)} class labels")
            return True
            
        except Exception as e:
            logger.error(f"Error loading class labels: {e}")
            return False
    
    def _verify_model_input_shape(self) -> None:
        """Verify model input shape matches expected dimensions"""
        try:
            if self.model is None:
                logger.warning("Model is None, cannot verify input shape")
                return
            input_shape = self.model.input_shape
            logger.info(f"Model input shape: {input_shape}")
            
            expected_frames = settings.max_frames
            expected_size = settings.target_size  # This is now a tuple (width, height)
            expected_height, expected_width = expected_size[1], expected_size[0]  # height, width

            if len(input_shape) == 5:
                # Expected shape: (batch_size, frames, height, width, channels)
                if input_shape[1] != expected_frames:
                    logger.warning(f"Model expects {input_shape[1]} frames, but configured for {expected_frames}")

                if input_shape[2] != expected_height or input_shape[3] != expected_width:
                    logger.warning(f"Model expects {input_shape[2]}x{input_shape[3]} images, but configured for {expected_height}x{expected_width}")

            elif len(input_shape) == 4:
                # Expected shape: (batch_size, height, width, channels)
                if input_shape[1] != expected_height or input_shape[2] != expected_width:
                    logger.warning(f"Model expects {input_shape[1]}x{input_shape[2]} images, but configured for {expected_height}x{expected_width}")
            
            else:
                logger.warning(f"Unexpected model input shape: {input_shape}")
                
        except Exception as e:
            logger.warning(f"Could not verify model input shape: {e}")
    
    def predict(self, video_frames: np.ndarray, top_k: int = 3) -> PredictionResponse:
        """
        Make prediction on video frames
        
        Args:
            video_frames: Preprocessed video frames
            top_k: Number of top predictions to return
            
        Returns:
            PredictionResponse with results
        """
        if not self.model_loaded or self.model is None:
            raise ValueError("Model not loaded")
        
        try:
            start_time = time.time()

            # Reshape video_frames to match model input shape (batch_size, height, width, channels)
            # video_frames shape: (frames, height, width, channels)
            if len(video_frames.shape) == 4:
                video_batch = video_frames.reshape((-1, video_frames.shape[1], video_frames.shape[2], video_frames.shape[3]))
            else:
                video_batch = video_frames

            # Make prediction
            predictions = self.model.predict(video_batch, verbose=0)

            processing_time = time.time() - start_time
            
            # Log raw prediction scores for debugging
            logger.debug(f"Raw prediction scores: {predictions}")
            
            # Aggregate predictions if multiple frames (batch)
            if len(predictions.shape) == 2 and predictions.shape[0] > 1:
                avg_prediction_scores = predictions.mean(axis=0)
            else:
                avg_prediction_scores = predictions[0]
            
            # Get top predictions
            top_indices = np.argsort(avg_prediction_scores)[::-1][:top_k]
            
            # Main prediction
            main_prediction_idx = top_indices[0]
            main_confidence = float(avg_prediction_scores[main_prediction_idx])
            main_word = self._get_class_name(main_prediction_idx)
            
            # Alternative predictions
            alternatives = []
            for i in range(1, min(top_k, len(top_indices))):
                idx = top_indices[i]
                confidence = float(avg_prediction_scores[idx])
                word = self._get_class_name(idx)
                
                alternatives.append(PredictionAlternative(
                    word=word,
                    confidence=confidence
                ))
            
            return PredictionResponse(
                predicted_word=main_word,
                confidence=main_confidence,
                alternatives=alternatives,
                processing_time=processing_time
            )
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise ValueError(f"Prediction failed: {str(e)}")

    def predict_single_frame(self, frame: np.ndarray, top_k: int = 3) -> PredictionResponse:
        """
        Make prediction on a single preprocessed frame
        
        Args:
            frame: Preprocessed single frame (height, width, channels)
            top_k: Number of top predictions to return
        
        Returns:
            PredictionResponse with results
        """
        if not self.model_loaded or self.model is None:
            raise ValueError("Model not loaded")

        try:
            start_time = time.time()

            # Expand dims to create batch size 1
            # Model expects input shape: (batch_size, height, width, channels)
            input_data = np.expand_dims(frame, axis=0)  # (1, height, width, channels)

            predictions = self.model.predict(input_data, verbose=0)

            processing_time = time.time() - start_time
            prediction_scores = predictions[0]

            top_indices = np.argsort(prediction_scores)[::-1][:top_k]

            main_prediction_idx = top_indices[0]
            main_confidence = float(prediction_scores[main_prediction_idx])
            main_word = self._get_class_name(main_prediction_idx)

            alternatives = []
            for i in range(1, min(top_k, len(top_indices))):
                idx = top_indices[i]
                confidence = float(prediction_scores[idx])
                word = self._get_class_name(idx)

                alternatives.append(PredictionAlternative(
                    word=word,
                    confidence=confidence
                ))

            return PredictionResponse(
                predicted_word=main_word,
                confidence=main_confidence,
                alternatives=alternatives,
                processing_time=processing_time
            )

        except Exception as e:
            logger.error(f"Error during single frame prediction: {e}")
            raise ValueError(f"Single frame prediction failed: {str(e)}")


    
    def _get_class_name(self, class_index: int) -> str:
        """Get class name from index"""
        if 0 <= class_index < len(self.class_labels):
            return self.class_labels[class_index]
        else:
            logger.warning(f"Invalid class index: {class_index}")
            return "unknown"
    
    def get_supported_classes(self) -> List[str]:
        """Get list of supported classes"""
        return self.class_labels.copy()
    
    def is_model_loaded(self) -> bool:
        """Check if model is loaded"""
        return self.model_loaded
    
    def get_model_info(self) -> dict:
        """Get model information"""
        if not self.model_loaded or self.model is None:
            return {"status": "not_loaded"}
        
        try:
            return {
                "status": "loaded",
                "input_shape": self.model.input_shape,
                "output_shape": self.model.output_shape,
                "total_params": self.model.count_params(),
                "num_classes": len(self.class_labels),
                "classes": self.class_labels[:10] if len(self.class_labels) > 10 else self.class_labels  # Show first 10 classes
            }
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {"status": "error", "message": str(e)}

# Global service instance
ml_service = MLModelService()