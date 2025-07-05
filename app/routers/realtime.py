from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import logging
import asyncio
from app.services.ml_service import ml_service

logger = logging.getLogger(__name__)

router = APIRouter()

# Disabled websocket endpoint to avoid conflict with prediction.py websocket
# @router.websocket("/ws/real_time_predict")
# async def websocket_endpoint(websocket: WebSocket):
#     await websocket.accept()
#     logger.info("WebSocket connection accepted for real-time prediction")
#     try:
#         frame_buffer = []
#         max_frames = 16  # Should match model expected frames
#         while True:
#             data = await websocket.receive_text()
#             import json
#             import base64
#             import numpy as np
#             from PIL import Image
#             import io

#             message = json.loads(data)
#             frame_base64 = message.get("frame")
#             if not frame_base64:
#                 await websocket.send_json({"error": "No frame data received"})
#                 continue
#             frame_bytes = base64.b64decode(frame_base64)

#             # Convert bytes to image
#             image = Image.open(io.BytesIO(frame_bytes)).convert('RGB')
#             # Resize to model expected size (128x128)
#             image = image.resize((128, 128))
#             # Convert to numpy array
#             frame_array = np.array(image) / 255.0  # Normalize
#             frame_buffer.append(frame_array)

#             if len(frame_buffer) == max_frames:
#                 # Prepare input batch
#                 input_batch = np.expand_dims(np.array(frame_buffer), axis=0)  # Shape (1, 16, 128, 128, 3)
#                 # Predict
#                 prediction = ml_service.predict(input_batch)
#                 # Clear buffer
#                 frame_buffer = []

#                 prediction_result = {
#                     "predicted_word": prediction.predicted_word,
#                     "confidence": prediction.confidence
#                 }
#                 await websocket.send_json(prediction_result)
#     except WebSocketDisconnect:
#         logger.info("WebSocket connection closed")
#     except Exception as e:
#         logger.error(f"Error in WebSocket real-time prediction: {e}")
#         await websocket.close()
