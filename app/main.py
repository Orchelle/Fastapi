import logging
import sys
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.utils.config import settings
from app.services.ml_service import ml_service
from app.routers import prediction, realtime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('signsafe_api.log')
    ]
)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager
    Handles startup and shutdown events
    """
    # Startup
    logger.info("Starting SignSafe API...")
    logger.info(f"Model path: {settings.model_path}")
    logger.info(f"Labels path: {settings.labels_path}")
    
    # Load ML model
    if settings.model_exists and settings.labels_exist:
        logger.info("Loading ML model...")
        model_loaded = ml_service.load_model()
        if model_loaded:
            logger.info("Model loaded successfully")
        else:
            logger.error("Failed to load model")
    else:
        logger.warning("Model or labels file not found. API will start but predictions will fail.")
        logger.warning(f"Model exists: {settings.model_exists}")
        logger.warning(f"Labels exist: {settings.labels_exist}")
    
    logger.info("SignSafe API startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down SignSafe API...")

# Create FastAPI app with lifespan
app = FastAPI(
    title=settings.api_title,
    description=settings.api_description,
    version=settings.api_version,
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=settings.allowed_methods,
    allow_headers=settings.allowed_headers,
)

# Include routers
app.include_router(prediction.router, prefix="/api/v1", tags=["prediction"])
app.include_router(realtime.router)

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Welcome to SignSafe API",
        "version": settings.api_version,
        "description": "Sign Language Recognition API using MobileNetV2 + BiLSTM",
        "model_loaded": ml_service.is_model_loaded(),
        "endpoints": {
            "health": "/health",
            "predict": "/api/v1/predict",
            "classes": "/api/v1/classes",
            "model_info": "/api/v1/model/info"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint for deployment platforms"""
    return {
        "status": "healthy",
        "service": "SignSafe API",
        "model_loaded": ml_service.is_model_loaded(),
        "timestamp": "2025-01-05T00:00:00Z"
    }

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc: Exception):
    """Global exception handler for unexpected errors"""
    logger.error(f"Unexpected error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "INTERNAL_SERVER_ERROR",
            "message": "An unexpected error occurred",
            "details": str(exc) if settings.debug else "Contact administrator"
        }
    )

if __name__ == "__main__":
    import uvicorn
    
    logger.info(f"Starting server on {settings.host}:{settings.port}")
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info" if settings.debug else "warning"
    )
