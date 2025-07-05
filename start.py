#!/usr/bin/env python3
"""
Robust startup script for Railway deployment
"""
import os
import sys
import logging
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    try:
        port = int(os.environ.get("PORT", 8000))
        host = "0.0.0.0"

        logger.info(f"Python version: {sys.version}")
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Python path: {sys.path}")
        logger.info(f"Starting SignSafe API on {host}:{port}")

        # Add current directory to Python path
        current_dir = os.path.dirname(os.path.abspath(__file__))
        if current_dir not in sys.path:
            sys.path.insert(0, current_dir)
            logger.info(f"Added {current_dir} to Python path")

        # Try to import the app first to catch any import errors
        try:
            from app.main import app
            logger.info("Successfully imported FastAPI app")
        except Exception as e:
            logger.error(f"Failed to import app: {e}")
            raise

        # Start the server
        uvicorn.run(
            "app.main:app",
            host=host,
            port=port,
            reload=False,
            log_level="info",
            access_log=True
        )

    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
