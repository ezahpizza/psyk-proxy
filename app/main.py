"""
FastAPI application entrypoint for mental health prediction service.
"""
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.model_loader import ModelLoader
from app.db.mongodb import mongodb
from app.api.predict import router as predict_router
from app.utils.logger import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown events.
    """
    # Startup: Connect to MongoDB and load ML models
    try:
        await mongodb.connect()
        logger.info("Connected to MongoDB successfully")
        
        # Load ML models
        model_loader = ModelLoader()
        app.state.model_loader = model_loader
        logger.info("ML models loaded successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize application: {e}")
        raise
    finally:
        # Shutdown: Disconnect from MongoDB
        await mongodb.disconnect()
        logger.info("Application shutdown completed")


# Create FastAPI app
app = FastAPI(
    title="Mental Health Prediction API",
    description="FastAPI service for predicting mental health care options",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(predict_router)

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "psyk-proxy"}

