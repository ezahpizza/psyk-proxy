"""
MongoDB connection and management using async PyMongo.
"""
import logging
from typing import Optional
from pymongo import AsyncMongoClient

from app.core.config import settings

logger = logging.getLogger(__name__)


class MongoDB:
    """MongoDB connection manager."""
    
    def __init__(self):
        self.client: Optional[AsyncMongoClient] = None
        self.db = None

    async def connect(self):
        """Connect to MongoDB."""
        try:
            self.client = AsyncMongoClient(settings.MONGODB_URL)
            self.db = self.client[settings.DATABASE_NAME]
            await self.client.admin.command('ping')
            logger.info(f"Connected to MongoDB: {settings.DATABASE_NAME}")
        except Exception as e:
            logger.error(f"Failed to connect to MongoDB: {e}")
            raise

    async def disconnect(self):
        """Disconnect from MongoDB."""
        if self.client:
            await self.client.close()
            logger.info("Disconnected from MongoDB")

    def get_care_predictions_collection(self):
        """Get the care predictions collection."""
        if self.db is None:
            raise RuntimeError("Database not connected")
        return self.db["care_predictions"]

    def get_user_collection(self):
        """Get the users collection."""
        if self.db is None:
            raise RuntimeError("Database not connected")
        return self.db["users"]

    def get_alerts_collection(self):
        """Get the alerts collection."""
        if self.db is None:
            raise RuntimeError("Database not connected")
        return self.db["alerts"]


# Global MongoDB instance
mongodb = MongoDB()
