"""
Test configuration and fixtures.
"""
import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient
import numpy as np

from app.core.model_loader import ModelLoader
from app.main import app
from app.db import mongodb


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_model():
    """Mock ML model for testing."""
    model = MagicMock()
    # Mock predict_proba to return realistic probabilities for 3 targets
    model.predict_proba.return_value = [
        np.array([[0.2, 0.8]]),  # care_options_No: 20%, care_options_Yes: 80%
        np.array([[0.6, 0.4]]),  # care_options_Maybe: 60%, care_options_Yes: 40%
        np.array([[0.9, 0.1]])   # care_options_NotSure: 90%, care_options_Yes: 10%
    ]
    return model


@pytest.fixture
def mock_encoder():
    """Mock encoder for testing."""
    encoder = MagicMock()
    encoder.transform.return_value = [1.5]  # Mock encoded country value
    return encoder


@pytest.fixture
def mock_feature_names():
    """Mock feature names for testing."""
    return [
        'gender', 'country', 'family_history', 'treatment', 'coping_struggles',
        'occupation_Corporate', 'occupation_Student', 'occupation_Teacher',
        'days_indoors_1-14 days', 'days_indoors_15-30 days',
        'stress_Low', 'stress_Medium', 'stress_High',
        'mental_health_history_No', 'mental_health_history_Yes',
        'work_interest_Same', 'work_interest_More', 'work_interest_Less',
        'social_anxiety_Never', 'social_anxiety_Sometimes',
        'consult_history_No', 'consult_history_Yes',
        'mood_swings_Low', 'mood_swings_Medium', 'mood_swings_High'
    ]


@pytest.fixture
def mock_target_names():
    """Mock target names for testing."""
    return ['care_options_No', 'care_options_Maybe', 'care_options_Yes']


@pytest.fixture
def mock_model_loader(mock_model, mock_encoder, mock_feature_names, mock_target_names):
    """Mock model loader for testing."""
    loader = MagicMock(spec=ModelLoader)
    loader.get_model.return_value = mock_model
    loader.get_encoder.return_value = mock_encoder
    loader.get_feature_names.return_value = mock_feature_names
    loader.get_target_names.return_value = mock_target_names
    return loader


@pytest.fixture
def mock_mongodb():
    """Mock MongoDB for testing."""
    mock_db = AsyncMock()
    mock_collection = AsyncMock()
    mock_collection.insert_one.return_value = AsyncMock(inserted_id="test_id_123")
    mock_collection.find.return_value = AsyncMock()
    mock_collection.find.return_value.sort.return_value = AsyncMock()
    mock_collection.find.return_value.sort.return_value.limit.return_value = AsyncMock()
    mock_collection.find.return_value.sort.return_value.limit.return_value.to_list.return_value = []
    
    mock_db.get_care_predictions_collection.return_value = mock_collection
    return mock_db


@pytest.fixture
def client(mock_model_loader, mock_mongodb):
    """Test client with mocked dependencies."""
    # Mock the app state
    app.state.model_loader = mock_model_loader
    
    # Mock MongoDB
    mongodb.mongodb = mock_mongodb
    
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def sample_request_data():
    """Sample valid request data for testing."""
    return {
        "user_id": "test_user_123",
        "gender": 1,
        "country": "USA",
        "occupation": "Student",
        "family_history": 0,
        "treatment": 1,
        "days_indoors": "1-14 days",
        "stress": "Low",
        "mental_health_history": "No",
        "mood_swings": "Medium",
        "coping_struggles": 0,
        "work_interest": "Same",
        "social_anxiety": "Never",
        "consult_history": "No"
    }
