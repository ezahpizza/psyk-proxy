"""
Test cases for prediction service.
"""
import pytest
import numpy as np
from unittest.mock import MagicMock
from datetime import datetime, timezone

from app.services.prediction import PredictionService


class TestPredictionService:
    """Test cases for PredictionService."""
    
    @pytest.fixture
    def mock_model(self):
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
    def mock_encoder(self):
        """Mock encoder for testing."""
        encoder = MagicMock()
        encoder.transform.return_value = [1.5]
        return encoder
    
    @pytest.fixture
    def feature_names(self):
        """Sample feature names."""
        return [
            'gender', 'country', 'family_history', 'treatment', 'coping_struggles',
            'occupation_Student', 'days_indoors_1-14 days', 'stress_Low',
            'mental_health_history_No', 'work_interest_Same', 'social_anxiety_Never',
            'consult_history_No', 'mood_swings_Medium'
        ]
    
    @pytest.fixture
    def target_names(self):
        """Sample target names."""
        return ['care_options_No', 'care_options_Maybe', 'care_options_Yes']
    
    @pytest.fixture
    def prediction_service(self, mock_model, mock_encoder, feature_names, target_names):
        """Create prediction service instance."""
        return PredictionService(mock_model, mock_encoder, feature_names, target_names)
    
    @pytest.fixture
    def sample_input_data(self):
        """Sample input data for testing."""
        return {
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
    
    @pytest.mark.asyncio
    async def test_predict_success(self, prediction_service, sample_input_data, target_names):
        """Test successful prediction."""
        prediction, confidence = await prediction_service.predict(sample_input_data)
        
        # Check return types
        assert isinstance(prediction, str)
        assert isinstance(confidence, float)
        
        # Check that prediction is one of the expected targets
        assert prediction in target_names
        
        # Check confidence is in valid range
        assert 0.0 <= confidence <= 1.0
    
    @pytest.mark.asyncio
    async def test_predict_selects_highest_probability(self, prediction_service, sample_input_data):
        """Test that prediction selects the target with highest probability."""
        prediction, confidence = await prediction_service.predict(sample_input_data)
        
        # Based on mock data, care_options_No should have highest probability (80%)
        assert prediction == "care_options_No"
        assert confidence == 0.8
    
    @pytest.mark.asyncio
    async def test_predict_invalid_input(self, prediction_service):
        """Test prediction with invalid input data."""
        invalid_data = {"invalid": "data"}
        
        # API should handle gracefully and return a prediction
        # (all features will be filled with defaults)
        prediction, confidence = await prediction_service.predict(invalid_data)
        
        assert isinstance(prediction, str)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
    
    @pytest.mark.asyncio
    async def test_predict_empty_input(self, prediction_service):
        """Test prediction with empty input data."""
        empty_data = {}
        
        # Should handle gracefully or raise ValueError
        try:
            prediction, confidence = await prediction_service.predict(empty_data)
            assert isinstance(prediction, str)
            assert isinstance(confidence, float)
        except ValueError:
            # This is also acceptable behavior
            pass
    
    def test_create_prediction_record(self, prediction_service, sample_input_data):
        """Test creation of prediction record for database storage."""
        user_id = "test_user_123"
        prediction = "care_options_Yes"
        confidence = 0.85
        
        record = prediction_service.create_prediction_record(
            user_id, sample_input_data, prediction, confidence
        )
        
        # Check record structure
        assert record["user_id"] == user_id
        assert record["input_data"] == sample_input_data
        assert record["prediction"] == prediction
        assert record["confidence"] == confidence
        assert "timestamp" in record
        assert isinstance(record["timestamp"], datetime)
    
    @pytest.mark.asyncio
    async def test_predict_with_single_output_model(self, mock_encoder, feature_names, target_names, sample_input_data):
        """Test prediction with single output model (fallback case)."""
        # Mock single output model
        single_output_model = MagicMock()
        single_output_model.predict_proba.return_value = np.array([[0.3, 0.7]])
        
        service = PredictionService(single_output_model, mock_encoder, feature_names, target_names)
        
        prediction, confidence = await service.predict(sample_input_data)
        
        assert isinstance(prediction, str)
        assert isinstance(confidence, float)
        assert 0.0 <= confidence <= 1.0
    
    @pytest.mark.asyncio
    async def test_predict_with_preprocessing_error(self, prediction_service):
        """Test prediction when preprocessing fails."""
        # This should trigger an error in preprocessing
        problematic_data = None
        
        with pytest.raises(ValueError):
            await prediction_service.predict(problematic_data)
    
    @pytest.mark.asyncio
    async def test_predict_with_model_error(self, mock_encoder, feature_names, target_names, sample_input_data):
        """Test prediction when model fails."""
        # Mock model that raises an error
        failing_model = MagicMock()
        failing_model.predict_proba.side_effect = Exception("Model error")
        
        service = PredictionService(failing_model, mock_encoder, feature_names, target_names)
        
        with pytest.raises(ValueError):
            await service.predict(sample_input_data)
    
    def test_prediction_record_timestamp(self, prediction_service, sample_input_data):
        """Test that prediction record includes current timestamp."""
        before_time = datetime.now(timezone.utc)
        
        record = prediction_service.create_prediction_record(
            "user_123", sample_input_data, "care_options_Yes", 0.9
        )
        
        after_time = datetime.now(timezone.utc)
        
        # Timestamp should be between before and after
        assert before_time <= record["timestamp"] <= after_time
