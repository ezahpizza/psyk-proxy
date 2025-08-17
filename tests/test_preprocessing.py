"""
Test cases for data preprocessing service.
"""
import pytest
import numpy as np
from unittest.mock import MagicMock

from app.services.preprocessing import PreprocessingService


class TestPreprocessingService:
    """Test cases for PreprocessingService."""
    
    @pytest.fixture
    def mock_encoder(self):
        """Mock encoder for testing."""
        encoder = MagicMock()
        encoder.transform.return_value = [1.5]  # Mock encoded country value
        return encoder
    
    @pytest.fixture
    def feature_names(self):
        """Sample feature names for testing."""
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
    def preprocessing_service(self, mock_encoder, feature_names):
        """Create preprocessing service instance."""
        return PreprocessingService(mock_encoder, feature_names)
    
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
    
    def test_preprocess_input_success(self, preprocessing_service, sample_input_data, feature_names):
        """Test successful preprocessing of input data."""
        result = preprocessing_service.preprocess_input(sample_input_data)
        
        # Check output type and shape
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 1  # Single sample
        assert result.shape[1] == len(feature_names)  # All features present
    
    def test_preprocess_input_missing_categorical(self, preprocessing_service, feature_names):
        """Test preprocessing with missing categorical values."""
        minimal_data = {
            "gender": 1,
            "country": "USA",
            "family_history": 0,
            "treatment": 1,
            "coping_struggles": 0
        }
        
        result = preprocessing_service.preprocess_input(minimal_data)
        
        # Should still work and return correct shape
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 1
        assert result.shape[1] == len(feature_names)
    
    def test_preprocess_input_invalid_data(self, preprocessing_service):
        """Test preprocessing with invalid data structure."""
        invalid_data = "not a dictionary"
        
        with pytest.raises(ValueError):
            preprocessing_service.preprocess_input(invalid_data)
    
    def test_preprocess_input_empty_data(self, preprocessing_service, feature_names):
        """Test preprocessing with empty data."""
        empty_data = {}
        
        # Should raise ValueError for empty data
        with pytest.raises(ValueError, match="Input data cannot be empty"):
            preprocessing_service.preprocess_input(empty_data)
    
    def test_validate_categorical_values_valid(self, preprocessing_service, sample_input_data):
        """Test validation with valid categorical values."""
        result = preprocessing_service.validate_categorical_values(sample_input_data)
        assert result is True
    
    def test_validate_categorical_values_invalid(self, preprocessing_service):
        """Test validation with invalid categorical values."""
        invalid_data = {
            "stress": "VeryHigh",  # Not in expected values
            "mood_swings": "Extreme"  # Not in expected values
        }
        
        # Should still return True but log warnings
        result = preprocessing_service.validate_categorical_values(invalid_data)
        assert result is True
    
    def test_validate_categorical_values_missing_fields(self, preprocessing_service):
        """Test validation with missing categorical fields."""
        minimal_data = {
            "gender": 1,
            "country": "USA"
        }
        
        result = preprocessing_service.validate_categorical_values(minimal_data)
        assert result is True
    
    def test_feature_alignment(self, preprocessing_service, sample_input_data, feature_names):
        """Test that features are properly aligned to match training order."""
        result = preprocessing_service.preprocess_input(sample_input_data)
        
        # Verify shape matches expected feature count
        assert result.shape[1] == len(feature_names)
        
        # Verify all values are numeric
        assert np.all(np.isfinite(result))
    
    def test_encoder_transform_called(self, mock_encoder, feature_names, sample_input_data):
        """Test that encoder transform is called for country encoding."""
        preprocessing_service = PreprocessingService(mock_encoder, feature_names)
        
        preprocessing_service.preprocess_input(sample_input_data)
        
        # Verify encoder transform was called
        mock_encoder.transform.assert_called_once()
