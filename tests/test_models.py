"""
Test cases for Pydantic models.
"""
import pytest
from pydantic import ValidationError

from app.models.request import PredictRequest
from app.models.response import PredictResponse, ErrorResponse


class TestPredictRequest:
    """Test cases for PredictRequest model."""
    
    def test_valid_request(self):
        """Test valid request data."""
        valid_data = {
            "user_id": "user_123",
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
        
        request = PredictRequest(**valid_data)
        
        assert request.user_id == "user_123"
        assert request.gender == 1
        assert request.country == "USA"
        assert request.occupation == "Student"
    
    def test_missing_required_field(self):
        """Test request with missing required field."""
        incomplete_data = {
            "gender": 1,
            "country": "USA"
            # Missing user_id and other required fields
        }
        
        with pytest.raises(ValidationError) as exc_info:
            PredictRequest(**incomplete_data)
        
        errors = exc_info.value.errors()
        missing_fields = [error['loc'][0] for error in errors if error['type'] == 'missing']
        assert 'user_id' in missing_fields
    
    def test_invalid_gender_value(self):
        """Test request with invalid gender value."""
        invalid_data = {
            "user_id": "user_123",
            "gender": 2,  # Should be 0 or 1
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
        
        with pytest.raises(ValidationError) as exc_info:
            PredictRequest(**invalid_data)
        
        errors = exc_info.value.errors()
        gender_errors = [error for error in errors if error['loc'][0] == 'gender']
        assert len(gender_errors) > 0
    
    def test_invalid_family_history_value(self):
        """Test request with invalid family_history value."""
        invalid_data = {
            "user_id": "user_123",
            "gender": 1,
            "country": "USA",
            "occupation": "Student",
            "family_history": -1,  # Should be 0 or 1
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
        
        with pytest.raises(ValidationError):
            PredictRequest(**invalid_data)
    
    def test_empty_string_fields(self):
        """Test request with empty string fields."""
        data_with_empty_strings = {
            "user_id": "",  # Empty string
            "gender": 1,
            "country": "",  # Empty string
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
        
        # Should accept empty strings (validation logic can be added later if needed)
        request = PredictRequest(**data_with_empty_strings)
        assert request.user_id == ""
        assert request.country == ""


class TestPredictResponse:
    """Test cases for PredictResponse model."""
    
    def test_valid_response(self):
        """Test valid response data."""
        valid_data = {
            "prediction": "care_options_Yes",
            "confidence": 0.91
        }
        
        response = PredictResponse(**valid_data)
        
        assert response.prediction == "care_options_Yes"
        assert response.confidence == 0.91
    
    def test_confidence_range_validation(self):
        """Test confidence value range validation."""
        # Test valid confidence values
        valid_confidences = [0.0, 0.5, 1.0, 0.999]
        
        for confidence in valid_confidences:
            response = PredictResponse(prediction="care_options_Yes", confidence=confidence)
            assert response.confidence == confidence
        
        # Test invalid confidence values
        invalid_confidences = [-0.1, 1.1, -1.0, 2.0]
        
        for confidence in invalid_confidences:
            with pytest.raises(ValidationError):
                PredictResponse(prediction="care_options_Yes", confidence=confidence)
    
    def test_missing_required_fields(self):
        """Test response with missing required fields."""
        with pytest.raises(ValidationError):
            PredictResponse(prediction="care_options_Yes")  # Missing confidence
        
        with pytest.raises(ValidationError):
            PredictResponse(confidence=0.8)  # Missing prediction
    
    def test_response_serialization(self):
        """Test response can be serialized to dict."""
        response = PredictResponse(prediction="care_options_Maybe", confidence=0.75)
        
        response_dict = response.model_dump()
        
        assert response_dict == {
            "prediction": "care_options_Maybe",
            "confidence": 0.75
        }


class TestErrorResponse:
    """Test cases for ErrorResponse model."""
    
    def test_valid_error_response(self):
        """Test valid error response data."""
        valid_data = {
            "detail": "Invalid input data",
            "error_code": "VALIDATION_ERROR"
        }
        
        response = ErrorResponse(**valid_data)
        
        assert response.detail == "Invalid input data"
        assert response.error_code == "VALIDATION_ERROR"
    
    def test_error_response_without_code(self):
        """Test error response without error code."""
        response = ErrorResponse(detail="Something went wrong")
        
        assert response.detail == "Something went wrong"
        assert response.error_code is None
    
    def test_missing_detail_field(self):
        """Test error response with missing detail field."""
        with pytest.raises(ValidationError):
            ErrorResponse(error_code="SOME_ERROR")  # Missing detail
    
    def test_error_response_serialization(self):
        """Test error response can be serialized to dict."""
        response = ErrorResponse(
            detail="Database connection failed",
            error_code="DB_ERROR"
        )
        
        response_dict = response.model_dump()
        
        assert response_dict == {
            "detail": "Database connection failed",
            "error_code": "DB_ERROR"
        }
    
    def test_error_response_with_none_code(self):
        """Test error response with explicitly None error code."""
        response = ErrorResponse(detail="Generic error", error_code=None)
        
        assert response.detail == "Generic error"
        assert response.error_code is None
