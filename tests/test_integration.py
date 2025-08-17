"""
Integration tests for the complete API workflow.
"""
from unittest.mock import patch, AsyncMock

from fastapi import status


class TestIntegration:
    """Integration tests for end-to-end API functionality."""
    
    @patch("app.db.mongodb.mongodb.get_care_predictions_collection")
    def test_complete_prediction_workflow(self, mock_collection, client, sample_request_data):
        """Test complete prediction workflow from request to response."""
        # Mock database collection
        mock_collection.return_value.insert_one = AsyncMock(
            return_value=AsyncMock(inserted_id="test_prediction_id")
        )
        
        # Make prediction request
        response = client.post("/predict", json=sample_request_data)
        
        # Verify response
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
        assert isinstance(data["prediction"], str)
        assert isinstance(data["confidence"], (int, float))
        assert 0.0 <= data["confidence"] <= 1.0
    
    def test_error_handling_chain(self, client):
        """Test error handling throughout the application."""
        # Test validation error
        invalid_data = {"invalid": "data"}
        response = client.post("/predict", json=invalid_data)
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
        
        # Test 404 for non-existent endpoint
        response = client.get("/nonexistent")
        assert response.status_code == status.HTTP_404_NOT_FOUND
    
    @patch("app.services.prediction.PredictionService.predict")
    def test_prediction_service_error_handling(self, mock_predict, client, sample_request_data):
        """Test handling of prediction service errors."""
        # Mock prediction service to raise an error
        mock_predict.side_effect = ValueError("Prediction failed")
        
        response = client.post("/predict", json=sample_request_data)
        
        assert response.status_code == status.HTTP_400_BAD_REQUEST
        error_data = response.json()
        assert "detail" in error_data
    
    def test_health_check_integration(self, client):
        """Test health check endpoint integration."""
        response = client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "psyk-proxy"
    
    def test_request_response_content_types(self, client, sample_request_data):
        """Test that request and response content types are handled correctly."""
        # Test JSON request
        response = client.post(
            "/predict",
            json=sample_request_data,
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        assert response.headers["content-type"] == "application/json"
    
    def test_large_request_handling(self, client):
        """Test handling of requests with many fields."""
        large_request = {
            "user_id": "user_with_long_id_" + "x" * 100,
            "gender": 1,
            "country": "United States of America",
            "occupation": "Software Engineer and Data Scientist",
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
        
        response = client.post("/predict", json=large_request)
        
        # Should handle large requests gracefully
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_422_UNPROCESSABLE_ENTITY]


class TestPerformance:
    """Basic performance tests."""
    
    def test_prediction_response_time(self, client, sample_request_data):
        """Test that prediction responses are reasonably fast."""
        import time
        
        start_time = time.time()
        response = client.post("/predict", json=sample_request_data)
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # Response should be faster than 5 seconds (very generous for testing)
        assert response_time < 5.0
        assert response.status_code == status.HTTP_200_OK
    
    def test_health_check_response_time(self, client):
        """Test that health check is fast."""
        import time
        
        start_time = time.time()
        response = client.get("/health")
        end_time = time.time()
        
        response_time = end_time - start_time
        
        # Health check should be very fast
        assert response_time < 1.0
        assert response.status_code == status.HTTP_200_OK
