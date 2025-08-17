"""
Test cases for API endpoints.
"""
from fastapi import status
from unittest.mock import patch, AsyncMock


class TestPredictEndpoint:
    """Test cases for the /predict endpoint."""
    
    def test_predict_success(self, client, sample_request_data):
        """Test successful prediction."""
        response = client.post("/predict", json=sample_request_data)
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
        assert isinstance(data["confidence"], (int, float))
        assert 0.0 <= data["confidence"] <= 1.0
        
        # Check that prediction is one of the expected target names
        expected_predictions = ["care_options_No", "care_options_Maybe", "care_options_Yes"]
        assert data["prediction"] in expected_predictions
    
    def test_predict_missing_required_field(self, client, sample_request_data):
        """Test prediction with missing required field."""
        # Remove a required field
        invalid_data = sample_request_data.copy()
        del invalid_data["user_id"]
        
        response = client.post("/predict", json=invalid_data)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_predict_invalid_gender(self, client, sample_request_data):
        """Test prediction with invalid gender value."""
        invalid_data = sample_request_data.copy()
        invalid_data["gender"] = 2  # Should be 0 or 1
        
        response = client.post("/predict", json=invalid_data)
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_predict_invalid_confidence_range(self, client, sample_request_data):
        """Test prediction with confidence values in valid range."""
        response = client.post("/predict", json=sample_request_data)
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert 0.0 <= data["confidence"] <= 1.0
    
    def test_predict_empty_body(self, client):
        """Test prediction with empty request body."""
        response = client.post("/predict", json={})
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY
    
    def test_predict_invalid_json(self, client):
        """Test prediction with invalid JSON."""
        response = client.post(
            "/predict", 
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

class TestPredictHistoryEndpoint:
    """Test cases for the /predict/history/{user_id} endpoint."""
    
    def test_get_history_success(self, client):
        """Test successful retrieval of prediction history."""
        user_id = "test_user_123"
        response = client.get(f"/predict/history/{user_id}")
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert "user_id" in data
        assert "prediction_count" in data
        assert "predictions" in data
        assert data["user_id"] == user_id
        assert isinstance(data["predictions"], list)
    
    def test_get_history_empty_user_id(self, client):
        """Test history endpoint with empty user_id."""
        response = client.get("/predict/history/")
        
        # Should return 404 as the path doesn't match
        assert response.status_code == status.HTTP_404_NOT_FOUND

class TestHealthEndpoint:
    """Test cases for the health check endpoint."""
    
    def test_health_check(self, client):
        """Test health check endpoint."""
        response = client.get("/health")
        
        assert response.status_code == status.HTTP_200_OK
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "psyk-proxy"


class TestAPIDocumentation:
    """Test cases for API documentation endpoints."""
    
    def test_openapi_docs(self, client):
        """Test that OpenAPI docs are accessible."""
        response = client.get("/docs")
        assert response.status_code == status.HTTP_200_OK
    
    def test_redoc_docs(self, client):
        """Test that ReDoc docs are accessible."""
        response = client.get("/redoc")
        assert response.status_code == status.HTTP_200_OK
    
    def test_openapi_json(self, client):
        """Test that OpenAPI JSON schema is accessible."""
        response = client.get("/openapi.json")
        assert response.status_code == status.HTTP_200_OK
        
        # Verify it's valid JSON
        data = response.json()
        assert "openapi" in data
        assert "info" in data
        assert "paths" in data
