"""
Pydantic models for API response formatting.
"""
from pydantic import BaseModel, Field, ConfigDict


class PredictResponse(BaseModel):
    """Response model for prediction endpoint."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prediction": "care_options_Yes",
                "confidence": 0.91
            }
        }
    )
    
    prediction: str = Field(..., description="Predicted care option")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Prediction confidence score")


class ErrorResponse(BaseModel):
    """Error response model."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "detail": "Invalid input data",
                "error_code": "VALIDATION_ERROR"
            }
        }
    )
    
    detail: str = Field(..., description="Error message")
    error_code: str | None = Field(None, description="Error code")
