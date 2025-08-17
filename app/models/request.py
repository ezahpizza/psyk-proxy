"""
Pydantic models for API request validation.
"""
from pydantic import BaseModel, Field, ConfigDict


class PredictRequest(BaseModel):
    """Request model for prediction endpoint."""
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "user_id": "user_1",
                "gender": 1,
                "country": "United States",
                "occupation": " Corporate",
                "family_history": 0,
                "treatment": 1,
                "days_indoors": "1-14 days",
                "stress": "Yes",
                "mental_health_history": "No",
                "mood_swings": "Medium",
                "coping_struggles": 1,
                "work_interest": "No",
                "social_anxiety": "Yes",
                "consult_history": "Yes"
            }
        }
    )
    
    user_id: str = Field(..., description="Clerk user ID")
    gender: int = Field(..., ge=0, le=1, description="Gender (0=Male, 1=Female)")
    country: str = Field(..., description="Country name")
    occupation: str = Field(..., description="Occupation")
    family_history: int = Field(..., ge=0, le=1, description="Family history of mental health (0=No, 1=Yes)")
    treatment: int = Field(..., ge=0, le=1, description="Currently seeking treatment (0=No, 1=Yes)")
    days_indoors: str = Field(..., description="Days spent indoors")
    stress: str = Field(..., description="Stress level")
    mental_health_history: str = Field(..., description="Mental health history")
    mood_swings: str = Field(..., description="Mood swings frequency")
    coping_struggles: int = Field(..., ge=0, le=1, description="Coping struggles (0=No, 1=Yes)")
    work_interest: str = Field(..., description="Work interest level")
    social_anxiety: str = Field(..., description="Social anxiety level")
    consult_history: str = Field(..., description="Mental health consultation history")
