"""
API endpoint for mental health prediction.
"""
import logging
from fastapi import APIRouter, HTTPException, Request, status

from app.models.request import PredictRequest
from app.models.response import PredictResponse, ErrorResponse
from app.services.prediction import PredictionService
from app.db.mongodb import mongodb

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Prediction"])


@router.post(
    "/predict",
    response_model=PredictResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Bad Request"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"}
    },
    summary="Predict mental health care options",
    description="Accept user mental health data and return predicted care option with confidence score"
)
async def predict_care_options(request_data: PredictRequest, request: Request):
    """
    Predict mental health care options based on user input data.
    
    This endpoint:
    1. Validates input data using Pydantic schema
    2. Preprocesses data (one-hot encoding, LeaveOneOut encoding)
    3. Makes prediction using the trained model
    4. Stores results in MongoDB
    5. Returns prediction with confidence score
    """
    try:
        # Get model loader from app state
        model_loader = request.app.state.model_loader
        
        # Create prediction service
        prediction_service = PredictionService(
            model=model_loader.get_model(),
            encoder=model_loader.get_encoder(),
            feature_names=model_loader.get_feature_names(),
            target_names=model_loader.get_target_names()
        )
        
        input_data = request_data.model_dump(exclude={"user_id"})
        
        # Make prediction
        prediction, confidence = await prediction_service.predict(input_data)
        
        # Create prediction record
        prediction_record = prediction_service.create_prediction_record(
            user_id=request_data.user_id,
            input_data=input_data,
            prediction=prediction,
            confidence=confidence
        )
        
        # Store in MongoDB
        try:
            collection = mongodb.get_care_predictions_collection()
            result = await collection.insert_one(prediction_record)
            logger.info(f"Stored prediction record with ID: {result.inserted_id}")
            
        except Exception as db_error:
            logger.error(f"Failed to store prediction in database: {db_error}")
            
        # Return response
        response = PredictResponse(
            prediction=prediction,
            confidence=confidence
        )
        
        logger.info(f"Successfully processed prediction for user {request_data.user_id}")
        return response
        
    except ValueError as ve:
        logger.error(f"Validation error in prediction: {ve}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(ve)
        )
        
    except Exception as e:
        logger.error(f"Unexpected error in prediction endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error occurred during prediction"
        )


@router.get(
    "/predict/history/{user_id}",
    summary="Get prediction history for user",
    description="Retrieve prediction history for a specific user"
)
async def get_user_prediction_history(user_id: str):
    """Get prediction history for a specific user."""
    try:
        collection = mongodb.get_care_predictions_collection()
        
        cursor = collection.find(
            {"user_id": user_id}
        ).sort("timestamp", -1).limit(50)  
        
        # Convert cursor to list
        predictions = await cursor.to_list(length=50)
        
        for prediction in predictions:
            prediction["_id"] = str(prediction["_id"])
            
        return {
            "user_id": user_id,
            "prediction_count": len(predictions),
            "predictions": predictions
        }
        
    except Exception as e:
        logger.error(f"Error retrieving prediction history: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve prediction history"
        )
