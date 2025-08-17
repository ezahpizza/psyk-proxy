"""
Model inference service for making predictions.
"""
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from typing import Dict, Any, Tuple

from app.services.preprocessing import PreprocessingService

logger = logging.getLogger(__name__)


class PredictionService:
    """Service for handling model inference and prediction logic."""
    
    def __init__(self, model, encoder, feature_names, target_names):
        """
        Initialize prediction service.
        
        Args:
            model: Trained ML model
            encoder: LeaveOneOutEncoder for preprocessing
            feature_names: List of feature names
            target_names: List of target names
        """
        self.model = model
        self.target_names = target_names
        self.preprocessing_service = PreprocessingService(encoder, feature_names)
        
    async def predict(self, input_data: Dict[str, Any]) -> Tuple[str, float]:
        """
        Make prediction for input data.
        
        Args:
            input_data: Dictionary containing user input data
            
        Returns:
            Tuple of (prediction_label, confidence_score)
        """
        try:
            # Validate categorical values
            self.preprocessing_service.validate_categorical_values(input_data)
            
            # Preprocess input data
            processed_data = self.preprocessing_service.preprocess_input(input_data)

            X = pd.DataFrame(
                processed_data,
                columns=self.preprocessing_service.feature_names,
            )
            
            probabilities = self.model.predict_proba(X)
            
            if isinstance(probabilities, list) and len(probabilities) > 0:
                target_probabilities = []
                for target_idx, target_probs in enumerate(probabilities):
                    if target_probs.shape[1] > 1:
                        positive_prob = target_probs[0, 1]  # Probability of class 1
                    else:
                        positive_prob = target_probs[0, 0]  # Only one class
                    target_probabilities.append(positive_prob)
                
                best_target_idx = np.argmax(target_probabilities)
                best_prediction = self.target_names[best_target_idx]
                best_confidence = float(target_probabilities[best_target_idx])
                
            else:
                # Fallback for single output
                if probabilities.shape[1] > 1:
                    best_target_idx = np.argmax(probabilities[0])
                    best_prediction = self.target_names[best_target_idx] if best_target_idx < len(self.target_names) else "unknown"
                    best_confidence = float(np.max(probabilities[0]))
                else:
                    best_prediction = self.target_names[0] if self.target_names else "unknown"
                    best_confidence = float(probabilities[0, 0])
            
            logger.info(f"Prediction: {best_prediction}, Confidence: {best_confidence:.3f}")
            return best_prediction, best_confidence
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            raise ValueError(f"Failed to make prediction: {str(e)}")
    
    def create_prediction_record(self, user_id: str, input_data: Dict[str, Any], 
                               prediction: str, confidence: float) -> Dict[str, Any]:
        """
        Create a prediction record for database storage.
        
        Args:
            user_id: User ID
            input_data: Original input data
            prediction: Prediction result
            confidence: Confidence score
            
        Returns:
            Dictionary containing the prediction record
        """
        return {
            "user_id": user_id,
            "input_data": input_data,
            "prediction": prediction,
            "confidence": confidence,
            "timestamp": datetime.now(timezone.utc)
        }
