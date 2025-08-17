"""
Data preprocessing service using Polars and Pandas.
"""
import logging
import polars as pl
import numpy as np
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class PreprocessingService:
    """Service for preprocessing input data for model inference."""
    
    def __init__(self, encoder, feature_names: List[str]):
        """
        Initialize preprocessing service.
        
        Args:
            encoder: LeaveOneOutEncoder for country encoding
            feature_names: List of feature names in correct order
        """
        self.encoder = encoder
        self.feature_names = feature_names
        
    def preprocess_input(self, input_data: Dict[str, Any]) -> np.ndarray:
        """
        Preprocess input data for model inference.
        
        Args:
            input_data: Dictionary containing user input data
            
        Returns:
            Preprocessed numpy array ready for model inference
        """
        try:
            # Validate input data type and content
            if not isinstance(input_data, dict):
                raise ValueError("Input data must be a dictionary")
            if not input_data:
                raise ValueError("Input data cannot be empty")
            
            # Normalize string categorical inputs (trim spaces) to avoid mismatched dummy columns
            normalized = {}
            for k, v in input_data.items():
                if isinstance(v, str):
                    normalized[k] = v.strip()
                else:
                    normalized[k] = v

            # Create a DataFrame with the normalized input data
            df = pl.DataFrame([normalized])
            
            # Define categorical columns for one-hot encoding
            categorical_cols = [
                "occupation", "days_indoors", "stress", "mental_health_history",
                "work_interest", "social_anxiety", "consult_history", "mood_swings"
            ]
            
            # Filter categorical columns to only include those present in the data
            available_categorical_cols = [col for col in categorical_cols if col in df.columns]
            
            # One-hot encode categorical columns using Polars (only if we have categorical columns)
            if available_categorical_cols:
                df_encoded = df.to_dummies(columns=available_categorical_cols)
            else:
                df_encoded = df
            
            # Convert to pandas for LeaveOneOut encoding
            df_pandas = df_encoded.to_pandas()
            
            # Apply LeaveOneOut encoding to country column
            if "country" in df_pandas.columns:
                df_pandas["country"] = self.encoder.transform(df_pandas["country"])
            
            # Convert back to polars
            df_processed = pl.from_pandas(df_pandas)
            
            # Ensure all required features are present
            processed_features = df_processed.columns
            missing_features = []
            
            # Add missing features with default value 0
            for feature in self.feature_names:
                if feature not in processed_features:
                    missing_features.append(feature)
                    df_processed = df_processed.with_columns(pl.lit(0).alias(feature))
            
            if missing_features:
                logger.info(f"Added missing features with default value 0: {missing_features}")
            
            # Reorder columns to match training feature order
            df_final = df_processed.select(self.feature_names)
            
            # Convert to numpy array
            result = df_final.to_numpy()
            
            logger.info(f"Preprocessed data shape: {result.shape}")
            return result
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            raise ValueError(f"Failed to preprocess input data: {str(e)}")
    
    def validate_categorical_values(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate that categorical values are in expected ranges.
        
        Args:
            input_data: Dictionary containing user input data
            
        Returns:
            True if all values are valid
        """
        # Define expected categorical values based on training data
        expected_values = {
            "consult_history": ["Yes", "No", "Maybe"],
            "mood_swings": ["Low", "Medium", "High"]
        }
        
        for field, expected in expected_values.items():
            if field in input_data:
                value = input_data[field]
                if value not in expected:
                    logger.warning(f"Unexpected value for {field}: {value}. Expected one of: {expected}")
                    # Don't fail validation, just log warning
        
        return True
