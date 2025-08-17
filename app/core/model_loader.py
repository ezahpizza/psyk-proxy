"""
Model loader for ML models and encoders.
"""
import os
import joblib
import logging
from typing import Dict, List, Any

from app.core.config import settings

logger = logging.getLogger(__name__)


class ModelLoader:
    """Loads and manages ML models and encoders."""
    
    def __init__(self):
        """Initialize model loader and load all required components."""
        self.model = None
        self.encoder = None
        self.feature_names = None
        self.target_names = None
        self.model_package = None
        
        self._load_models()
    
    def _load_models(self):
        """Load all model components from pickle files."""
        try:
            model_path = settings.MODEL_PATH
            
            # Load main model
            model_file = os.path.join(model_path, "model.pkl")
            self.model = joblib.load(model_file)
            logger.info("Loaded main model successfully")
            
            # Load encoder
            encoder_file = os.path.join(model_path, "encoder.pkl")
            self.encoder = joblib.load(encoder_file)
            logger.info("Loaded LeaveOneOutEncoder successfully")
            
            # Load feature names
            features_file = os.path.join(model_path, "features.pkl")
            self.feature_names = joblib.load(features_file)
            logger.info(f"Loaded {len(self.feature_names)} feature names")
            
            # Load target names
            targets_file = os.path.join(model_path, "targets.pkl")
            self.target_names = joblib.load(targets_file)
            logger.info(f"Loaded {len(self.target_names)} target names")
            
            # Load complete model package (optional, for metadata)
            package_file = os.path.join(model_path, "model_package.pkl")
            if os.path.exists(package_file):
                self.model_package = joblib.load(package_file)
                logger.info("Loaded model package with performance metrics")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
    
    def get_model(self):
        """Get the loaded model."""
        return self.model
    
    def get_encoder(self):
        """Get the loaded encoder."""
        return self.encoder
    
    def get_feature_names(self) -> List[str]:
        """Get the feature names."""
        return self.feature_names
    
    def get_target_names(self) -> List[str]:
        """Get the target names."""
        return self.target_names
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get model performance metrics if available."""
        if self.model_package and 'performance' in self.model_package:
            return self.model_package['performance']
        return {}
