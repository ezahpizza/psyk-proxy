"""
Test cases for model loader.
"""
import pytest
from unittest.mock import patch, MagicMock

from app.core.model_loader import ModelLoader


class TestModelLoader:
    """Test cases for ModelLoader."""
    
    @patch('app.core.model_loader.joblib.load')
    @patch('app.core.model_loader.os.path.exists')
    def test_model_loader_initialization_success(self, mock_exists, mock_joblib_load):
        """Test successful model loader initialization."""
        # Mock file existence
        mock_exists.return_value = True
        
        # Mock joblib.load returns
        mock_model = MagicMock()
        mock_encoder = MagicMock()
        mock_features = ['feature1', 'feature2', 'feature3']
        mock_targets = ['target1', 'target2', 'target3']
        mock_package = {'performance': {'accuracy': 0.95}}
        
        mock_joblib_load.side_effect = [
            mock_model,    # model.pkl
            mock_encoder,  # encoder.pkl
            mock_features, # features.pkl
            mock_targets,  # targets.pkl
            mock_package   # model_package.pkl
        ]
        
        loader = ModelLoader()
        
        # Verify all components are loaded
        assert loader.get_model() == mock_model
        assert loader.get_encoder() == mock_encoder
        assert loader.get_feature_names() == mock_features
        assert loader.get_target_names() == mock_targets
        
        # Verify joblib.load was called for each file
        assert mock_joblib_load.call_count == 5
    
    @patch('app.core.model_loader.joblib.load')
    @patch('app.core.model_loader.os.path.exists')
    def test_model_loader_without_package(self, mock_exists, mock_joblib_load):
        """Test model loader when model_package.pkl doesn't exist."""
        # Mock file existence - package file doesn't exist
        def mock_exists_side_effect(path):
            return not path.endswith('model_package.pkl')
        
        mock_exists.side_effect = mock_exists_side_effect
        
        # Mock joblib.load returns for required files only
        mock_model = MagicMock()
        mock_encoder = MagicMock()
        mock_features = ['feature1', 'feature2']
        mock_targets = ['target1', 'target2']
        
        mock_joblib_load.side_effect = [
            mock_model,    # model.pkl
            mock_encoder,  # encoder.pkl
            mock_features, # features.pkl
            mock_targets   # targets.pkl
        ]
        
        loader = ModelLoader()
        
        # Verify required components are loaded
        assert loader.get_model() == mock_model
        assert loader.get_encoder() == mock_encoder
        assert loader.get_feature_names() == mock_features
        assert loader.get_target_names() == mock_targets
        
        # Verify performance metrics return empty dict
        assert loader.get_performance_metrics() == {}
    
    @patch('app.core.model_loader.joblib.load')
    def test_model_loader_file_not_found(self, mock_joblib_load):
        """Test model loader when required files are missing."""
        # Mock joblib.load to raise FileNotFoundError
        mock_joblib_load.side_effect = FileNotFoundError("File not found")
        
        with pytest.raises(FileNotFoundError):
            ModelLoader()
    
    @patch('app.core.model_loader.joblib.load')
    def test_model_loader_loading_error(self, mock_joblib_load):
        """Test model loader when joblib loading fails."""
        # Mock joblib.load to raise an exception
        mock_joblib_load.side_effect = Exception("Loading failed")
        
        with pytest.raises(Exception):
            ModelLoader()
    
    @patch('app.core.model_loader.joblib.load')
    @patch('app.core.model_loader.os.path.exists')
    def test_get_performance_metrics_with_package(self, mock_exists, mock_joblib_load):
        """Test getting performance metrics when package is available."""
        mock_exists.return_value = True
        
        mock_performance = {
            'roc_auc_scores': {'target1': 0.95, 'target2': 0.88},
            'accuracy_scores': {'target1': 0.92, 'target2': 0.85}
        }
        mock_package = {'performance': mock_performance}
        
        mock_joblib_load.side_effect = [
            MagicMock(),     # model.pkl
            MagicMock(),     # encoder.pkl
            ['feature1'],    # features.pkl
            ['target1'],     # targets.pkl
            mock_package     # model_package.pkl
        ]
        
        loader = ModelLoader()
        
        metrics = loader.get_performance_metrics()
        assert metrics == mock_performance
    
    @patch('app.core.model_loader.joblib.load')
    @patch('app.core.model_loader.os.path.exists')
    def test_feature_and_target_names_types(self, mock_exists, mock_joblib_load):
        """Test that feature and target names are returned as lists."""
        mock_exists.return_value = False  # No package file
        
        mock_features = ['gender', 'country', 'occupation_Student']
        mock_targets = ['care_options_Yes', 'care_options_No']
        
        mock_joblib_load.side_effect = [
            MagicMock(),     # model.pkl
            MagicMock(),     # encoder.pkl
            mock_features,   # features.pkl
            mock_targets     # targets.pkl
        ]
        
        loader = ModelLoader()
        
        features = loader.get_feature_names()
        targets = loader.get_target_names()
        
        assert isinstance(features, list)
        assert isinstance(targets, list)
        assert features == mock_features
        assert targets == mock_targets
    
    @patch('app.core.model_loader.settings')
    @patch('app.core.model_loader.joblib.load')
    @patch('app.core.model_loader.os.path.exists')
    def test_model_path_configuration(self, mock_exists, mock_joblib_load, mock_settings):
        """Test that model path is correctly used from settings."""
        mock_exists.return_value = False
        mock_settings.MODEL_PATH = "custom/model/path"
        
        mock_joblib_load.side_effect = [
            MagicMock(),  # model.pkl
            MagicMock(),  # encoder.pkl
            [],          # features.pkl
            []           # targets.pkl
        ]
        
        loader = ModelLoader()
        
        # Verify that os.path.join was called with the custom path
        expected_calls = [
            "custom/model/path/model.pkl",
            "custom/model/path/encoder.pkl",
            "custom/model/path/features.pkl",
            "custom/model/path/targets.pkl"
        ]
        
        # Check that joblib.load was called with paths containing the custom model path
        for call_args in mock_joblib_load.call_args_list:
            path = call_args[0][0]
            # Handle both Windows and Unix path separators
            normalized_path = path.replace("\\", "/")
            assert normalized_path.startswith("custom/model/path/")
