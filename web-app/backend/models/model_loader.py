"""
Model Loader Module

Loads and manages trained ML models for inference.
Handles both Keras (ANN) and Scikit-learn (Random Forest) models.
"""

import os
import pickle
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow import keras


class ModelLoader:
    """
    Loads and manages trained machine learning models.
    """
    
    def __init__(self, config):
        """
        Initialize model loader with configuration.
        
        Args:
            config: Configuration object with model paths
        """
        self.config = config
        self.ann_model = None
        self.rf_model = None
        self.scaler = None
        self.models_loaded = False
        
    def load_models(self):
        """
        Load all trained models (ANN and Random Forest).
        
        Returns:
            bool: True if models loaded successfully
        """
        print("\n" + "="*80)
        print("LOADING TRAINED MODELS")
        print("="*80)
        
        try:
            # Load Regularized ANN Model
            print("\n[1/3] Loading Regularized ANN model...")
            ann_path = self._find_model_file('regularized', ['.h5', '.keras'])
            
            if ann_path:
                self.ann_model = keras.models.load_model(ann_path)
                print(f"[OK] ANN model loaded from: {ann_path}")
                print(f"  Architecture: {len(self.ann_model.layers)} layers")
            else:
                print("[WARN] ANN model not found")
                
            # Load Random Forest Model
            print("\n[2/3] Loading Random Forest model...")
            rf_path = self._find_model_file('random_forest', ['.pkl', '.joblib'])
            
            if rf_path:
                with open(rf_path, 'rb') as f:
                    self.rf_model = pickle.load(f)
                print(f"[OK] Random Forest model loaded from: {rf_path}")
                print(f"  Estimators: {self.rf_model.n_estimators}")
            else:
                print("[WARN] Random Forest model not found")
            
            # Load Scaler (if exists)
            print("\n[3/3] Loading feature scaler...")
            scaler_path = self._find_model_file('scaler', ['.pkl', '.joblib'])
            
            if scaler_path:
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                print(f"[OK] Scaler loaded from: {scaler_path}")
            else:
                print("[INFO] Scaler not found - will use raw features")
                
            # Check if at least one model is loaded
            if self.ann_model is None and self.rf_model is None:
                print("\n[ERROR] No models could be loaded!")
                return False
                
            self.models_loaded = True
            print("\n" + "="*80)
            print("MODEL LOADING COMPLETE")
            print("="*80 + "\n")
            return True
            
        except Exception as e:
            print(f"\n[ERROR] Failed loading models: {str(e)}")
            return False
    
    def _find_model_file(self, keyword, extensions):
        """
        Find model file by keyword and extensions.
        
        Args:
            keyword: Keyword to search in filename
            extensions: List of file extensions to try
            
        Returns:
            str: Path to model file or None
        """
        # Check multiple possible locations
        search_dirs = [
            self.config.MODELS_DIR,
            self.config.RESULTS_DIR,
            self.config.PROJECT_ROOT / 'src'
        ]
        
        for directory in search_dirs:
            if directory.exists():
                for file in directory.iterdir():
                    if keyword.lower() in file.name.lower():
                        if any(file.name.endswith(ext) for ext in extensions):
                            return str(file)
        return None
    
    def predict_ann(self, features):
        """
        Make prediction using ANN model.
        
        Args:
            features: numpy array of input features
            
        Returns:
            dict: Prediction results
        """
        if self.ann_model is None:
            return {
                'error': 'ANN model not loaded',
                'prediction': None,
                'probability': None
            }
        
        try:
            # Preprocess features
            X = self._preprocess_features(features)
            
            # Make prediction
            probability = float(self.ann_model.predict(X, verbose=0)[0][0])
            
            # Use optimal threshold from ROC analysis
            optimal_threshold = self.config.OPTIMAL_THRESHOLDS['regularized_ann']['threshold']
            prediction = 'HIGH_RISK' if probability >= optimal_threshold else 'LOW_RISK'
            
            return {
                'prediction': prediction,
                'probability': probability,
                'confidence': f"{probability*100:.2f}%",
                'threshold_used': optimal_threshold
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'prediction': None,
                'probability': None
            }
    
    def predict_rf(self, features):
        """
        Make prediction using Random Forest model.
        
        Args:
            features: numpy array of input features
            
        Returns:
            dict: Prediction results
        """
        if self.rf_model is None:
            return {
                'error': 'Random Forest model not loaded',
                'prediction': None,
                'probability': None
            }
        
        try:
            # Preprocess features (apply scaling like training!)
            X = self._preprocess_features(features)
            
            # Make prediction
            probability = float(self.rf_model.predict_proba(X)[0][1])
            
            # Use optimal threshold from ROC analysis
            optimal_threshold = self.config.OPTIMAL_THRESHOLDS['random_forest']['threshold']
            prediction = 'HIGH_RISK' if probability >= optimal_threshold else 'LOW_RISK'
            
            return {
                'prediction': prediction,
                'probability': probability,
                'confidence': f"{probability*100:.2f}%",
                'threshold_used': optimal_threshold
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'prediction': None,
                'probability': None
            }
    
    def predict_all(self, features):
        """
        Make predictions using all available models.
        
        Args:
            features: numpy array of input features
            
        Returns:
            dict: Predictions from all models
        """
        results = {}
        
        if self.ann_model is not None:
            results['regularized_ann'] = self.predict_ann(features)
            
        if self.rf_model is not None:
            results['random_forest'] = self.predict_rf(features)
            
        return results
    
    def _preprocess_features(self, features):
        """
        Preprocess features for model input.
        
        Args:
            features: Raw feature array
            
        Returns:
            numpy array: Preprocessed features
        """
        # Reshape to 2D if needed
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        
        # Apply scaling if scaler is available
        if self.scaler is not None:
            features = self.scaler.transform(features)
        
        return features
    
    def get_model_info(self):
        """
        Get information about loaded models.
        
        Returns:
            dict: Model information
        """
        info = {
            'ann_loaded': self.ann_model is not None,
            'rf_loaded': self.rf_model is not None,
            'scaler_loaded': self.scaler is not None,
            'models_ready': self.models_loaded
        }
        
        if self.ann_model is not None:
            info['ann_info'] = {
                'layers': len(self.ann_model.layers),
                'parameters': self.ann_model.count_params(),
                'input_shape': self.ann_model.input_shape
            }
        
        if self.rf_model is not None:
            info['rf_info'] = {
                'n_estimators': self.rf_model.n_estimators,
                'max_depth': self.rf_model.max_depth,
                'n_features': self.rf_model.n_features_in_
            }
        
        return info


# Standalone testing
if __name__ == '__main__':
    import sys
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from config import Config
    
    print("Testing Model Loader...")
    
    loader = ModelLoader(Config)
    success = loader.load_models()
    
    if success:
        print("\n[OK] Model loader test successful!")
        print("\nModel Info:")
        import json
        print(json.dumps(loader.get_model_info(), indent=2))
    else:
        print("\n[FAIL] Model loader test failed!")

