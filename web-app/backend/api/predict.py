"""
Prediction API Endpoints

Handles prediction requests and returns results from ML models.
"""

from flask import Blueprint, request, jsonify
from datetime import datetime
import numpy as np

from ..utils.preprocessing import InputValidator, FeatureEncoder, RiskAssessor

# Create blueprint
predict_bp = Blueprint('predict', __name__)

# Global model loader (will be set by app.py)
model_loader = None


def init_predict_api(loader):
    """
    Initialize prediction API with model loader.
    
    Args:
        loader: ModelLoader instance
    """
    global model_loader
    model_loader = loader


@predict_bp.route('/predict', methods=['POST'])
def predict():
    """
    Main prediction endpoint.
    
    Expects JSON payload with patient features.
    Returns predictions from both ANN and Random Forest models.
    """
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({
                'success': False,
                'error': 'No data provided'
            }), 400
        
        # Validate input
        is_valid, errors = InputValidator.validate_all_features(data)
        
        if not is_valid:
            return jsonify({
                'success': False,
                'error': 'Invalid input data',
                'details': errors
            }), 400
        
        # Encode features
        features = FeatureEncoder.encode_all_features(data)
        
        # Make predictions
        predictions = model_loader.predict_all(features)
        
        # Check for errors in predictions
        for model_name, result in predictions.items():
            if 'error' in result:
                return jsonify({
                    'success': False,
                    'error': f'Prediction failed for {model_name}',
                    'details': result['error']
                }), 500
        
        # Get recommendation
        recommendation = RiskAssessor.get_recommendation(predictions)
        
        # Prepare response
        response = {
            'success': True,
            'predictions': predictions,
            'recommendation': recommendation,
            'input_summary': {
                'age': int(data['age']),
                'gender': data['gender'],
                'high_risk_factors': _count_risk_factors(data)
            },
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': 'Internal server error',
            'details': str(e)
        }), 500


@predict_bp.route('/predict/batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction endpoint for multiple patients.
    
    Expects JSON array of patient data.
    """
    try:
        data = request.get_json()
        
        if not isinstance(data, list):
            return jsonify({
                'success': False,
                'error': 'Expected array of patient data'
            }), 400
        
        if len(data) > 100:
            return jsonify({
                'success': False,
                'error': 'Maximum batch size is 100 patients'
            }), 400
        
        results = []
        
        for i, patient_data in enumerate(data):
            # Validate
            is_valid, errors = InputValidator.validate_all_features(patient_data)
            
            if not is_valid:
                results.append({
                    'index': i,
                    'success': False,
                    'errors': errors
                })
                continue
            
            # Encode and predict
            features = FeatureEncoder.encode_all_features(patient_data)
            predictions = model_loader.predict_all(features)
            
            results.append({
                'index': i,
                'success': True,
                'predictions': predictions,
                'recommendation': RiskAssessor.get_recommendation(predictions)
            })
        
        return jsonify({
            'success': True,
            'total': len(data),
            'results': results,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@predict_bp.route('/models/info', methods=['GET'])
def get_model_info():
    """
    Get information about loaded models and dataset.
    """
    try:
        # Get model info from loader
        loader_info = model_loader.get_model_info()
        
        # Get configuration info
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).parent.parent.parent))
        from config import Config
        
        response = {
            'success': True,
            'models': {
                'regularized_ann': {
                    **Config.MODEL_INFO['regularized_ann'],
                    'loaded': loader_info.get('ann_loaded', False)
                },
                'random_forest': {
                    **Config.MODEL_INFO['random_forest'],
                    'loaded': loader_info.get('rf_loaded', False)
                }
            },
            'dataset': Config.DATASET_INFO,
            'features': {
                'count': len(Config.FEATURE_NAMES),
                'names': Config.FEATURE_NAMES,
                'descriptions': Config.FEATURE_DESCRIPTIONS
            },
            'status': 'ready' if loader_info['models_ready'] else 'not_ready'
        }
        
        return jsonify(response), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@predict_bp.route('/health', methods=['GET'])
def health_check():
    """
    Health check endpoint for monitoring.
    """
    try:
        status = 'healthy' if model_loader and model_loader.models_loaded else 'unhealthy'
        
        return jsonify({
            'status': status,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'models_loaded': model_loader.models_loaded if model_loader else False
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500


def _count_risk_factors(data):
    """
    Count number of positive risk factors.
    
    Args:
        data: Patient data dictionary
        
    Returns:
        int: Number of risk factors present
    """
    risk_factors = [
        'smoking', 'yellow_fingers', 'anxiety', 'chronic_disease',
        'fatigue', 'wheezing', 'coughing', 'shortness_of_breath',
        'swallowing_difficulty', 'chest_pain'
    ]
    
    count = 0
    for factor in risk_factors:
        if factor in data and str(data[factor]) in ['2', 'Yes', 'YES']:
            count += 1
    
    return count

