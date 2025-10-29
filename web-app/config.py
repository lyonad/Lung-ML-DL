"""
Configuration file for Lung Cancer Prediction Web Application

Contains all application settings, paths, and constants.
"""

import os
from pathlib import Path

# Base directory
BASE_DIR = Path(__file__).parent
PROJECT_ROOT = BASE_DIR.parent

# Flask Configuration
class Config:
    """Base configuration"""
    DEBUG = True
    TESTING = False
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev-secret-key-change-in-production')
    
    # Base Paths
    BASE_DIR = Path(__file__).parent
    PROJECT_ROOT = BASE_DIR.parent
    
    # Server Configuration
    HOST = '0.0.0.0'
    PORT = 5000
    
    # CORS Configuration
    CORS_ORIGINS = ['http://localhost:3000', 'http://localhost:5000']
    
    # Model Paths
    MODELS_DIR = PROJECT_ROOT / 'models'
    RESULTS_DIR = PROJECT_ROOT / 'results'
    
    # Trained Models
    REGULARIZED_ANN_PATH = str(MODELS_DIR / 'regularized_ann_best.h5')
    RANDOM_FOREST_PATH = str(MODELS_DIR / 'random_forest_best.pkl')
    
    # Fallback to results directory if models directory doesn't exist
    if not MODELS_DIR.exists():
        print(f"Warning: Models directory not found at {MODELS_DIR}")
        print(f"Looking for models in results directory...")
    
    # Prediction Configuration
    # Optimal thresholds from ROC analysis (maximize Youden's J statistic)
    OPTIMAL_THRESHOLDS = {
        'regularized_ann': {
            'threshold': 0.6862,
            'sensitivity': 0.8889,
            'specificity': 1.0000
        },
        'random_forest': {
            'threshold': 0.5467,
            'sensitivity': 0.9259,
            'specificity': 1.0000
        }
    }
    
    # Legacy thresholds (for reference)
    HIGH_RISK_THRESHOLD = 0.7
    MEDIUM_RISK_THRESHOLD = 0.4
    
    # Feature Names (must match training data)
    FEATURE_NAMES = [
        'AGE',
        'GENDER',
        'SMOKING',
        'YELLOW_FINGERS',
        'ANXIETY',
        'PEER_PRESSURE',
        'CHRONIC DISEASE',
        'FATIGUE',
        'ALLERGY',
        'WHEEZING',
        'ALCOHOL CONSUMING',
        'COUGHING',
        'SHORTNESS OF BREATH',
        'SWALLOWING DIFFICULTY',
        'CHEST PAIN'
    ]
    
    # Feature Descriptions (for UI tooltips)
    FEATURE_DESCRIPTIONS = {
        'AGE': 'Patient age in years',
        'GENDER': 'Male (M) = 1, Female (F) = 0',
        'SMOKING': 'Smoking status: No = 1, Yes = 2',
        'YELLOW_FINGERS': 'Yellow fingers symptom: No = 1, Yes = 2',
        'ANXIETY': 'Anxiety symptom: No = 1, Yes = 2',
        'PEER_PRESSURE': 'Peer pressure factor: No = 1, Yes = 2',
        'CHRONIC DISEASE': 'Has chronic disease: No = 1, Yes = 2',
        'FATIGUE': 'Fatigue symptom: No = 1, Yes = 2',
        'ALLERGY': 'Allergy history: No = 1, Yes = 2',
        'WHEEZING': 'Wheezing symptom: No = 1, Yes = 2',
        'ALCOHOL CONSUMING': 'Alcohol consumption: No = 1, Yes = 2',
        'COUGHING': 'Coughing symptom: No = 1, Yes = 2',
        'SHORTNESS OF BREATH': 'Shortness of breath: No = 1, Yes = 2',
        'SWALLOWING DIFFICULTY': 'Swallowing difficulty: No = 1, Yes = 2',
        'CHEST PAIN': 'Chest pain symptom: No = 1, Yes = 2'
    }
    
    # Model Information (Actual Research Results)
    MODEL_INFO = {
        'regularized_ann': {
            'name': 'Regularized Artificial Neural Network',
            'architecture': '3 Hidden Layers with L2 Regularization + Class Weights',
            'accuracy': 0.9032,  # 90.32% - 2nd best
            'sensitivity': 0.9074,  # 90.74%
            'specificity': 0.8750,  # 87.50%
            'precision': 0.9800,  # 98.00%
            'f1_score': 0.9423,  # 94.23%
            'roc_auc': 0.9514,  # 95.14% - best AUC
            'training_samples': 247,
            'test_samples': 62
        },
        'random_forest': {
            'name': 'Random Forest Classifier',
            'architecture': '200 Estimators with Balanced Class Weights',
            'accuracy': 0.9194,  # 91.94% - BEST overall
            'sensitivity': 0.9259,  # 92.59%
            'specificity': 0.8750,  # 87.50%
            'precision': 0.9804,  # 98.04%
            'f1_score': 0.9524,  # 95.24%
            'roc_auc': 0.9444,  # 94.44%
            'training_samples': 247,
            'test_samples': 62
        }
    }
    
    # Dataset Context
    DATASET_INFO = {
        'name': 'Lung Cancer Survey - Pakistan',
        'population': 'Pakistani lung cancer patients',
        'total_samples': 309,
        'features': 15,
        'source': 'Clinical survey data',
        'year': 2025
    }
    
    # API Rate Limiting
    RATE_LIMIT = '100 per hour'
    
    # Logging
    LOG_LEVEL = 'INFO'
    LOG_FILE = str(BASE_DIR / 'app.log')


class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False


class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False
    
    # Use environment variables for sensitive data
    SECRET_KEY = os.environ.get('SECRET_KEY', 'production-secret-key-change-me')
    
    def __init__(self):
        if not os.environ.get('SECRET_KEY'):
            print("WARNING: Using default SECRET_KEY. Set SECRET_KEY environment variable in production!")


class TestingConfig(Config):
    """Testing configuration"""
    DEBUG = True
    TESTING = True
    

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}


def get_config(env='default'):
    """Get configuration based on environment"""
    return config.get(env, config['default'])

