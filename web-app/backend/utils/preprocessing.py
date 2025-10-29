"""
Data Preprocessing Utilities

Handles input validation, feature encoding, and data transformation
for web application requests.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any


class InputValidator:
    """
    Validates user input from web forms and API requests.
    """
    
    @staticmethod
    def validate_age(age: Any) -> Tuple[bool, str]:
        """
        Validate age input.
        
        Args:
            age: Age value to validate
            
        Returns:
            tuple: (is_valid, error_message)
        """
        try:
            age = int(age)
            if age < 0 or age > 120:
                return False, "Age must be between 0 and 120"
            return True, ""
        except (ValueError, TypeError):
            return False, "Age must be a valid number"
    
    @staticmethod
    def validate_gender(gender: Any) -> Tuple[bool, str]:
        """
        Validate gender input.
        
        Args:
            gender: Gender value to validate
            
        Returns:
            tuple: (is_valid, error_message)
        """
        valid_values = ['M', 'F', 'Male', 'Female', '1', '0', 1, 0]
        if gender not in valid_values:
            return False, "Gender must be 'M'/'F' or '1'/'0'"
        return True, ""
    
    @staticmethod
    def validate_binary_feature(value: Any, feature_name: str) -> Tuple[bool, str]:
        """
        Validate binary feature (Yes/No or 1/2).
        
        Args:
            value: Feature value to validate
            feature_name: Name of the feature
            
        Returns:
            tuple: (is_valid, error_message)
        """
        valid_values = [1, 2, '1', '2', 'Yes', 'No', 'YES', 'NO']
        if value not in valid_values:
            return False, f"{feature_name} must be 1 (No) or 2 (Yes)"
        return True, ""
    
    @staticmethod
    def validate_all_features(data: Dict) -> Tuple[bool, List[str]]:
        """
        Validate all input features.
        
        Args:
            data: Dictionary of input features
            
        Returns:
            tuple: (is_valid, list_of_errors)
        """
        errors = []
        
        # Required features
        required_features = [
            'age', 'gender', 'smoking', 'yellow_fingers', 'anxiety',
            'peer_pressure', 'chronic_disease', 'fatigue', 'allergy',
            'wheezing', 'alcohol_consuming', 'coughing',
            'shortness_of_breath', 'swallowing_difficulty', 'chest_pain'
        ]
        
        # Check for missing features
        for feature in required_features:
            if feature not in data:
                errors.append(f"Missing required feature: {feature}")
        
        if errors:
            return False, errors
        
        # Validate age
        is_valid, error = InputValidator.validate_age(data['age'])
        if not is_valid:
            errors.append(error)
        
        # Validate gender
        is_valid, error = InputValidator.validate_gender(data['gender'])
        if not is_valid:
            errors.append(error)
        
        # Validate binary features
        binary_features = [f for f in required_features if f not in ['age', 'gender']]
        for feature in binary_features:
            is_valid, error = InputValidator.validate_binary_feature(
                data[feature], feature
            )
            if not is_valid:
                errors.append(error)
        
        return len(errors) == 0, errors


class FeatureEncoder:
    """
    Encodes input features to match training data format.
    """
    
    @staticmethod
    def encode_gender(gender: Any) -> int:
        """
        Encode gender to numeric format.
        
        Args:
            gender: Gender value (M/F or Male/Female)
            
        Returns:
            int: Encoded gender (1 for Male, 0 for Female)
        """
        if str(gender).upper() in ['M', 'MALE', '1']:
            return 1
        return 0
    
    @staticmethod
    def encode_binary_feature(value: Any) -> int:
        """
        Encode binary feature to numeric format.
        
        Args:
            value: Feature value (Yes/No or 1/2)
            
        Returns:
            int: Encoded value (0 for No, 1 for Yes) - MATCHES TRAINING FORMAT
        """
        # Convert to match training preprocessing: NO=0, YES=1
        if str(value).upper() in ['YES', '2', 2]:
            return 1  # YES
        return 0  # NO
    
    @staticmethod
    def encode_all_features(data: Dict) -> np.ndarray:
        """
        Encode all input features to numpy array.
        MUST MATCH training preprocessing exactly!
        
        Training feature order:
        0-13: GENDER, SMOKING, ..., CHEST PAIN
        14: AGE_NORMALIZED (LAST position)
        
        Args:
            data: Dictionary of input features
            
        Returns:
            numpy array: Encoded features (matches training format)
        """
        # Constants from training data
        AGE_MIN = 21
        AGE_MAX = 87
        
        # Feature order (MUST match training data exactly!)
        feature_order = [
            'gender', 'smoking', 'yellow_fingers', 'anxiety',
            'peer_pressure', 'chronic_disease', 'fatigue', 'allergy',
            'wheezing', 'alcohol_consuming', 'coughing',
            'shortness_of_breath', 'swallowing_difficulty', 'chest_pain'
        ]
        
        encoded = []
        
        # Features 0-13: All features EXCEPT age
        for feature in feature_order:
            if feature == 'gender':
                encoded.append(FeatureEncoder.encode_gender(data[feature]))
            else:
                encoded.append(FeatureEncoder.encode_binary_feature(data[feature]))
        
        # Feature 14: AGE_NORMALIZED (LAST position, matches training!)
        age = int(data['age'])
        age_normalized = (age - AGE_MIN) / (AGE_MAX - AGE_MIN)
        encoded.append(age_normalized)
        
        return np.array(encoded, dtype=np.float32)


class RiskAssessor:
    """
    Assesses risk level and provides recommendations based on predictions.
    """
    
    @staticmethod
    def assess_risk_level(probability: float) -> str:
        """
        Determine risk level from probability.
        
        Args:
            probability: Prediction probability
            
        Returns:
            str: Risk level (LOW_RISK, MEDIUM_RISK, HIGH_RISK)
        """
        if probability >= 0.7:
            return 'HIGH_RISK'
        elif probability >= 0.4:
            return 'MEDIUM_RISK'
        else:
            return 'LOW_RISK'
    
    @staticmethod
    def get_recommendation(predictions: Dict) -> Dict:
        """
        Generate recommendation based on model predictions.
        
        Args:
            predictions: Dictionary of model predictions
            
        Returns:
            dict: Structured recommendation with risk level, confidence, message, and action
        """
        # Get probabilities from both models
        ann_prob = predictions.get('regularized_ann', {}).get('probability', 0)
        rf_prob = predictions.get('random_forest', {}).get('probability', 0)
        
        # Use average probability for confidence
        if ann_prob and rf_prob:
            avg_prob = (ann_prob + rf_prob) / 2
            max_prob = max(ann_prob, rf_prob)
        else:
            avg_prob = ann_prob or rf_prob
            max_prob = avg_prob
        
        # Determine risk level and messages
        if max_prob >= 0.8:
            return {
                'risk_level': 'High Risk',
                'confidence': avg_prob,
                'message': 'Both models indicate high risk.',
                'action': 'Please consult with a healthcare professional immediately for comprehensive screening.'
            }
        elif max_prob >= 0.6:
            return {
                'risk_level': 'Moderate-High Risk',
                'confidence': avg_prob,
                'message': 'Elevated risk factors detected.',
                'action': 'Schedule a consultation with your healthcare provider soon for thorough evaluation.'
            }
        elif max_prob >= 0.4:
            return {
                'risk_level': 'Moderate Risk',
                'confidence': avg_prob,
                'message': 'Some risk factors present.',
                'action': 'Consider routine screening and discuss lifestyle modifications with your doctor.'
            }
        else:
            return {
                'risk_level': 'Low Risk',
                'confidence': avg_prob,
                'message': 'Low risk based on current indicators.',
                'action': 'Maintain a healthy lifestyle and schedule routine check-ups.'
            }
    
    @staticmethod
    def get_risk_color(probability: float) -> str:
        """
        Get color code for risk visualization.
        
        Args:
            probability: Prediction probability
            
        Returns:
            str: Color code
        """
        if probability >= 0.7:
            return '#dc3545'  # Red
        elif probability >= 0.4:
            return '#ffc107'  # Yellow
        else:
            return '#28a745'  # Green


# Standalone testing
if __name__ == '__main__':
    print("Testing Input Validator...")
    
    # Test data
    test_data = {
        'age': 65,
        'gender': 'M',
        'smoking': 2,
        'yellow_fingers': 2,
        'anxiety': 1,
        'peer_pressure': 1,
        'chronic_disease': 2,
        'fatigue': 2,
        'allergy': 2,
        'wheezing': 2,
        'alcohol_consuming': 1,
        'coughing': 2,
        'shortness_of_breath': 2,
        'swallowing_difficulty': 2,
        'chest_pain': 2
    }
    
    # Validate
    is_valid, errors = InputValidator.validate_all_features(test_data)
    print(f"\nValidation: {'PASS' if is_valid else 'FAIL'}")
    if errors:
        print("Errors:", errors)
    
    # Encode
    if is_valid:
        features = FeatureEncoder.encode_all_features(test_data)
        print(f"\nEncoded features shape: {features.shape}")
        print(f"Feature values: {features}")
        
        # Test risk assessment
        test_predictions = {
            'regularized_ann': {'probability': 0.95},
            'random_forest': {'probability': 0.92}
        }
        recommendation = RiskAssessor.get_recommendation(test_predictions)
        print(f"\nRecommendation:")
        import json
        print(json.dumps(recommendation, indent=2))

