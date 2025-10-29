"""
Verify preprocessing consistency between training and web app
"""

import numpy as np
import sys
sys.path.insert(0, '../web-app/backend/utils')

from preprocessing import FeatureEncoder

# Training preprocessing
from data_preprocessing import LungCancerDataPreprocessor

# Test with actual LOW_RISK sample from dataset
test_input = {
    'age': 59,
    'gender': 'F',
    'smoking': 1,  # NO
    'yellow_fingers': 1,  # NO
    'anxiety': 1,  # NO
    'peer_pressure': 1,  # NO
    'chronic_disease': 1,  # NO
    'fatigue': 1,  # NO
    'allergy': 1,  # NO
    'wheezing': 2,  # YES
    'alcohol_consuming': 1,  # NO
    'coughing': 2,  # YES
    'shortness_of_breath': 2,  # YES
    'swallowing_difficulty': 1,  # NO
    'chest_pain': 2  # YES
}

print("="*80)
print("PREPROCESSING COMPARISON")
print("="*80)

# Web app preprocessing
web_features = FeatureEncoder.encode_all_features(test_input)
print("\nWeb App Encoding:")
print(f"  Shape: {web_features.shape}")
print(f"  Values: {web_features}")

# Load training data to get actual encoding
preprocessor = LungCancerDataPreprocessor('../Data/survey lung cancer.csv')
X_train, X_test, y_train, y_test = preprocessor.prepare_train_test_split()

# Get first LOW_RISK sample from test set
import pandas as pd
df = pd.read_csv('../Data/survey lung cancer.csv')
low_risk_samples = df[df['LUNG_CANCER'] == 'NO']
first_sample = low_risk_samples.iloc[2]  # Sample 3 from earlier analysis

print("\n" + "="*80)
print("TRAINING DATA ENCODING (actual LOW_RISK sample)")
print("="*80)

# Manually encode like training does
from data_preprocessing import LungCancerDataPreprocessor
prep = LungCancerDataPreprocessor('../Data/survey lung cancer.csv')
df_processed = prep.preprocess_features()

# Get the same sample after preprocessing
low_risk_processed = df_processed[df_processed['LUNG_CANCER_ENCODED'] == 0]
first_processed = low_risk_processed.iloc[2]

# Get feature columns (excluding target)
feature_cols = [col for col in df_processed.columns 
                if col not in ['LUNG_CANCER', 'LUNG_CANCER_ENCODED', 'AGE']]

# Get features before scaling
features_before_scaling = first_processed[feature_cols].values

print(f"\nFeatures before scaling:")
print(f"  Shape: {features_before_scaling.shape}")
print(f"  Values: {features_before_scaling}")

# Apply scaler
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X_train)

features_after_scaling = scaler.transform(features_before_scaling.reshape(1, -1))

print(f"\nFeatures after scaling:")
print(f"  Shape: {features_after_scaling.shape}")
print(f"  Values: {features_after_scaling}")

print("\n" + "="*80)
print("COMPARISON")
print("="*80)

print(f"\nWeb App vs Training (before scaling):")
print(f"  Match: {np.allclose(web_features, features_before_scaling)}")
print(f"  Difference: {np.abs(web_features - features_before_scaling).max():.6f}")

print("\n" + "="*80)
print("KEY ISSUE")
print("="*80)
print("\n**WEB APP DOES NOT APPLY STANDARDSCALER!**")
print("Training data is scaled, but web app sends unscaled features.")
print("\nThis causes prediction mismatch.")
print("="*80)

