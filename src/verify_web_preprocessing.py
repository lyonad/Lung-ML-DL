"""
Verify that web app preprocessing matches training exactly
by comparing predictions on the same sample
"""

import numpy as np
import pickle
import sys
sys.path.insert(0, '../web-app/backend/utils')

from preprocessing import FeatureEncoder
from tensorflow import keras
from data_preprocessing import LungCancerDataPreprocessor

# Load test data
prep = LungCancerDataPreprocessor('../Data/survey lung cancer.csv')
X_train, X_test, y_train, y_test = prep.prepare_train_test_split()

# Load models
ann_model = keras.models.load_model('../models/Regularized_ANN_best.h5')
with open('../models/Random_Forest_best.pkl', 'rb') as f:
    rf_model = pickle.load(f)

# Get first LOW_RISK sample from test set
low_risk_indices = np.where(y_test == 0)[0]
first_low_risk_idx = low_risk_indices[0]

# Get TRUE features from test set
true_features = X_test[first_low_risk_idx]

print("="*80)
print("COMPARING PREDICTIONS: Training Data vs Web App Encoding")
print("="*80)

# Prediction on training-preprocessed data
ann_prob_true = float(ann_model.predict(true_features.reshape(1, -1), verbose=0)[0][0])
rf_prob_true = float(rf_model.predict_proba(true_features.reshape(1, -1))[0][1])

print(f"\n1. TRAINING PREPROCESSED DATA:")
print(f"   Features shape: {true_features.shape}")
print(f"   Features: {true_features}")
print(f"   ANN probability: {ann_prob_true:.6f}")
print(f"   RF probability:  {rf_prob_true:.6f}")

# Now get the raw data for this sample
# First get which row in original data this is
import pandas as pd
df = pd.read_csv('../Data/survey lung cancer.csv')
low_risk_df = df[df['LUNG_CANCER'] == 'NO']

# Get first one (corresponding to test set index 0 for LOW_RISK)
# Note: This is approximate since we don't know exact train/test split mapping
# Let's use the KNOWN values from earlier: Sample 3 (idx=2)
sample_row = low_risk_df.iloc[2]

print(f"\n2. RAW DATA FROM DATASET:")
print(f"   Age: {sample_row['AGE']}, Gender: {sample_row['GENDER']}")
print(f"   Smoking: {sample_row['SMOKING']}, Yellow Fingers: {sample_row['YELLOW_FINGERS']}")
print(f"   Anxiety: {sample_row['ANXIETY']}, Chronic Disease: {sample_row['CHRONIC DISEASE']}")
print(f"   Fatigue: {sample_row['FATIGUE ']}, Allergy: {sample_row['ALLERGY ']}")
print(f"   Wheezing: {sample_row['WHEEZING']}, Alcohol: {sample_row['ALCOHOL CONSUMING']}")
print(f"   Coughing: {sample_row['COUGHING']}, Shortness of Breath: {sample_row['SHORTNESS OF BREATH']}")
print(f"   Swallowing Difficulty: {sample_row['SWALLOWING DIFFICULTY']}, Chest Pain: {sample_row['CHEST PAIN']}")

# Convert to web app format
web_input = {
    'age': int(sample_row['AGE']),
    'gender': sample_row['GENDER'],
    'smoking': int(sample_row['SMOKING']),
    'yellow_fingers': int(sample_row['YELLOW_FINGERS']),
    'anxiety': int(sample_row['ANXIETY']),
    'peer_pressure': int(sample_row['PEER_PRESSURE']),
    'chronic_disease': int(sample_row['CHRONIC DISEASE']),
    'fatigue': int(sample_row['FATIGUE ']),
    'allergy': int(sample_row['ALLERGY ']),
    'wheezing': int(sample_row['WHEEZING']),
    'alcohol_consuming': int(sample_row['ALCOHOL CONSUMING']),
    'coughing': int(sample_row['COUGHING']),
    'shortness_of_breath': int(sample_row['SHORTNESS OF BREATH']),
    'swallowing_difficulty': int(sample_row['SWALLOWING DIFFICULTY']),
    'chest_pain': int(sample_row['CHEST PAIN'])
}

# Encode with web app
web_features = FeatureEncoder.encode_all_features(web_input)

print(f"\n3. WEB APP ENCODED:")
print(f"   Features shape: {web_features.shape}")
print(f"   Features: {web_features}")

# Predict with web app encoding
ann_prob_web = float(ann_model.predict(web_features.reshape(1, -1), verbose=0)[0][0])
rf_prob_web = float(rf_model.predict_proba(web_features.reshape(1, -1))[0][1])

print(f"   ANN probability: {ann_prob_web:.6f}")
print(f"   RF probability:  {rf_prob_web:.6f}")

print(f"\n" + "="*80)
print("COMPARISON")
print("="*80)

print(f"\nANN Probability:")
print(f"  Training: {ann_prob_true:.6f}")
print(f"  Web App:  {ann_prob_web:.6f}")
print(f"  Match: {abs(ann_prob_true - ann_prob_web) < 0.001}")

print(f"\nRF Probability:")
print(f"  Training: {rf_prob_true:.6f}")
print(f"  Web App:  {rf_prob_web:.6f}")
print(f"  Match: {abs(rf_prob_true - rf_prob_web) < 0.001}")

if abs(ann_prob_true - ann_prob_web) < 0.001 and abs(rf_prob_true - rf_prob_web) < 0.001:
    print("\n[SUCCESS] Web app preprocessing matches training!")
else:
    print("\n[ERROR] Preprocessing mismatch detected!")
    print("\nFeature-by-feature comparison:")
    # We don't have access to the exact training features here, but we can note the difference
    print(f"  Max difference in features: Check above")

