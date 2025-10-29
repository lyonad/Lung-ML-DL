"""
Analyze model predictions on test set to understand LOW_RISK detection
"""

import numpy as np
import pickle
from tensorflow import keras
from data_preprocessing import LungCancerDataPreprocessor

# Load data
print("="*80)
print("ANALYZING MODEL PREDICTIONS ON TEST SET")
print("="*80)

preprocessor = LungCancerDataPreprocessor('../Data/survey lung cancer.csv')
X_train, X_test, y_train, y_test = preprocessor.prepare_train_test_split()

print(f"\nTest Set: {len(y_test)} samples")
print(f"  TRUE Negatives (NO cancer):  {sum(y_test == 0)} ({sum(y_test == 0)/len(y_test)*100:.1f}%)")
print(f"  TRUE Positives (YES cancer): {sum(y_test == 1)} ({sum(y_test == 1)/len(y_test)*100:.1f}%)")

# Load models
ann_model = keras.models.load_model('../models/Regularized_ANN_best.h5')
with open('../models/Random_Forest_best.pkl', 'rb') as f:
    rf_model = pickle.load(f)

# Get predictions
ann_probs = ann_model.predict(X_test, verbose=0).flatten()
rf_probs = rf_model.predict_proba(X_test)[:, 1]

# Optimal thresholds
ann_threshold = 0.6862
rf_threshold = 0.5467

print("\n" + "="*80)
print("ANN MODEL ANALYSIS")
print("="*80)

print(f"\nProbability Statistics:")
print(f"  Min:  {ann_probs.min():.4f}")
print(f"  Max:  {ann_probs.max():.4f}")
print(f"  Mean: {ann_probs.mean():.4f}")
print(f"  Median: {np.median(ann_probs):.4f}")

print(f"\nPredictions with Optimal Threshold ({ann_threshold:.4f}):")
ann_preds = (ann_probs >= ann_threshold).astype(int)
print(f"  Predicted HIGH_RISK: {sum(ann_preds)} ({sum(ann_preds)/len(ann_preds)*100:.1f}%)")
print(f"  Predicted LOW_RISK:  {sum(ann_preds == 0)} ({sum(ann_preds == 0)/len(ann_preds)*100:.1f}%)")

print(f"\nProbabilities for TRUE Negatives (should be LOW_RISK):")
true_neg_probs = ann_probs[y_test == 0]
for i, prob in enumerate(true_neg_probs):
    prediction = "HIGH_RISK" if prob >= ann_threshold else "LOW_RISK"
    print(f"  Sample {i+1}: {prob:.4f} => {prediction}")

print("\n" + "="*80)
print("RANDOM FOREST ANALYSIS")
print("="*80)

print(f"\nProbability Statistics:")
print(f"  Min:  {rf_probs.min():.4f}")
print(f"  Max:  {rf_probs.max():.4f}")
print(f"  Mean: {rf_probs.mean():.4f}")
print(f"  Median: {np.median(rf_probs):.4f}")

print(f"\nPredictions with Optimal Threshold ({rf_threshold:.4f}):")
rf_preds = (rf_probs >= rf_threshold).astype(int)
print(f"  Predicted HIGH_RISK: {sum(rf_preds)} ({sum(rf_preds)/len(rf_preds)*100:.1f}%)")
print(f"  Predicted LOW_RISK:  {sum(rf_preds == 0)} ({sum(rf_preds == 0)/len(rf_preds)*100:.1f}%)")

print(f"\nProbabilities for TRUE Negatives (should be LOW_RISK):")
true_neg_probs_rf = rf_probs[y_test == 0]
for i, prob in enumerate(true_neg_probs_rf):
    prediction = "HIGH_RISK" if prob >= rf_threshold else "LOW_RISK"
    print(f"  Sample {i+1}: {prob:.4f} => {prediction}")

print("\n" + "="*80)
print("CONCLUSION")
print("="*80)

# Check if models can detect any LOW_RISK
ann_low_risk_detected = sum(ann_probs[y_test == 0] < ann_threshold)
rf_low_risk_detected = sum(rf_probs[y_test == 0] < rf_threshold)

print(f"\nANN correctly identifies {ann_low_risk_detected}/{sum(y_test==0)} LOW_RISK samples")
print(f"RF correctly identifies {rf_low_risk_detected}/{sum(y_test==0)} LOW_RISK samples")

if ann_low_risk_detected > 0 or rf_low_risk_detected > 0:
    print("\n[SUCCESS] Models CAN detect LOW_RISK with optimal thresholds!")
else:
    print("\n[WARNING] Models still struggle with LOW_RISK detection")
    print("  This is due to severe class imbalance (87% positive) and")
    print("  similarity of symptom profiles between classes.")

print("="*80)

