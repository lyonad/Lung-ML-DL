"""
Save the StandardScaler to disk for use in web app
"""

import pickle
from data_preprocessing import LungCancerDataPreprocessor

print("="*80)
print("SAVING STANDARDSCALER")
print("="*80)

# Load and preprocess data
prep = LungCancerDataPreprocessor('../Data/survey lung cancer.csv')
X_train, X_test, y_train, y_test = prep.prepare_train_test_split()

# The scaler is fitted in prepare_train_test_split()
scaler = prep.scaler

print(f"\nScaler parameters:")
print(f"  Mean: {scaler.mean_}")
print(f"  Scale: {scaler.scale_}")

# Save scaler
output_path = '../models/feature_scaler.pkl'
with open(output_path, 'wb') as f:
    pickle.dump(scaler, f)

print(f"\n[SUCCESS] Scaler saved to: {output_path}")
print("="*80)

