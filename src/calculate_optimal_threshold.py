"""
Calculate Optimal Prediction Thresholds for Imbalanced Dataset
Uses ROC curve analysis to find best threshold for each model
"""

import numpy as np
import pickle
from sklearn.metrics import roc_curve, auc
from tensorflow import keras
from data_preprocessing import LungCancerDataPreprocessor
import json

def calculate_optimal_threshold(y_true, y_pred_proba):
    """
    Calculate optimal threshold using Youden's J statistic.
    J = Sensitivity + Specificity - 1
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    
    # Calculate Youden's J statistic for each threshold
    j_scores = tpr - fpr
    
    # Find optimal threshold
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    optimal_sensitivity = tpr[optimal_idx]
    optimal_specificity = 1 - fpr[optimal_idx]
    
    return {
        'threshold': float(optimal_threshold),
        'sensitivity': float(optimal_sensitivity),
        'specificity': float(optimal_specificity),
        'youden_index': float(j_scores[optimal_idx]),
        'auc': float(auc(fpr, tpr))
    }

def main():
    print("="*80)
    print("CALCULATING OPTIMAL PREDICTION THRESHOLDS")
    print("="*80)
    
    # Load and prepare data
    print("\n1. Loading dataset...")
    preprocessor = LungCancerDataPreprocessor('../Data/survey lung cancer.csv')
    X_train, X_test, y_train, y_test = preprocessor.prepare_train_test_split()
    
    print(f"   Test set: {len(y_test)} samples")
    print(f"   Positive: {sum(y_test)} ({sum(y_test)/len(y_test)*100:.1f}%)")
    print(f"   Negative: {len(y_test)-sum(y_test)} ({(len(y_test)-sum(y_test))/len(y_test)*100:.1f}%)")
    
    optimal_thresholds = {}
    
    # 2. Regularized ANN
    print("\n2. Analyzing Regularized ANN...")
    try:
        ann_model = keras.models.load_model('../models/Regularized_ANN_best.h5')
        ann_pred_proba = ann_model.predict(X_test, verbose=0).flatten()
        
        ann_threshold = calculate_optimal_threshold(y_test, ann_pred_proba)
        optimal_thresholds['regularized_ann'] = ann_threshold
        
        print(f"   Default threshold (0.5):")
        print(f"     Predictions: {sum(ann_pred_proba > 0.5)} HIGH_RISK, {sum(ann_pred_proba <= 0.5)} LOW_RISK")
        print(f"   Optimal threshold: {ann_threshold['threshold']:.4f}")
        print(f"     Sensitivity: {ann_threshold['sensitivity']:.4f}")
        print(f"     Specificity: {ann_threshold['specificity']:.4f}")
        print(f"     Youden's J:  {ann_threshold['youden_index']:.4f}")
        print(f"     Predictions: {sum(ann_pred_proba > ann_threshold['threshold'])} HIGH_RISK, {sum(ann_pred_proba <= ann_threshold['threshold'])} LOW_RISK")
    except Exception as e:
        print(f"   Error: {e}")
    
    # 3. Random Forest
    print("\n3. Analyzing Random Forest...")
    try:
        with open('../models/Random_Forest_best.pkl', 'rb') as f:
            rf_model = pickle.load(f)
        
        rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
        
        rf_threshold = calculate_optimal_threshold(y_test, rf_pred_proba)
        optimal_thresholds['random_forest'] = rf_threshold
        
        print(f"   Default threshold (0.5):")
        print(f"     Predictions: {sum(rf_pred_proba > 0.5)} HIGH_RISK, {sum(rf_pred_proba <= 0.5)} LOW_RISK")
        print(f"   Optimal threshold: {rf_threshold['threshold']:.4f}")
        print(f"     Sensitivity: {rf_threshold['sensitivity']:.4f}")
        print(f"     Specificity: {rf_threshold['specificity']:.4f}")
        print(f"     Youden's J:  {rf_threshold['youden_index']:.4f}")
        print(f"     Predictions: {sum(rf_pred_proba > rf_threshold['threshold'])} HIGH_RISK, {sum(rf_pred_proba <= rf_threshold['threshold'])} LOW_RISK")
    except Exception as e:
        print(f"   Error: {e}")
    
    # Save thresholds
    print("\n4. Saving optimal thresholds...")
    output_path = '../models/optimal_thresholds.json'
    with open(output_path, 'w') as f:
        json.dump(optimal_thresholds, f, indent=2)
    print(f"   Saved to: {output_path}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nOptimal thresholds will improve LOW_RISK detection while")
    print("maintaining high sensitivity for HIGH_RISK cases.")
    print("\nRecommendation:")
    print("  - Use optimal threshold for production deployment")
    print("  - Monitor false negative rate (missed HIGH_RISK cases)")
    print("  - Adjust based on clinical risk tolerance")
    print("="*80)

if __name__ == "__main__":
    main()

