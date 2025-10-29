"""
Data Analysis Only Script (No Deep Learning Required)
Deep Learning-Based Lung Cancer Risk Prediction

This script runs data preprocessing and analysis without TensorFlow/Keras.
Use this to validate data and perform EDA before deep learning.

Usage:
    python run_data_analysis_only.py

Author: Research Team
Date: October 2025
"""

import sys
import os
sys.path.append('src')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
warnings.filterwarnings('ignore')

def main():
    print("="*80)
    print("DATA ANALYSIS AND FEATURE IMPORTANCE (Without Deep Learning)")
    print("="*80)
    print("\nThis script analyzes the data without requiring TensorFlow.")
    print("\n" + "="*80 + "\n")
    
    try:
        # Step 1: Load Data
        print("Step 1: Loading dataset...")
        DATA_PATH = 'Data/survey lung cancer.csv'
        
        if not os.path.exists(DATA_PATH):
            print(f"Error: Dataset not found at {DATA_PATH}")
            return
        
        df = pd.read_csv(DATA_PATH)
        print(f"[OK] Data loaded: {df.shape}")
        print(f"  Samples: {len(df)}")
        print(f"  Features: {len(df.columns)}")
        
        # Step 2: Data Exploration
        print("\nStep 2: Data exploration...")
        print(f"\nTarget Variable Distribution:")
        target_counts = df['LUNG_CANCER'].value_counts()
        for label, count in target_counts.items():
            pct = (count / len(df)) * 100
            print(f"  {label}: {count} ({pct:.2f}%)")
        
        print(f"\nMissing Values:")
        missing = df.isnull().sum().sum()
        if missing == 0:
            print("  [OK] No missing values!")
        else:
            print(f"  [ERROR] {missing} missing values found")
        
        # Step 3: Preprocessing
        print("\nStep 3: Preprocessing data...")
        
        # Encode Gender
        df_processed = df.copy()
        df_processed['GENDER'] = df_processed['GENDER'].map({'M': 1, 'F': 0})
        
        # Convert binary features (2->1, 1->0)
        binary_features = [
            'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE',
            'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING',
            'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH',
            'SWALLOWING DIFFICULTY', 'CHEST PAIN'
        ]
        
        for feature in binary_features:
            df_processed[feature] = df_processed[feature] - 1
        
        # Normalize age
        df_processed['AGE_NORMALIZED'] = (df_processed['AGE'] - df_processed['AGE'].min()) / \
                                         (df_processed['AGE'].max() - df_processed['AGE'].min())
        
        # Encode target
        df_processed['LUNG_CANCER_ENCODED'] = df_processed['LUNG_CANCER'].map({'YES': 1, 'NO': 0})
        
        print("[OK] Preprocessing completed")
        
        # Step 4: Prepare features
        print("\nStep 4: Preparing features...")
        feature_cols = [col for col in df_processed.columns 
                       if col not in ['LUNG_CANCER', 'LUNG_CANCER_ENCODED', 'AGE']]
        
        X = df_processed[feature_cols].values
        y = df_processed['LUNG_CANCER_ENCODED'].values
        
        print(f"[OK] Feature matrix: {X.shape}")
        print(f"  Features: {feature_cols}")
        
        # Step 5: Train-Test Split
        print("\nStep 5: Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        print(f"[OK] Training samples: {len(X_train)}")
        print(f"[OK] Testing samples: {len(X_test)}")
        
        # Step 6: Feature Importance (Random Forest)
        print("\nStep 6: Calculating feature importance...")
        rf = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
        rf.fit(X_train_scaled, y_train)
        
        importance_df = pd.DataFrame({
            'Feature': feature_cols,
            'Importance': rf.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\n[OK] Top 10 Most Important Features:")
        print(importance_df.head(10).to_string(index=False))
        
        # Step 7: Quick Model Evaluation
        print("\nStep 7: Quick model evaluation (Random Forest)...")
        y_pred = rf.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n[OK] Random Forest Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=['NO', 'YES']))
        
        # Step 8: Visualizations
        print("\nStep 8: Creating visualizations...")
        
        # Create figures directory
        os.makedirs('figures', exist_ok=True)
        
        # Plot 1: Target Distribution
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        target_counts.plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'])
        axes[0].set_title('Lung Cancer Distribution', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Lung Cancer Status')
        axes[0].set_ylabel('Count')
        axes[0].tick_params(axis='x', rotation=0)
        axes[0].grid(True, alpha=0.3)
        
        axes[1].pie(target_counts, labels=target_counts.index, autopct='%1.1f%%',
                   colors=['#2ecc71', '#e74c3c'], startangle=90)
        axes[1].set_title('Lung Cancer Distribution')
        
        plt.tight_layout()
        plt.savefig('figures/target_distribution_simple.png', dpi=300, bbox_inches='tight')
        print("[OK] Saved: figures/target_distribution_simple.png")
        plt.close()
        
        # Plot 2: Feature Importance
        plt.figure(figsize=(12, 8))
        colors = sns.color_palette("viridis", len(importance_df))
        plt.barh(importance_df['Feature'], importance_df['Importance'], color=colors)
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title('Feature Importance Analysis (Random Forest)', fontsize=14, fontweight='bold')
        plt.gca().invert_yaxis()
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig('figures/feature_importance_simple.png', dpi=300, bbox_inches='tight')
        print("[OK] Saved: figures/feature_importance_simple.png")
        plt.close()
        
        # Plot 3: Correlation Matrix
        df_corr = df_processed[feature_cols + ['LUNG_CANCER_ENCODED']].copy()
        plt.figure(figsize=(14, 12))
        corr_matrix = df_corr.corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', 
                   cmap='coolwarm', center=0, square=True,
                   linewidths=1, cbar_kws={"shrink": 0.8})
        plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig('figures/correlation_matrix_simple.png', dpi=300, bbox_inches='tight')
        print("[OK] Saved: figures/correlation_matrix_simple.png")
        plt.close()
        
        # Step 9: Save Results
        print("\nStep 9: Saving results...")
        os.makedirs('results', exist_ok=True)
        
        # Save feature importance
        importance_df.to_csv('results/feature_importance_simple.csv', index=False)
        print("[OK] Saved: results/feature_importance_simple.csv")
        
        # Save summary report
        with open('results/data_analysis_report.txt', 'w') as f:
            f.write("="*80 + "\n")
            f.write("DATA ANALYSIS REPORT (Without Deep Learning)\n")
            f.write("="*80 + "\n\n")
            f.write(f"Dataset: {DATA_PATH}\n")
            f.write(f"Total Samples: {len(df)}\n")
            f.write(f"Features: {len(feature_cols)}\n")
            f.write(f"Training Samples: {len(X_train)}\n")
            f.write(f"Testing Samples: {len(X_test)}\n\n")
            f.write("Target Distribution:\n")
            for label, count in target_counts.items():
                pct = (count / len(df)) * 100
                f.write(f"  {label}: {count} ({pct:.2f}%)\n")
            f.write(f"\nRandom Forest Baseline Accuracy: {accuracy:.4f}\n\n")
            f.write("Top 10 Most Important Features:\n")
            f.write(importance_df.head(10).to_string(index=False))
            f.write("\n\n" + "="*80 + "\n")
        
        print("[OK] Saved: results/data_analysis_report.txt")
        
        print("\n" + "="*80)
        print("DATA ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*80)
        print("\nSummary:")
        print(f"[OK] Dataset analyzed: {len(df)} samples")
        print(f"[OK] Features: {len(feature_cols)}")
        print(f"[OK] Baseline accuracy: {accuracy:.4f}")
        print(f"[OK] Visualizations saved to: figures/")
        print(f"[OK] Results saved to: results/")
        
        print("\n" + "="*80)
        print("NEXT STEPS:")
        print("="*80)
        print("1. Fix TensorFlow installation (see instructions below)")
        print("2. Run full deep learning pipeline: python src/main_training.py")
        print("3. Or use notebooks: jupyter notebook")
        
        print("\n" + "="*80)
        print("FIXING TENSORFLOW ON WINDOWS:")
        print("="*80)
        print("Try these solutions:")
        print("1. Install Microsoft Visual C++ Redistributable:")
        print("   https://aka.ms/vs/17/release/vc_redist.x64.exe")
        print("\n2. Or use CPU-only version:")
        print("   pip uninstall tensorflow")
        print("   pip install tensorflow-cpu")
        print("\n3. Or downgrade to TensorFlow 2.10:")
        print("   pip uninstall tensorflow")
        print("   pip install tensorflow==2.10.0")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    main()

