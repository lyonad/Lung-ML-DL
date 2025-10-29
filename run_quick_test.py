"""
Quick Test Script
Deep Learning-Based Lung Cancer Risk Prediction

This script performs a quick test of the entire pipeline with reduced epochs
for rapid validation of the setup.

Usage:
    python run_quick_test.py

Author: Research Team
Date: October 2025
"""

import sys
import os

# Add src to path
sys.path.append('src')

from src.data_preprocessing import LungCancerDataPreprocessor
from models import ANNModelBuilder, ModelTrainer
from src.evaluation import ModelEvaluator, VisualizationTools

def quick_test():
    """
    Run a quick test of the pipeline with minimal epochs.
    """
    print("="*80)
    print("QUICK TEST - LUNG CANCER RISK PREDICTION")
    print("="*80)
    print("\nThis is a quick test with reduced epochs for validation.")
    print("For full training, use: python src/main_training.py")
    print("\n" + "="*80 + "\n")
    
    try:
        # Step 1: Data Loading
        print("Step 1: Loading and preprocessing data...")
        DATA_PATH = 'Data/survey lung cancer.csv'
        
        if not os.path.exists(DATA_PATH):
            print(f"Error: Dataset not found at {DATA_PATH}")
            print("Please ensure the dataset is in the correct location.")
            return
        
        preprocessor = LungCancerDataPreprocessor(DATA_PATH)
        df = preprocessor.load_data()
        print(f"[OK] Data loaded: {df.shape}")
        
        # Step 2: Train-Test Split
        print("\nStep 2: Preparing train-test split...")
        X_train, X_test, y_train, y_test = preprocessor.prepare_train_test_split()
        print(f"[OK] Training samples: {X_train.shape[0]}")
        print(f"[OK] Testing samples: {X_test.shape[0]}")
        print(f"[OK] Features: {X_train.shape[1]}")
        
        # Step 3: Build Model
        print("\nStep 3: Building Simple ANN model...")
        builder = ANNModelBuilder(input_dim=X_train.shape[1], random_state=42)
        model = builder.build_simple_ann()
        print("[OK] Model built successfully")
        print("\nModel Architecture:")
        model.summary()
        
        # Step 4: Quick Training
        print("\nStep 4: Training model (10 epochs for testing)...")
        trainer = ModelTrainer(model, 'Quick_Test_ANN')
        history = trainer.train(
            X_train, y_train,
            X_test, y_test,
            epochs=10,  # Quick test with 10 epochs
            batch_size=32,
            verbose=1
        )
        print("[OK] Training completed")
        
        # Step 5: Evaluation
        print("\nStep 5: Evaluating model...")
        metrics = trainer.evaluate(X_test, y_test)
        
        print("\n" + "="*80)
        print("QUICK TEST RESULTS")
        print("="*80)
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"AUC:       {metrics['auc']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        print("="*80)
        
        print("\n[SUCCESS] Quick test completed successfully!")
        print("\nNOTE: These results are from a quick 10-epoch training.")
        print("For full results with 100 epochs, run: python src/main_training.py")
        
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Error during quick test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def check_dependencies():
    """
    Check if all required dependencies are installed.
    """
    print("Checking dependencies...")
    
    required_packages = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'tensorflow': 'tensorflow',
        'sklearn': 'scikit-learn',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn'
    }
    
    missing_packages = []
    
    for package, pip_name in required_packages.items():
        try:
            __import__(package)
            print(f"[OK] {package}")
        except ImportError:
            print(f"[MISSING] {package}")
            missing_packages.append(pip_name)
    
    if missing_packages:
        print("\nMissing packages detected!")
        print("Please install missing packages using:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    else:
        print("\n[OK] All dependencies are installed!")
        return True


if __name__ == "__main__":
    print("\n" + "="*80)
    print("DEPENDENCY CHECK")
    print("="*80 + "\n")
    
    if check_dependencies():
        print("\n" + "="*80)
        print("RUNNING QUICK TEST")
        print("="*80 + "\n")
        quick_test()
    else:
        print("\nPlease install missing dependencies before running the test.")
        print("Run: pip install -r requirements.txt")

