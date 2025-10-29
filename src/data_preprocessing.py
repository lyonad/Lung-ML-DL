"""
Data Preprocessing Module for Lung Cancer Risk Prediction
Deep Learning-Based Comparative Study

This module handles data loading, cleaning, preprocessing, and feature engineering
for the lung cancer prediction dataset.

Author: Research Team
Date: October 2025
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
import warnings
warnings.filterwarnings('ignore')


class LungCancerDataPreprocessor:
    """
    Comprehensive data preprocessing pipeline for lung cancer dataset.
    """
    
    def __init__(self, data_path):
        """
        Initialize the preprocessor with data path.
        
        Args:
            data_path (str): Path to the CSV file containing lung cancer data
        """
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        
    def load_data(self):
        """Load the dataset from CSV file."""
        print("Loading dataset...")
        self.df = pd.read_csv(self.data_path)
        print(f"Dataset loaded successfully with shape: {self.df.shape}")
        return self.df
    
    def explore_data(self):
        """Perform basic exploratory data analysis."""
        if self.df is None:
            self.load_data()
        
        print("\n" + "="*70)
        print("DATASET OVERVIEW")
        print("="*70)
        print(f"\nDataset Shape: {self.df.shape}")
        print(f"Number of Features: {self.df.shape[1] - 1}")
        print(f"Number of Samples: {self.df.shape[0]}")
        
        print("\n" + "-"*70)
        print("FEATURE INFORMATION")
        print("-"*70)
        print(self.df.info())
        
        print("\n" + "-"*70)
        print("STATISTICAL SUMMARY")
        print("-"*70)
        print(self.df.describe())
        
        print("\n" + "-"*70)
        print("MISSING VALUES")
        print("-"*70)
        missing = self.df.isnull().sum()
        if missing.sum() == 0:
            print("No missing values found in the dataset.")
        else:
            print(missing[missing > 0])
        
        print("\n" + "-"*70)
        print("TARGET VARIABLE DISTRIBUTION")
        print("-"*70)
        target_counts = self.df['LUNG_CANCER'].value_counts()
        print(target_counts)
        print(f"\nClass Distribution:")
        for label, count in target_counts.items():
            percentage = (count / len(self.df)) * 100
            print(f"  {label}: {count} ({percentage:.2f}%)")
        
        return {
            'shape': self.df.shape,
            'missing_values': missing,
            'target_distribution': target_counts
        }
    
    def preprocess_features(self):
        """
        Preprocess features including encoding and transformation.
        """
        if self.df is None:
            self.load_data()
        
        print("\nPreprocessing features...")
        
        # Create a copy to avoid modifying original data
        df_processed = self.df.copy()
        
        # Encode Gender: M=1, F=0
        df_processed['GENDER'] = df_processed['GENDER'].map({'M': 1, 'F': 0})
        
        # All other features (except LUNG_CANCER) are already numeric (1 or 2)
        # Convert them to binary (0 or 1) for better neural network performance
        binary_features = [
            'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE',
            'CHRONIC DISEASE', 'FATIGUE ', 'ALLERGY ', 'WHEEZING',
            'ALCOHOL CONSUMING', 'COUGHING', 'SHORTNESS OF BREATH',
            'SWALLOWING DIFFICULTY', 'CHEST PAIN'
        ]
        
        for feature in binary_features:
            # Convert 2->1 (YES) and 1->0 (NO)
            df_processed[feature] = df_processed[feature] - 1
        
        # Normalize age to 0-1 range
        df_processed['AGE_NORMALIZED'] = (df_processed['AGE'] - df_processed['AGE'].min()) / \
                                         (df_processed['AGE'].max() - df_processed['AGE'].min())
        
        # Encode target variable: YES=1, NO=0
        df_processed['LUNG_CANCER_ENCODED'] = self.label_encoder.fit_transform(df_processed['LUNG_CANCER'])
        
        print("Feature preprocessing completed.")
        
        return df_processed
    
    def prepare_train_test_split(self, test_size=0.2, random_state=42, use_stratify=True):
        """
        Prepare training and testing datasets with proper scaling.
        
        Args:
            test_size (float): Proportion of dataset to include in test split
            random_state (int): Random state for reproducibility
            use_stratify (bool): Whether to use stratified splitting
        
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        df_processed = self.preprocess_features()
        
        # Select features for modeling
        feature_cols = [col for col in df_processed.columns 
                       if col not in ['LUNG_CANCER', 'LUNG_CANCER_ENCODED', 'AGE']]
        
        X = df_processed[feature_cols]
        y = df_processed['LUNG_CANCER_ENCODED']
        
        self.feature_names = feature_cols
        
        # Split data
        stratify_param = y if use_stratify else None
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=stratify_param
        )
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"\nTraining set size: {self.X_train.shape[0]} samples")
        print(f"Testing set size: {self.X_test.shape[0]} samples")
        print(f"Number of features: {self.X_train.shape[1]}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def get_cross_validation_splits(self, n_splits=5, random_state=42):
        """
        Prepare cross-validation splits for model evaluation.
        
        Args:
            n_splits (int): Number of folds
            random_state (int): Random state for reproducibility
        
        Returns:
            StratifiedKFold: Cross-validation splitter
        """
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        return skf
    
    def get_feature_names(self):
        """Return the list of feature names used in the model."""
        return self.feature_names
    
    def get_full_dataset(self):
        """
        Get the full preprocessed dataset (X, y) for advanced analysis.
        
        Returns:
            tuple: X (features), y (target)
        """
        df_processed = self.preprocess_features()
        
        feature_cols = [col for col in df_processed.columns 
                       if col not in ['LUNG_CANCER', 'LUNG_CANCER_ENCODED', 'AGE']]
        
        X = df_processed[feature_cols].values
        y = df_processed['LUNG_CANCER_ENCODED'].values
        
        return X, y


def main():
    """
    Main function for testing the preprocessing pipeline.
    """
    # Example usage
    data_path = '../Data/survey lung cancer.csv'
    
    preprocessor = LungCancerDataPreprocessor(data_path)
    
    # Load and explore data
    preprocessor.load_data()
    preprocessor.explore_data()
    
    # Prepare train-test split
    X_train, X_test, y_train, y_test = preprocessor.prepare_train_test_split()
    
    print("\n" + "="*70)
    print("PREPROCESSING COMPLETE")
    print("="*70)
    print(f"Training features shape: {X_train.shape}")
    print(f"Testing features shape: {X_test.shape}")
    print(f"Feature names: {preprocessor.get_feature_names()}")


if __name__ == "__main__":
    main()

