"""
Neural Network Models Module for Lung Cancer Risk Prediction
Deep Learning vs. Random Forest: A Comparative Study on Pakistani Clinical Data

This module contains multiple ANN architectures and Random Forest for comparative analysis:
- Simple ANN (Baseline)
- Deep ANN (Multiple hidden layers)
- Advanced ANN (with dropout and batch normalization)
- Regularized ANN (with L2 regularization)
- Random Forest (Classical Machine Learning baseline)

Author: Research Team
Date: October 2025
"""

import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout, BatchNormalization, Input
from keras.optimizers import Adam, SGD, RMSprop
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.regularizers import l1, l2, l1_l2
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import warnings
warnings.filterwarnings('ignore')


class ANNModelBuilder:
    """
    Builder class for creating various ANN architectures for comparative study.
    """
    
    def __init__(self, input_dim, random_state=42):
        """
        Initialize the model builder.
        
        Args:
            input_dim (int): Number of input features
            random_state (int): Random seed for reproducibility
        """
        self.input_dim = input_dim
        self.random_state = random_state
        
        # Set random seeds for reproducibility
        np.random.seed(random_state)
        tf.random.set_seed(random_state)
    
    def build_simple_ann(self, learning_rate=0.001):
        """
        Build a simple ANN with one hidden layer (Baseline Model).
        
        Architecture:
        - Input Layer
        - Hidden Layer: 32 neurons, ReLU activation
        - Output Layer: 1 neuron, Sigmoid activation
        
        Args:
            learning_rate (float): Learning rate for optimizer
        
        Returns:
            keras.Model: Compiled model
        """
        model = Sequential([
            Input(shape=(self.input_dim,)),
            Dense(32, activation='relu', name='hidden_layer'),
            Dense(1, activation='sigmoid', name='output_layer')
        ], name='Simple_ANN')
        
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc'), 
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall')]
        )
        
        return model
    
    def build_deep_ann(self, learning_rate=0.001):
        """
        Build a deep ANN with multiple hidden layers.
        
        Architecture:
        - Input Layer
        - Hidden Layer 1: 128 neurons, ReLU activation
        - Hidden Layer 2: 64 neurons, ReLU activation
        - Hidden Layer 3: 32 neurons, ReLU activation
        - Output Layer: 1 neuron, Sigmoid activation
        
        Args:
            learning_rate (float): Learning rate for optimizer
        
        Returns:
            keras.Model: Compiled model
        """
        model = Sequential([
            Input(shape=(self.input_dim,)),
            Dense(128, activation='relu', name='hidden_layer_1'),
            Dense(64, activation='relu', name='hidden_layer_2'),
            Dense(32, activation='relu', name='hidden_layer_3'),
            Dense(1, activation='sigmoid', name='output_layer')
        ], name='Deep_ANN')
        
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall')]
        )
        
        return model
    
    def build_advanced_ann(self, learning_rate=0.001, dropout_rate=0.3):
        """
        Build an advanced ANN with dropout and batch normalization.
        
        Architecture:
        - Input Layer
        - Hidden Layer 1: 128 neurons, ReLU, Batch Norm, Dropout
        - Hidden Layer 2: 64 neurons, ReLU, Batch Norm, Dropout
        - Hidden Layer 3: 32 neurons, ReLU, Batch Norm, Dropout
        - Hidden Layer 4: 16 neurons, ReLU
        - Output Layer: 1 neuron, Sigmoid activation
        
        Args:
            learning_rate (float): Learning rate for optimizer
            dropout_rate (float): Dropout rate for regularization
        
        Returns:
            keras.Model: Compiled model
        """
        model = Sequential([
            Input(shape=(self.input_dim,)),
            
            Dense(128, activation='relu', name='hidden_layer_1'),
            BatchNormalization(name='batch_norm_1'),
            Dropout(dropout_rate, name='dropout_1'),
            
            Dense(64, activation='relu', name='hidden_layer_2'),
            BatchNormalization(name='batch_norm_2'),
            Dropout(dropout_rate, name='dropout_2'),
            
            Dense(32, activation='relu', name='hidden_layer_3'),
            BatchNormalization(name='batch_norm_3'),
            Dropout(dropout_rate, name='dropout_3'),
            
            Dense(16, activation='relu', name='hidden_layer_4'),
            
            Dense(1, activation='sigmoid', name='output_layer')
        ], name='Advanced_ANN')
        
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall')]
        )
        
        return model
    
    def build_regularized_ann(self, learning_rate=0.001, l2_lambda=0.01):
        """
        Build an ANN with L2 regularization.
        
        Architecture:
        - Input Layer
        - Hidden Layer 1: 128 neurons, ReLU, L2 regularization
        - Hidden Layer 2: 64 neurons, ReLU, L2 regularization
        - Hidden Layer 3: 32 neurons, ReLU, L2 regularization
        - Output Layer: 1 neuron, Sigmoid activation
        
        Args:
            learning_rate (float): Learning rate for optimizer
            l2_lambda (float): L2 regularization parameter
        
        Returns:
            keras.Model: Compiled model
        """
        model = Sequential([
            Input(shape=(self.input_dim,)),
            Dense(128, activation='relu', kernel_regularizer=l2(l2_lambda), name='hidden_layer_1'),
            Dense(64, activation='relu', kernel_regularizer=l2(l2_lambda), name='hidden_layer_2'),
            Dense(32, activation='relu', kernel_regularizer=l2(l2_lambda), name='hidden_layer_3'),
            Dense(1, activation='sigmoid', name='output_layer')
        ], name='Regularized_ANN')
        
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc'),
                    tf.keras.metrics.Precision(name='precision'),
                    tf.keras.metrics.Recall(name='recall')]
        )
        
        return model
    
    @staticmethod
    def get_callbacks(model_name, patience=20, min_delta=0.001):
        """
        Get training callbacks for model optimization.
        
        Args:
            model_name (str): Name of the model for checkpoint file
            patience (int): Patience for early stopping
            min_delta (float): Minimum change to qualify as improvement
        
        Returns:
            list: List of callback objects
        """
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=patience,
                min_delta=min_delta,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                filepath=f'../models/{model_name}_best.h5',
                monitor='val_auc',
                mode='max',
                save_best_only=True,
                verbose=0
            )
        ]
        
        return callbacks


class ModelTrainer:
    """
    Trainer class for training and evaluating ANN models.
    """
    
    def __init__(self, model, model_name):
        """
        Initialize the trainer.
        
        Args:
            model: Compiled Keras model
            model_name (str): Name of the model
        """
        self.model = model
        self.model_name = model_name
        self.history = None
    
    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=32, verbose=1, class_weight=None):
        """
        Train the model with optional class weights for imbalanced data.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            verbose (int): Verbosity mode
            class_weight (dict): Class weights for handling imbalanced data
        
        Returns:
            History: Training history object
        """
        print(f"\n{'='*70}")
        print(f"Training {self.model_name}")
        print(f"{'='*70}\n")
        
        if class_weight is not None:
            print(f"Using class weights: {class_weight}")
        
        callbacks = ANNModelBuilder.get_callbacks(self.model_name)
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=verbose,
            class_weight=class_weight
        )
        
        print(f"\n{self.model_name} training completed.")
        
        return self.history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data.
        
        Args:
            X_test: Test features
            y_test: Test labels
        
        Returns:
            dict: Evaluation metrics
        """
        print(f"\nEvaluating {self.model_name}...")
        
        results = self.model.evaluate(X_test, y_test, verbose=0)
        
        metrics = {
            'loss': results[0],
            'accuracy': results[1],
            'auc': results[2],
            'precision': results[3],
            'recall': results[4]
        }
        
        # Calculate F1 score
        if metrics['precision'] + metrics['recall'] > 0:
            metrics['f1_score'] = 2 * (metrics['precision'] * metrics['recall']) / \
                                  (metrics['precision'] + metrics['recall'])
        else:
            metrics['f1_score'] = 0
        
        print(f"\n{self.model_name} Test Results:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  AUC:       {metrics['auc']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  Loss:      {metrics['loss']:.4f}")
        
        return metrics
    
    def get_model_summary(self):
        """Get model architecture summary."""
        return self.model.summary()
    
    def predict(self, X):
        """
        Make predictions.
        
        Args:
            X: Input features
        
        Returns:
            numpy.ndarray: Predictions
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Get prediction probabilities.
        
        Args:
            X: Input features
        
        Returns:
            numpy.ndarray: Prediction probabilities
        """
        return self.model.predict(X).flatten()


class RandomForestModelBuilder:
    """
    Builder class for Random Forest classifier (Classical ML baseline).
    Provides a strong baseline for comparison with deep learning models.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the Random Forest model builder.
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
    
    def build_random_forest(self, n_estimators=200, max_depth=10, 
                           min_samples_split=5, min_samples_leaf=2,
                           class_weight='balanced', optimize=False):
        """
        Build a Random Forest classifier optimized for small datasets.
        
        Args:
            n_estimators (int): Number of trees in the forest
            max_depth (int): Maximum depth of trees (prevents overfitting)
            min_samples_split (int): Minimum samples to split a node
            min_samples_leaf (int): Minimum samples at leaf node
            class_weight (str): Handle class imbalance ('balanced' recommended)
            optimize (bool): Whether to perform grid search optimization
            
        Returns:
            RandomForestClassifier or GridSearchCV object
        """
        if optimize:
            # Parameter grid for optimization
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': ['balanced', 'balanced_subsample']
            }
            
            base_rf = RandomForestClassifier(
                random_state=self.random_state,
                n_jobs=-1,
                oob_score=True
            )
            
            model = GridSearchCV(
                base_rf,
                param_grid,
                cv=5,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            
            print("Grid Search enabled for Random Forest optimization")
        else:
            # Optimized parameters for small datasets
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                class_weight=class_weight,
                random_state=self.random_state,
                n_jobs=-1,  # Use all CPU cores
                oob_score=True,  # Out-of-bag score for validation
                max_features='sqrt',  # Good default for classification
                bootstrap=True
            )
        
        return model
    
    def get_model_info(self):
        """
        Return information about the Random Forest configuration.
        
        Returns:
            dict: Model configuration information
        """
        return {
            'model_type': 'Random Forest Classifier',
            'algorithm': 'Ensemble Decision Trees',
            'suitable_for': 'Small-to-medium datasets (100-10,000 samples)',
            'advantages': [
                'Works well with limited data',
                'Built-in feature importance',
                'Robust to outliers',
                'No feature scaling needed',
                'High interpretability'
            ],
            'parameters': {
                'n_estimators': 'Number of trees (more = better, slower)',
                'max_depth': 'Tree depth (lower = less overfitting)',
                'class_weight': 'balanced = handles imbalanced data'
            }
        }


class RandomForestTrainer:
    """
    Trainer class for Random Forest models.
    """
    
    def __init__(self, model):
        """
        Initialize the trainer.
        
        Args:
            model: RandomForestClassifier or GridSearchCV object
        """
        self.model = model
        self.is_fitted = False
        self.training_history = {}
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the Random Forest model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional, for reporting)
            y_val: Validation labels (optional, for reporting)
            
        Returns:
            dict: Training results including OOB score
        """
        print(f"\nTraining Random Forest...")
        print(f"Training samples: {len(X_train)}")
        
        # Train the model
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Get training metrics
        train_score = self.model.score(X_train, y_train)
        
        # Get OOB score if available
        oob_score = None
        if hasattr(self.model, 'oob_score_'):
            oob_score = self.model.oob_score_
        elif hasattr(self.model, 'best_estimator_'):
            if hasattr(self.model.best_estimator_, 'oob_score_'):
                oob_score = self.model.best_estimator_.oob_score_
        
        # Validation score
        val_score = None
        if X_val is not None and y_val is not None:
            val_score = self.model.score(X_val, y_val)
        
        # Store training history
        self.training_history = {
            'train_accuracy': train_score,
            'oob_score': oob_score,
            'val_accuracy': val_score
        }
        
        # Print results
        print(f"Training Accuracy: {train_score:.4f}")
        if oob_score is not None:
            print(f"OOB Score: {oob_score:.4f}")
        if val_score is not None:
            print(f"Validation Accuracy: {val_score:.4f}")
        
        # If grid search was used, print best parameters
        if hasattr(self.model, 'best_params_'):
            print(f"\nBest Parameters: {self.model.best_params_}")
        
        # Save the trained model
        import pickle
        import os
        model_path = '../models/Random_Forest_best.pkl'
        os.makedirs('../models', exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {model_path}")
        
        return self.training_history
    
    def get_feature_importance(self, feature_names=None):
        """
        Get feature importance from the trained model.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            dict: Feature importance scores
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before getting feature importance")
        
        # Get the actual model (handle GridSearchCV)
        if hasattr(self.model, 'best_estimator_'):
            model = self.model.best_estimator_
        else:
            model = self.model
        
        importance = model.feature_importances_
        
        if feature_names is not None:
            return dict(zip(feature_names, importance))
        else:
            return {f'feature_{i}': imp for i, imp in enumerate(importance)}


def main():
    """
    Main function for testing model architectures.
    """
    print("="*70)
    print("LUNG CANCER PREDICTION MODELS")
    print("Deep Learning vs. Random Forest Comparison")
    print("Pakistani Clinical Data (n=309)")
    print("="*70)
    
    # Example usage
    input_dim = 14  # Number of features in the dataset
    
    # Build ANN models
    print("\n" + "="*70)
    print("DEEP LEARNING MODELS (Artificial Neural Networks)")
    print("="*70)
    
    ann_builder = ANNModelBuilder(input_dim=input_dim)
    
    ann_models = {
        'Simple_ANN': ann_builder.build_simple_ann(),
        'Deep_ANN': ann_builder.build_deep_ann(),
        'Advanced_ANN': ann_builder.build_advanced_ann(),
        'Regularized_ANN': ann_builder.build_regularized_ann()
    }
    
    # Display ANN architectures
    for name, model in ann_models.items():
        print(f"\n{'-'*70}")
        print(f"{name} Architecture:")
        print(f"{'-'*70}")
        model.summary()
        print()
    
    # Build Random Forest model
    print("\n" + "="*70)
    print("CLASSICAL MACHINE LEARNING MODEL (Random Forest)")
    print("="*70)
    
    rf_builder = RandomForestModelBuilder()
    rf_model = rf_builder.build_random_forest()
    
    print(f"\n{'-'*70}")
    print("Random Forest Configuration:")
    print(f"{'-'*70}")
    print(f"Model Type: {type(rf_model).__name__}")
    print(f"Number of Trees: {rf_model.n_estimators}")
    print(f"Max Depth: {rf_model.max_depth}")
    print(f"Min Samples Split: {rf_model.min_samples_split}")
    print(f"Class Weight: {rf_model.class_weight}")
    print(f"OOB Score: Enabled")
    print(f"Max Features: {rf_model.max_features}")
    
    model_info = rf_builder.get_model_info()
    print(f"\nAdvantages for Small Datasets:")
    for adv in model_info['advantages']:
        print(f"  • {adv}")
    
    print(f"\nSuitable for: {model_info['suitable_for']}")


if __name__ == "__main__":
    main()

