# API Reference
## Deep Learning-Based Lung Cancer Risk Prediction

This document provides detailed API reference for all modules in the project.

---

## Table of Contents

1. [Data Preprocessing](#data-preprocessing)
2. [Models](#models)
3. [Evaluation](#evaluation)
4. [Feature Analysis](#feature-analysis)

---

## Data Preprocessing

### `LungCancerDataPreprocessor`

Main class for data preprocessing and preparation.

#### Constructor

```python
LungCancerDataPreprocessor(data_path: str)
```

**Parameters:**
- `data_path` (str): Path to the CSV file containing lung cancer data

**Example:**
```python
preprocessor = LungCancerDataPreprocessor('data/survey lung cancer.csv')
```

---

#### Methods

##### `load_data()`

Load the dataset from CSV file.

**Returns:**
- `pd.DataFrame`: Loaded dataset

**Example:**
```python
df = preprocessor.load_data()
```

---

##### `explore_data()`

Perform basic exploratory data analysis.

**Returns:**
- `dict`: Dictionary containing exploration results
  - `shape`: Dataset shape
  - `missing_values`: Missing value counts
  - `target_distribution`: Target variable distribution

**Example:**
```python
results = preprocessor.explore_data()
print(results['shape'])
```

---

##### `preprocess_features()`

Preprocess features including encoding and transformation.

**Returns:**
- `pd.DataFrame`: Preprocessed dataframe

**Example:**
```python
df_processed = preprocessor.preprocess_features()
```

---

##### `prepare_train_test_split(test_size=0.2, random_state=42, use_stratify=True)`

Prepare training and testing datasets with proper scaling.

**Parameters:**
- `test_size` (float, default=0.2): Proportion of dataset for testing
- `random_state` (int, default=42): Random state for reproducibility
- `use_stratify` (bool, default=True): Whether to use stratified splitting

**Returns:**
- `tuple`: (X_train, X_test, y_train, y_test)

**Example:**
```python
X_train, X_test, y_train, y_test = preprocessor.prepare_train_test_split()
```

---

##### `get_feature_names()`

Return the list of feature names used in the model.

**Returns:**
- `list`: Feature names

**Example:**
```python
features = preprocessor.get_feature_names()
```

---

## Models

### `ANNModelBuilder`

Builder class for creating various ANN architectures.

#### Constructor

```python
ANNModelBuilder(input_dim: int, random_state: int = 42)
```

**Parameters:**
- `input_dim` (int): Number of input features
- `random_state` (int, default=42): Random seed for reproducibility

**Example:**
```python
builder = ANNModelBuilder(input_dim=14)
```

---

#### Methods

##### `build_simple_ann(learning_rate=0.001)`

Build a simple ANN with one hidden layer (Baseline Model).

**Parameters:**
- `learning_rate` (float, default=0.001): Learning rate for optimizer

**Returns:**
- `keras.Model`: Compiled model

**Architecture:**
- Input Layer
- Hidden Layer: 32 neurons, ReLU activation
- Output Layer: 1 neuron, Sigmoid activation

**Example:**
```python
model = builder.build_simple_ann()
```

---

##### `build_deep_ann(learning_rate=0.001)`

Build a deep ANN with multiple hidden layers.

**Parameters:**
- `learning_rate` (float, default=0.001): Learning rate for optimizer

**Returns:**
- `keras.Model`: Compiled model

**Architecture:**
- Input Layer
- Hidden Layer 1: 128 neurons, ReLU
- Hidden Layer 2: 64 neurons, ReLU
- Hidden Layer 3: 32 neurons, ReLU
- Output Layer: 1 neuron, Sigmoid

**Example:**
```python
model = builder.build_deep_ann()
```

---

##### `build_advanced_ann(learning_rate=0.001, dropout_rate=0.3)`

Build an advanced ANN with dropout and batch normalization.

**Parameters:**
- `learning_rate` (float, default=0.001): Learning rate for optimizer
- `dropout_rate` (float, default=0.3): Dropout rate for regularization

**Returns:**
- `keras.Model`: Compiled model

**Example:**
```python
model = builder.build_advanced_ann(dropout_rate=0.4)
```

---

##### `build_regularized_ann(learning_rate=0.001, l2_lambda=0.01)`

Build an ANN with L2 regularization.

**Parameters:**
- `learning_rate` (float, default=0.001): Learning rate for optimizer
- `l2_lambda` (float, default=0.01): L2 regularization parameter

**Returns:**
- `keras.Model`: Compiled model

**Example:**
```python
model = builder.build_regularized_ann(l2_lambda=0.02)
```

---

### `ModelTrainer`

Trainer class for training and evaluating ANN models.

#### Constructor

```python
ModelTrainer(model, model_name: str)
```

**Parameters:**
- `model`: Compiled Keras model
- `model_name` (str): Name of the model

**Example:**
```python
trainer = ModelTrainer(model, 'My_ANN')
```

---

#### Methods

##### `train(X_train, y_train, X_val, y_val, epochs=100, batch_size=32, verbose=1)`

Train the model.

**Parameters:**
- `X_train`: Training features
- `y_train`: Training labels
- `X_val`: Validation features
- `y_val`: Validation labels
- `epochs` (int, default=100): Number of training epochs
- `batch_size` (int, default=32): Batch size for training
- `verbose` (int, default=1): Verbosity mode

**Returns:**
- `History`: Training history object

**Example:**
```python
history = trainer.train(X_train, y_train, X_test, y_test, epochs=50)
```

---

##### `evaluate(X_test, y_test)`

Evaluate the model on test data.

**Parameters:**
- `X_test`: Test features
- `y_test`: Test labels

**Returns:**
- `dict`: Evaluation metrics
  - `loss`: Test loss
  - `accuracy`: Test accuracy
  - `auc`: Area under ROC curve
  - `precision`: Precision score
  - `recall`: Recall score
  - `f1_score`: F1 score

**Example:**
```python
metrics = trainer.evaluate(X_test, y_test)
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

---

## Evaluation

### `ModelEvaluator`

Comprehensive model evaluation and comparison class.

#### Constructor

```python
ModelEvaluator()
```

**Example:**
```python
evaluator = ModelEvaluator()
```

---

#### Methods

##### `evaluate_model(model_name, y_true, y_pred_proba, threshold=0.5)`

Evaluate a single model with comprehensive metrics.

**Parameters:**
- `model_name` (str): Name of the model
- `y_true`: True labels
- `y_pred_proba`: Predicted probabilities
- `threshold` (float, default=0.5): Classification threshold

**Returns:**
- `dict`: Dictionary containing all evaluation metrics

**Example:**
```python
results = evaluator.evaluate_model('My_Model', y_test, predictions)
```

---

##### `compare_models()`

Compare all evaluated models and create a comparison dataframe.

**Returns:**
- `pd.DataFrame`: Comparison table

**Example:**
```python
comparison = evaluator.compare_models()
print(comparison)
```

---

### `VisualizationTools`

Visualization tools for model evaluation and comparison.

#### Static Methods

##### `plot_confusion_matrix(cm, model_name, save_path=None)`

Plot confusion matrix heatmap.

**Parameters:**
- `cm`: Confusion matrix
- `model_name` (str): Name of the model
- `save_path` (str, optional): Path to save the figure

**Example:**
```python
VisualizationTools.plot_confusion_matrix(cm, 'My_Model', 'figures/cm.png')
```

---

##### `plot_roc_curves(evaluator, save_path=None)`

Plot ROC curves for all models.

**Parameters:**
- `evaluator` (ModelEvaluator): Evaluator object with results
- `save_path` (str, optional): Path to save the figure

**Example:**
```python
VisualizationTools.plot_roc_curves(evaluator, 'figures/roc.png')
```

---

## Feature Analysis

### `FeatureAnalyzer`

Comprehensive feature analysis for clinical features.

#### Constructor

```python
FeatureAnalyzer(X, y, feature_names: list)
```

**Parameters:**
- `X`: Feature matrix
- `y`: Target labels
- `feature_names` (list): List of feature names

**Example:**
```python
analyzer = FeatureAnalyzer(X, y, feature_names)
```

---

#### Methods

##### `calculate_feature_importance_rf()`

Calculate feature importance using Random Forest.

**Returns:**
- `pd.DataFrame`: Feature importance scores

**Example:**
```python
importance = analyzer.calculate_feature_importance_rf()
```

---

##### `calculate_mutual_information()`

Calculate mutual information between features and target.

**Returns:**
- `pd.DataFrame`: Mutual information scores

**Example:**
```python
mi_scores = analyzer.calculate_mutual_information()
```

---

### `FeatureVisualizer`

Visualization tools for feature analysis.

#### Static Methods

##### `plot_feature_importance(importance_df, title, save_path=None)`

Plot feature importance bar chart.

**Parameters:**
- `importance_df`: DataFrame with Feature and Importance columns
- `title` (str): Plot title
- `save_path` (str, optional): Path to save the figure

**Example:**
```python
FeatureVisualizer.plot_feature_importance(importance_df, 'Feature Importance')
```

---

## Complete Example

```python
# Import modules
from src.data_preprocessing import LungCancerDataPreprocessor
from src.models import ANNModelBuilder, ModelTrainer
from src.evaluation import ModelEvaluator, VisualizationTools
from src.feature_analysis import FeatureAnalyzer

# 1. Data Preprocessing
preprocessor = LungCancerDataPreprocessor('data/survey lung cancer.csv')
X_train, X_test, y_train, y_test = preprocessor.prepare_train_test_split()

# 2. Build Model
builder = ANNModelBuilder(input_dim=X_train.shape[1])
model = builder.build_deep_ann()

# 3. Train Model
trainer = ModelTrainer(model, 'Deep_ANN')
history = trainer.train(X_train, y_train, X_test, y_test)

# 4. Evaluate Model
evaluator = ModelEvaluator()
y_pred = trainer.predict_proba(X_test)
results = evaluator.evaluate_model('Deep_ANN', y_test, y_pred)

# 5. Visualize Results
evaluator.print_evaluation_report('Deep_ANN')
VisualizationTools.plot_confusion_matrix(
    results['confusion_matrix'], 
    'Deep_ANN', 
    'figures/cm.png'
)
```

---

**Version:** 1.0.0  
**Last Updated:** October 28, 2025

