# Getting Started Guide
## Deep Learning-Based Lung Cancer Risk Prediction

This guide will help you get started with the lung cancer risk prediction project.

---

## Quick Start (5 Minutes)

### 1. Install Dependencies

```bash
# Navigate to project directory
cd Lung-Cancer

# Create virtual environment (optional but recommended)
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 2. Verify Installation

```bash
python -c "import tensorflow; import pandas; import sklearn; print('All dependencies installed successfully!')"
```

### 3. Run EDA Notebook

```bash
jupyter notebook notebooks/01_Data_Exploration_and_EDA.ipynb
```

---

## Detailed Walkthrough

### Step 1: Understanding the Data

Before training models, explore the dataset:

```python
from src.data_preprocessing import LungCancerDataPreprocessor

# Initialize preprocessor
preprocessor = LungCancerDataPreprocessor('data/survey lung cancer.csv')

# Load and explore data
df = preprocessor.load_data()
preprocessor.explore_data()
```

**What to look for:**
- Dataset size and shape
- Missing values (should be none)
- Target variable distribution
- Feature distributions

---

### Step 2: Feature Analysis

Understand which features are most important:

```python
from src.feature_analysis import FeatureAnalyzer

# Get preprocessed data
X, y = preprocessor.get_full_dataset()
feature_names = preprocessor.get_feature_names()

# Analyze features
analyzer = FeatureAnalyzer(X, y, feature_names)
importance = analyzer.calculate_feature_importance_rf()
print(importance.head(10))
```

**Key Questions:**
- Which features are most predictive?
- Are there correlations between features?
- Do features show statistical significance?

---

### Step 3: Train Your First Model

Start with a simple model:

```python
from src.models import ANNModelBuilder, ModelTrainer

# Prepare data
X_train, X_test, y_train, y_test = preprocessor.prepare_train_test_split()

# Build simple model
builder = ANNModelBuilder(input_dim=X_train.shape[1])
model = builder.build_simple_ann()

# Train model
trainer = ModelTrainer(model, 'Simple_ANN')
history = trainer.train(X_train, y_train, X_test, y_test, epochs=50)

# Evaluate
metrics = trainer.evaluate(X_test, y_test)
```

**Expected Results:**
- Training should converge in 20-50 epochs
- Accuracy should be > 80%
- Loss should decrease steadily

---

### Step 4: Compare Multiple Models

Run the complete comparison:

```python
from src.main_training import LungCancerPredictionPipeline

# Initialize pipeline
pipeline = LungCancerPredictionPipeline(
    data_path='data/survey lung cancer.csv',
    results_dir='results',
    figures_dir='figures'
)

# Run complete analysis
pipeline.run_complete_pipeline(epochs=100, batch_size=32)
```

**This will:**
1. Preprocess data
2. Analyze features
3. Build 4 different ANN models
4. Train all models
5. Evaluate and compare
6. Generate visualizations
7. Create research report

---

## Understanding the Results

### Model Comparison

After training, check `results/model_comparison.csv`:

```python
import pandas as pd

comparison = pd.read_csv('results/model_comparison.csv')
print(comparison)
```

**Metrics to focus on:**
- **Accuracy:** Overall correctness
- **Sensitivity:** Ability to detect cancer (minimize false negatives)
- **Specificity:** Ability to identify healthy patients (minimize false alarms)
- **ROC-AUC:** Overall discriminative ability (higher is better)

---

### Visualizations

Check the `figures/` directory for:

1. **ROC Curves** (`roc_curves_comparison.png`)
   - Shows trade-off between TPR and FPR
   - Curves closer to top-left are better
   - Area under curve (AUC) quantifies performance

2. **Confusion Matrices** (`confusion_matrix_*.png`)
   - Shows actual vs predicted classifications
   - Diagonal elements are correct predictions
   - Off-diagonal are errors

3. **Training History** (`training_history_*.png`)
   - Shows loss and accuracy over epochs
   - Look for convergence and overfitting
   - Validation curve should track training curve

4. **Feature Importance** (`feature_importance_*.png`)
   - Shows most predictive features
   - Helps understand model decisions
   - Useful for clinical interpretation

---

## Common Issues and Solutions

### Issue 1: Import Errors

**Problem:** `ModuleNotFoundError: No module named 'src'`

**Solution:**
```python
import sys
sys.path.append('path/to/Lung-Cancer')
```

---

### Issue 2: GPU Not Detected

**Problem:** Training is slow, GPU not used

**Solution:**
```python
import tensorflow as tf
print("GPUs Available:", tf.config.list_physical_devices('GPU'))

# If no GPU, training will use CPU (slower but works)
```

---

### Issue 3: Out of Memory

**Problem:** Training crashes with OOM error

**Solution:**
Reduce batch size:
```python
pipeline.run_complete_pipeline(epochs=100, batch_size=16)  # Instead of 32
```

---

### Issue 4: Poor Model Performance

**Problem:** Accuracy < 70%

**Troubleshooting:**
1. Check data preprocessing:
   ```python
   print(X_train.min(), X_train.max())  # Should be normalized
   print(y_train.value_counts())  # Check class balance
   ```

2. Increase training epochs:
   ```python
   history = trainer.train(..., epochs=150)  # Instead of 100
   ```

3. Try different learning rates:
   ```python
   model = builder.build_simple_ann(learning_rate=0.0001)  # Lower LR
   ```

---

## Next Steps

### 1. Experiment with Hyperparameters

Try different configurations:

```python
# Different learning rates
for lr in [0.001, 0.0001, 0.00001]:
    model = builder.build_deep_ann(learning_rate=lr)
    # ... train and evaluate

# Different dropout rates
for dropout in [0.2, 0.3, 0.4, 0.5]:
    model = builder.build_advanced_ann(dropout_rate=dropout)
    # ... train and evaluate
```

---

### 2. Cross-Validation

Implement k-fold cross-validation:

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = []

for train_idx, val_idx in skf.split(X, y):
    X_train_fold, X_val_fold = X[train_idx], X[val_idx]
    y_train_fold, y_val_fold = y[train_idx], y[val_idx]
    
    # Train and evaluate
    # ... (training code)
    scores.append(accuracy)

print(f"Mean Accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
```

---

### 3. Model Interpretation

Understand model predictions:

```python
# Feature importance through permutation
from sklearn.inspection import permutation_importance

result = permutation_importance(model, X_test, y_test, n_repeats=10)
importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': result.importances_mean
}).sort_values('Importance', ascending=False)

print(importance)
```

---

### 4. Deploy Model

Save and load models:

```python
# Save model
model.save('models/best_model.h5')

# Load model
from tensorflow.keras.models import load_model
loaded_model = load_model('models/best_model.h5')

# Make predictions
predictions = loaded_model.predict(new_data)
```

---

## Best Practices

### 1. Reproducibility

Always set random seeds:

```python
import numpy as np
import tensorflow as tf
import random

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)
```

---

### 2. Data Validation

Check data quality:

```python
# Check for NaN values
assert not df.isnull().any().any(), "Data contains NaN values"

# Check value ranges
assert df['SMOKING'].isin([1, 2]).all(), "Invalid values in SMOKING"

# Check target variable
assert df['LUNG_CANCER'].isin(['YES', 'NO']).all(), "Invalid target values"
```

---

### 3. Model Validation

Validate model output:

```python
# Predictions should be probabilities [0, 1]
preds = model.predict(X_test)
assert (preds >= 0).all() and (preds <= 1).all(), "Invalid predictions"

# Check model structure
model.summary()
```

---

### 4. Documentation

Document your experiments:

```python
# Create experiment log
experiment_log = {
    'date': datetime.now(),
    'model': 'Deep_ANN',
    'hyperparameters': {
        'learning_rate': 0.001,
        'epochs': 100,
        'batch_size': 32
    },
    'results': {
        'accuracy': 0.92,
        'auc': 0.95
    }
}

# Save log
with open('experiments.json', 'a') as f:
    json.dump(experiment_log, f)
    f.write('\n')
```

---

## Resources

### Internal Documentation
- [Project Documentation](PROJECT_DOCUMENTATION.md)
- [README](../README.md)

### External Resources
- [TensorFlow Tutorials](https://www.tensorflow.org/tutorials)
- [Keras Documentation](https://keras.io/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)

### Research Papers
- Deep Learning in Medical Diagnosis
- Neural Networks for Cancer Prediction
- Regularization Techniques in Deep Learning

---

## Support

If you encounter issues:

1. Check the documentation
2. Review common issues section
3. Check GitHub issues
4. Contact the maintainers

---

**Happy Researching! 🚀**

