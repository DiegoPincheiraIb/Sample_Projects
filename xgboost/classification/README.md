# XGBoost Classification: Iris Dataset
<!-- Conda Env Name: xgboost_env -->

A very simple project that uses **XGBoost** to classify iris flower species.

## What is this experiment about?

- How gradient boosting works
- Training and evaluating an XGBoost model
- Feature importance analysis

## Setup

```bash
pip install -r requirements.txt
```

## How to Run

```bash
python train.py
```

## Expected Output

```python
Dataset: 150 samples, 4 features
Classes: ['setosa', 'versicolor', 'virginica']

Test Accuracy: 1.0000

Classification Report:
              precision    recall  f1-score   support
      setosa       1.00      1.00      1.00        10
  versicolor       1.00      1.00      1.00        9
   virginica       1.00      1.00      1.00        11

Feature Importances:
- Feature 0: 0.0204
- Feature 1: 0.0374
- Feature 2: 0.6733
- Feature 3: 0.2688

```

## Dataset

The classic Iris dataset (150 samples, 4 features, 3 classes) from `sklearn.datasets`.

## Examples using HTML pages

TO DO!
