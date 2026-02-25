# Scikit-Learn Pipeline — Wine Quality Prediction

This project shows a barebones **Scikit-Learn** machine learning pipeline that features: preprocessing algorithm, dimensionality reduction, and a Random Forest classifier.

## What You'll Find Here
- `Pipeline` uses for clean, reproducible workflows
- `StandardScaler` for feature normalization
- `PCA` for dimensionality reduction
- `RandomForestClassifier` for classification
- Cross-validation with `cross_val_score`

## Setup

```bash
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Key Concepts

| Step | Class | Purpose |
|------|-------|---------|
| 1 | `StandardScaler` | Zero-mean, unit-variance normalization |
| 2 | `PCA` | Reduce 13 features → 8 principal components |
| 3 | `RandomForestClassifier` | Ensemble of decision trees |

## Pipeline Benefits
- Prevents data leakage (fit only on training set)
- Makes prediction on new data a single `.predict()` call
- Easy to swap components
