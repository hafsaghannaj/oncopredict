# OncoPredict

OncoPredict is a machine learning project focused on breast cancer classification using the Wisconsin Diagnostic Breast Cancer dataset. The goal is to build clinically responsible predictive models that prioritize not only accuracy but also calibration, interpretability, and uncertainty quantification.

This project demonstrates a full applied medical machine learning workflow from data processing to advanced model validation and uncertainty estimation.

---

## Project Goals

- Build predictive models for breast cancer classification
- Evaluate models using clinically meaningful metrics
- Compare classical ML and ensemble methods
- Quantify prediction uncertainty using modern statistical techniques
- Demonstrate research-grade evaluation methodology

---

## Dataset

Source: scikit-learn Breast Cancer Wisconsin (Diagnostic) dataset

Target Encoding:
- 0 = Malignant
- 1 = Benign

Features include tumor measurements such as:
- Radius
- Texture
- Perimeter
- Area
- Smoothness
- Concavity
- Symmetry
- Fractal dimension

---

## Models Implemented

### Baseline Model
- Logistic Regression (Pipeline with StandardScaler)

### Classical ML Models
- Random Forest
- Gradient Boosting

---

## Evaluation Framework

### Classification Performance
- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

### Calibration Quality
- Calibration curves
- Brier score
- Calibration slope and intercept

### Clinical Utility
- Decision Curve Analysis (Net Benefit)

### Model Interpretability
- SHAP value feature importance
- Feature contribution visualization

---

## Uncertainty Quantification

### Bootstrap Predictive Uncertainty
- Prediction probability confidence intervals (P05–P95)

### Conformal Prediction
- Coverage estimation
- Prediction set size measurement
- Probability inclusion threshold estimation

---

## Model Selection Strategy

Final model selection is based on a weighted combination of:

- Discrimination performance (ROC-AUC)
- Calibration reliability (Brier Score + slope)
- Clinical safety (False Negative minimization)
- Probability stability

---

## Key Findings

- Logistic Regression demonstrated strong calibration and probability reliability
- Ensemble models provided competitive discrimination performance
- Uncertainty quantification showed stable prediction confidence
- Conformal prediction achieved near-target coverage with low prediction set size

---

## Project Structure

```
OncoPredict/
│
├── data/
│   └── breast_cancer_sklearn.csv
│
├── models/
│   └── oncopredict_logreg_v1.joblib
│
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline_models.ipynb
│   ├── 03_classical_models.ipynb
│   ├── 04_ensemble_models.ipynb
│   ├── 05_model_interpretability.ipynb
│
├── requirements.txt
└── README.md
```

---

## How To Run

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Run Notebooks

Open Jupyter and run notebooks sequentially.

---

## Saved Model Artifact

The saved model file contains:

* Trained pipeline
* Optimized decision threshold
* Feature names
* Target class names

Example loading:

```python
import joblib

artifact = joblib.load("models/oncopredict_logreg_v1.joblib")
model = artifact["model"]
threshold = artifact["threshold"]
```

---

## Reproducibility

* Random seeds fixed where applicable
* Stratified train/test splitting used
* Pipeline-based preprocessing

---

## Disclaimer

This project is for research and educational purposes only.
It is not a medical diagnostic tool and should not be used for clinical decision making.

---

## Author

Hafsa Ghannaj
GitHub: [https://github.com/hafsaghannaj](https://github.com/hafsaghannaj)
