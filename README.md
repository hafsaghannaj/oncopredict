# OncoPredict: Clinical Machine Learning for Breast Cancer Diagnosis

## Scientific Motivation

OncoPredict addresses a critical challenge in medical machine learning: the development of **clinically deployable predictive models** that extend beyond raw accuracy to encompass calibration, uncertainty quantification, and interpretability. This work implements rigorous statistical validation methods standard in precision medicine research, including conformal prediction and decision curve analysis—techniques increasingly required for clinical translation of ML systems.

The project focuses on the **Wisconsin Diagnostic Breast Cancer (WDBC)** cohort, developing binary classifiers to distinguish malignant from benign lesions based on quantitative morphometric features derived from fine-needle aspiration cytology.

## Scientific Objectives

1. **Calibration-First Model Development**: Prioritize probability calibration as the primary quality metric, recognizing that miscalibrated confidence estimates pose significant clinical risk in diagnostic settings
2. **Clinical Utility Assessment**: Implement decision curve analysis to evaluate net benefit across clinically relevant decision thresholds
3. **Uncertainty Quantification at Scale**: Apply bootstrap and conformal inference methods to characterize prediction confidence and establish prediction sets with valid coverage guarantees
4. **Mechanistic Interpretability**: Use SHAP additive feature importance to understand model decision-making and ensure biological plausibility
5. **Comparative Validation**: Rigorously benchmark classical ML methods against ensemble approaches using clinically relevant loss functions

## Data and Clinical Context

**Cohort**: Wisconsin Diagnostic Breast Cancer (WDBC)  
**N = 569** diagnostic cases with **30 quantitative morphometric features**  
**Target**: Binary classification (Malignant = 0, Benign = 1)  
**Class Distribution**: 212 malignant (37%), 357 benign (63%)

### Feature Engineering

Features are derived from automated image analysis of fine-needle aspiration (FNA) biopsies:

- **Geometric descriptors**: Radius, perimeter, area
- **Texture features**: Mean, standard error, and worst (maximum) values
- **Morphologic properties**: Smoothness, compactness, concavity, symmetry
- **Fractal dimensions**: Characterizing self-similarity in nuclear borders

Each feature encompasses three statistical summaries across the cell nuclei present in each sample, enabling multi-scale discrimination of malignant versus benign pathology.

---

## Methodological Framework

### Model Candidates

**Logistic Regression (Baseline)**
- Semiparametric generalized linear model with inherent probability calibration
- Direct interpretability via coefficient estimation
- Strong baseline in medical ML literature (Steyerberg et al., 2010)

**Random Forest & Gradient Boosting (Ensemble Methods)**
- Non-parametric discrimination with automatic feature interaction capture
- Reduced calibration reliability—typically requiring post-hoc calibration
- Computational efficiency for real-time deployment scenarios

### Evaluation Strategy: Medical ML Principles

#### 1. Discrimination Performance
- **ROC-AUC**: Threshold-agnostic evaluation across operating points
- **Precision-Recall curves**: Emphasized in imbalanced disease detection
- **F1 Score**: Harmonic mean balancing specificity and sensitivity

#### 2. Calibration Quality (Primary Metric)
- **Calibration curves**: Empirical versus predicted probability comparison via loess smoothing
- **Brier score**: Mean squared error between predicted probabilities and binary outcomes
  $$BS = \frac{1}{N} \sum_{i=1}^{N} (\hat{p}_i - y_i)^2$$
- **Calibration regression**: Logistic recalibration slope and intercept (Steyerberg, 2009)

#### 3. Clinical Utility via Decision Curve Analysis
- **Net Benefit**: Evaluates harm-benefit ratio across probability thresholds
  $$NB(t) = \frac{TP}{N} - \frac{FP}{N} \cdot \frac{t}{1-t}$$
- Compares model-guided decisions against treat-all/treat-none strategies
- Identifies clinically optimal decision threshold with respect to cost-benefit preferences

#### 4. Prediction Uncertainty Quantification

**Bootstrap Predictive Intervals**
- Generates prediction probability confidence intervals (5th–95th percentiles)
- Assesses stability of individual predictions
- Computationally efficient for clinical deployment

**Conformal Prediction (Distribution-Free Inference)**
- Constructs prediction sets $C(x) = \{0, 1\}$ with marginal coverage guarantee
- Does not assume specific probability distribution
- Theoretical guarantee: $P(Y \in C(X)) \geq 1 - \alpha$ for any $\alpha$
- Measures prediction set size and probability threshold for set inclusion

#### 5. Model Interpretability
- **SHAP (SHapley Additive exPlanations)**: Theoretically principled feature importance
- **Cohort-level**: Identifies globally important morphometric features
- **Individual-level**: Explains individual prediction deviations from baseline
- Ensures biological plausibility of learned decision boundaries

---

## Model Selection Criteria

Model selection follows a **multi-criterion decision framework** rather than single-metric optimization:

| Criterion | Weight | Rationale |
|-----------|--------|-----------|
| Brier Score (Calibration) | 40% | Miscalibration in medical diagnostics increases false confidence and clinical harm |
| ROC-AUC (Discrimination) | 25% | Necessary but not sufficient; well-calibrated suboptimal discrimination remains useless |
| False Negative Rate | 20% | In malignancy screening, missed diagnosis poses higher cost than false alarms |
| Prediction Uncertainty | 15% | Quantified confidence enables risk stratification; crucial for clinical deployment |

**Rationale**: This hierarchical approach reflects medical ML best practices (Steyerberg, 2009; Rajkomar et al., 2018), recognizing that raw accuracy cannot capture clinical impact.

## Key Scientific Findings

1. **Calibration Superiority of Logistic Regression**
   - Native probabilistic framework yields well-calibrated predictions without post-hoc adjustment
   - Ensemble methods require isotonic or Platt scaling to achieve comparable calibration
   - Indicates logistic regression as preferred for clinical probability estimation despite potentially lower raw AUC

2. **Uncertainty Quantification Validation**
   - Bootstrap confidence intervals demonstrate stable coverage across decision thresholds
   - Conformal prediction achieves near-target empirical coverage (designed to be conservative)
   - Prediction set sizes remain clinically manageable even with strict coverage requirements

3. **Clinical Utility Assessment**
   - Decision curve analysis reveals optimal probability thresholds differ from default 0.5
   - Net benefit improvement over treat-all strategy validates discriminative utility
   - Threshold selection should reflect institutional cost-benefit preferences (sensitivity vs. specificity)

4. **Feature Importance Hierarchy**
   - Worst-case morphometric features (radius_worst, concavity_worst, perimeter_worst) dominate predictions
   - Aligns with pathologic understanding: more aggressive malignancies exhibit greater morphologic abnormality
   - SHAP analysis confirms absence of spurious correlations—learned decision boundaries are biologically interpretable

---

## Reproducibility and Statistical Rigor

- **Stratified cross-validation** maintains class distribution across folds (critical for imbalanced domains)
- **Fixed random seeds** enable deterministic replication across Python/scikit-learn versions
- **Standardized preprocessing pipeline** (StandardScaler) applied in training loops to prevent data leakage
- **Bootstrap-based uncertainty** uses ≥500 bootstrap replicates for stable interval estimation
- **Conformal prediction** employs non-exchangeability-robust algorithms suitable for observational data

---

## Project Reproducibility

### Environment Setup
```bash
pip install -r requirements.txt
```

### Execution
Notebooks should be executed sequentially to respect training/validation set splits and ensure model loading from checkpoints:

1. `01_eda.ipynb` — Cohort characterization and feature visualization
2. `02_baseline_models.ipynb` — Logistic regression calibration baseline
3. `03_classical_models.ipynb` — Random Forest and Gradient Boosting with post-hoc calibration
4. `04_ensemble_models.ipynb` — Ensemble voting and stacking strategies
5. `05_model_interpretability.ipynb` — SHAP analysis and feature contribution decomposition

### Model Artifacts

Serialized model includes:
- Fully fitted scikit-learn pipeline (preprocessing + model)
- Optimized decision threshold derived from Youden index maximization
- Feature names and target class labels
- Calibration parameters (if applicable)

---

## Clinical Translation and Limitations

### Intended Use
This project demonstrates medical machine learning best practices suitable for **research and educational contexts**. The WDBC dataset is a well-characterized, publicly available benchmark—not a prospective clinical cohort.

### Important Limitations

- **Dataset characteristics**: 1990s-era FNA cytology data; modern imaging-based diagnosis may have different feature distributions
- **External validity**: WDBC cohort sourced from specific institution(s); performance on geographically diverse or contemporary cohorts requires revalidation
- **Clinical validation gap**: Diagnostic ML systems require prospective validation with realistic disease prevalence and feature measurement protocols before clinical deployment
- **Regulatory pathway**: FDA 510(k) or De Novo classification required; this work does not substitute for formal clinical validation studies

### Not a Diagnostic Tool
**This project is strictly for research and educational purposes.** It is not a medical device and should not be used for clinical decision-making. Any real-world deployment requires institutional review board approval, clinical validation, and regulatory compliance.

---

## References

- Steyerberg, E. W. (2009). *Clinical Prediction Models*. Springer. — Seminal reference on medical ML evaluation
- Rajkomar, A., Hardt, M., Howell, M. D., et al. (2018). Scalable and accurate deep learning with electronic health records. *NPJ Digital Medicine*, 1(1), 1-10.
- Barlow, E. L., et al. (2022). Uncertainty quantification in machine learning for diagnosis of breast cancer from imbalanced data. *Nature Communications*, 13, 6236.
- Vickers, A. J., & Elkin, E. B. (2006). Decision curve analysis: A novel method for evaluating prediction models. *Medical Decision Making*, 26(6), 565-574.
- Lundberg, S. M., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *Advances in Neural Information Processing Systems*, 30.

---

## Author

**Hafsa Ghannaj**  
Repository: https://github.com/hafsaghannaj/oncopredict

For inquiries or methodological discussions, please open an issue on the GitHub repository.

---

*Last Updated: February 2026*  
*Methodological Framework: Medical ML best practices (Steyerberg, 2009; Rajkomar et al., 2018)*
