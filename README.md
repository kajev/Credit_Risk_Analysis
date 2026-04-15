# Credit Risk Analysis — ML Model Comparison

Comparing six supervised machine learning approaches for imbalanced credit risk classification, using real-world loan data from LendingClub (Q1 2019). Models are evaluated across accuracy, precision, recall, and F1 score to determine the best fit for identifying high-risk loan applicants.

---

## Tech Stack

- **Language:** Python 3
- **Notebooks:** Jupyter Notebook
- **ML Libraries:** scikit-learn, imbalanced-learn
- **Data:** LendingClub loan dataset (Q1 2019, 68,817 applications)

---

## Problem Statement

Credit risk is inherently an imbalanced classification problem — in this dataset, 99% of applications are low-risk and only 1% are high-risk. Standard classifiers will bias toward the majority class. This project evaluates six strategies for handling class imbalance and surfaces the most effective model for flagging high-risk applicants without drowning in false positives.

---

## Models Evaluated

### Resampling Approaches (`credit_risk_resampling.ipynb`)

| Model | Strategy | Balanced Accuracy | High-Risk Recall | F1 (High-Risk) |
|---|---|---|---|---|
| RandomOverSampler | Oversampling | 64.0% | 66% | 2% |
| SMOTE | Oversampling | 65.1% | 61% | 2% |
| ClusterCentroids | Undersampling | 54.5% | 69% | 1% |
| SMOTEENN | Combined | 64.5% | 72% | 2% |

### Ensemble Classifiers (`credit_risk_ensemble.ipynb`)

| Model | Balanced Accuracy | High-Risk Recall | F1 (High-Risk) |
|---|---|---|---|
| BalancedRandomForestClassifier | 78.9% | 70% | 6% |
| **EasyEnsembleClassifier** | **93.2%** | **92%** | **16%** |

---

## Results

**EasyEnsembleClassifier** is the clear winner across all metrics:
- 93.2% balanced accuracy
- 92% recall on high-risk applicants (critical for a lending use case)
- 94% recall on low-risk applicants
- F1 score of 16% for high-risk (vs. 2% for all resampling models)

The top feature by importance in the BalancedRandomForest model was `total_rec_prncp` (total principal received) at 7.9%.

### Caveat

The dataset's extreme class imbalance (99% low-risk) means even strong recall scores may mask unreliable precision. In a real lending context, a 9% precision rate on high-risk predictions would generate significant false positives — this model should be used as a screening layer, not a final decision-maker.

---

## Setup

```bash
# Clone the repo
git clone https://github.com/kajev/Credit_Risk_Analysis.git
cd Credit_Risk_Analysis

# Install dependencies
pip install scikit-learn imbalanced-learn pandas numpy jupyter

# Launch notebooks
jupyter notebook
```

---

## Notebooks

- `credit_risk_resampling.ipynb` — Oversampling (RandomOverSampler, SMOTE), undersampling (ClusterCentroids), and combined (SMOTEENN) approaches
- `credit_risk_ensemble.ipynb` — Ensemble methods (BalancedRandomForestClassifier, EasyEnsembleClassifier)

---

## Key Takeaways

- Ensemble methods significantly outperform resampling-only approaches on heavily imbalanced data
- Recall is the more meaningful metric than accuracy for high-stakes imbalanced classification
- SMOTEENN's combined sampling improves recall vs. SMOTE alone but still trails ensembles by a wide margin
- Undersampling (ClusterCentroids) performed worst — losing majority-class information hurt generalization
