# Credit Risk Analysis

Comparing six supervised machine learning models for imbalanced credit risk classification, using real-world loan data from LendingClub (Q1 2019). Models are evaluated across accuracy, precision, recall, and F1 score to find the best approach for identifying high-risk loan applicants.

## Tech Stack

- Python 3
- Jupyter Notebook
- scikit-learn, imbalanced-learn
- LendingClub loan dataset (Q1 2019, 68,817 applications)

## The Problem

Credit risk is an inherently imbalanced classification problem. In this dataset, 99% of applications are low-risk and only 1% are high-risk. Standard classifiers will just bias toward the majority class. This project tests six strategies for handling that imbalance and finds the most effective model for flagging high-risk applicants.

## Results

### Resampling Models (`credit_risk_resampling.ipynb`)

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

**EasyEnsembleClassifier** is the best performing model across every metric: 93.2% balanced accuracy, 92% recall on high-risk applicants, and an F1 score of 16% compared to 2% for all the resampling models.

The top feature by importance in the BalancedRandomForest model was `total_rec_prncp` (total principal received) at 7.9%.

### One thing to note

The 99/1 class split means even strong recall scores can hide unreliable precision. At 9% precision on high-risk predictions, this model would generate a lot of false positives in production. It works well as a screening layer, not as a final decision-maker.

## Setup

```bash
git clone https://github.com/kajev/Credit_Risk_Analysis.git
cd Credit_Risk_Analysis

pip install scikit-learn imbalanced-learn pandas numpy jupyter

jupyter notebook
```

## Notebooks

- `credit_risk_resampling.ipynb` covers oversampling (RandomOverSampler, SMOTE), undersampling (ClusterCentroids), and combined sampling (SMOTEENN)
- `credit_risk_ensemble.ipynb` covers ensemble methods (BalancedRandomForestClassifier, EasyEnsembleClassifier)

## Key Takeaways

- Ensemble methods significantly outperform resampling-only approaches on heavily imbalanced data
- Recall matters more than accuracy when the classes are this skewed
- SMOTEENN improves recall over SMOTE alone but still falls well short of ensembles
- Undersampling with ClusterCentroids performed worst, losing too much majority-class information
