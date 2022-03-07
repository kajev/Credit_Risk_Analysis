# Analysis

## Overview
Using **Machine Learning statistical algorithms** to help us make predictions on the given data. We focus on **Supervised Learning**, utilizing a free dataset from **LendingClub** (lending service company to evaluate and **predict credit risk**). The data includes a labeled outcome, so we use Supervised Learning. We will be using vatious techniques to train and evaluate the data with unbalanced classes. These algorithms include **RandomOverSampler**, **SMOTE**, **ClusterCentroids**, **SMOTEENN**, **BalancedRandomForestClassifier**, and **EasyEnsembleClassifier**.

## Table of Contents
  - Deliverable 1: Use Resampling Models to Predict Credit Risk
  - Deliverable 2: Use the SMOTEENN Algorithm to Predict Credit Risk
  - Deliverable 3: Use Ensemble Classifiers to Predict Credit Risk
  
### Deliverable 1: Use Resampling Models to Predict Credit Risk (30 points)
Deliverable 1 Instructions
Using your knowledge of the imbalanced-learn and scikit-learn libraries, you’ll evaluate three machine learning models by using resampling to determine which is better at predicting credit risk. First, you’ll use the oversampling RandomOverSampler and SMOTE algorithms, and then you’ll use the undersampling ClusterCentroids algorithm. Using these algorithms, you’ll resample the dataset, view the count of the target classes, train a logistic regression classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.

### Deliverable 2: Use the SMOTEENN algorithm to Predict Credit Risk (15 points)
Deliverable 2 Instructions
Using your knowledge of the imbalanced-learn and scikit-learn libraries, you’ll use a combinatorial approach of over- and undersampling with the SMOTEENN algorithm to determine if the results from the combinatorial approach are better at predicting credit risk than the resampling algorithms from Deliverable 1. Using the SMOTEENN algorithm, you’ll resample the dataset, view the count of the target classes, train a logistic regression classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.

### Deliverable 3: Use Ensemble Classifiers to Predict Credit Risk (25 points)
Deliverable 3 Instructions
Using your knowledge of the imblearn.ensemble library, you’ll train and compare two different ensemble classifiers, BalancedRandomForestClassifier and EasyEnsembleClassifier, to predict credit risk and evaluate each model. Using both algorithms, you’ll resample the dataset, view the count of the target classes, train the ensemble classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.

## Results

As mentioned in the overview, we use Machine Learning to resample the dataset using Python libraries: scikit-learn and imbalanced-learn evaluate the results and provide a comparison for our analysis.

The original dataset contained 115,675 loan applications in Q1 of 2019. We used the "loan status" to determine whether the application was considered "low" or "high" risk. Applications that had "current" as the "loan status" were classified as "low risk" and the remaining as "high risk". This reduced the dataset to 68,817 total applications with 99% classified as "low risk".

Using the 75/25% method to split the data for training vs. testing, 51,366 "low risk" and 246 "high risk" applications were categorized into the training set.v
## Deliverable 1: Use Resampling Models to Predict Credit Risk
### **Oversampling:** 
  - **RandomOverSampler Model** randomly selects from the minority class and adds it to the training set until both classifications are equal. The results classified 51,366 records each as High Risk and Low Risk.
  - Balanced accuracy score: 64%
  - The "High Risk" precision rate was only 1% with the recall at 66% giving this model an F1 score of 2%.
  - "Low Risk" had a precision rate of 100% and recall at 62%.
#### SMOTE (Synthetic Minority Oversampling Technique) Model  like RandomOverSampler increases the size of the minority class by creating new values based on the value of the closest neighbors to the minority class instead of random selection.
  - The balanced accuracy score improved slightly to 65.1%.
  - Like RandomOverSampler, the "High Risk" precision rate again was only 1% with the recall degraded to 61% giving this model an F1 score of 2%.
  - "Low Risk" had a precision rate of 100% and an improved recall at 69%.
### Undersampling
  - **ClusterCentroids Model** , an algorithm that identifies clusters of the majority class to generate synthetic data points that are representative of the clusters. The model classified 246 records each as High Risk and Low Risk.
  - Balanced accuracy score was lower than the oversampling models at 54.5%.
  - The "High Risk" precision rate again was only at 1% with the recall at 69% giving this model an F1 score of 1%.
  - "Low Risk" had a precision rate of 100% and with a lower recall at 40% compared to the oversampling models.


## Deliverable 2: Use the SMOTEENN algorithm to Predict Credit Risk
### Combination Sampling
  - **SMOTEENN** (Synthetic Minority Oversampling Technique + Edited NearestNeighbors) Model combines aspects of both oversampling and undersampling. The model classified 68,460 records as High Risk and 62,011 as Low Risk.
  - The balanced accuracy score improved to 64.5% when using a combined sampling model.
  - The "High Risk" precision rate did not improve was only 1%, however the recall increased to 72% giving this model an F1 score of 2%.
  - "Low Risk" still showed a precision rate of 100% with the recall at 57%.

## Deliverable 3: Use Ensemble Classifiers to Predict Credit Risk
Compare two new Machine Learning models that reduce bias to predict credit risk. The models classified 51,366 as High Risk and 246 as Low Risk.
  - **BalancedRandomForestClassifier Model:** two trees of the same size and equal size to the minority class are constructed to represent one for the majority class and one for the minority class.
  - The balanced accuracy score increased to 78.9% for this model.
  - The "High Risk precision rate increased to 3% with the recall at 70% giving this model an F1 score of 6%.
  - "Low Risk" still had a precision rate of 100% with the recall at 87%.
  - The top feature by importance was "total_rec_prncp" at 7.9% of the total.
  - **EasyEnsembleClassifier Model:** a set of classifiers where individual decisions are combined to classify new examples.
  - The balanced accuracy score increased to 93.2% with this model.
  - The "High Risk precision rate increased to 9% with the recall at 92% giving this model an F1 score of 16%.
  - "Low Risk" still had a precision rate of 100% with the recall now at 94%.

## Summary
After all these tests, the **EasyEnsembleClassifies** model performed the best with an accuracy of 93.2%and a 9% precision rate when predicting "high risk candidate".The sensitivity rate (aka recall) was also the highest at 92% compared to the other models. The result for predicting "Low Risk" was also the highest with the sensitivity rate at 94% and an F1 score of 97%. Therefore, if a model needed to be recommended to perform this type of analysis, then this one would be the clear choice.

### Ordering models from best to worst
  - **EasyEnsembleClassifer**: 93.2% accuracy, 9% precision, 92% recall, and 16% F1 Score
  - BalancedRandomForestClassifer: 78.9% accuracy, 3% precision, 70% recall and 6% F1 Score
  - SMOTE: 65.2% accuracy, 1% precision, 61% recall and 2% F1 Score
  - SMOTEENN: 64.5% accuracy, 1% precision, 72% recall and 2% F1 Score
  - RandomOverSampler: 64.0% accuracy, 1% precision, 66% recall and 2% F1 Score
  - ClusterCentroids: 54.5% accuracy, 1% precision, 69% recall and 1% F1 Score

**Note**: the original dataset had 99% of the applications classified as "Low Risk" with only 1% of the data classified in the "High Risk" category. This may skew the results greatly as there is a risk that the Machine Learning algorithms are creating clusters drawing from too small of a dataset of actual "High Risk" applications. This margin of risk might not be something that banks would be comfortable accepting.






