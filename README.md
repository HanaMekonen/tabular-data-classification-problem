## Knee Condition Classification Pipeline

This repository contains the full pipeline for tabular data classifying using high-dimensional data. The workflow includes data preprocessing, dimensionality reduction, feature selection, model training with hyperparameter tuning, evaluation, and generation of prediction outputs for submission.

##📁 Project Structure

knee-classification/
├── data/                # Original datasets
│   ├── train.csv
│   ├── test.csv
│   └── blind_test.csv
├── output/              # Prediction results
│   ├── test_predictions.csv
│   └── blind_test_predictions.csv
├── report/              # Methodology and evaluation summary
│   └── methodology_report.pdf
├── src/                 # Source code
│   ├── train_models.ipynb
│   
├── requirements.txt     # Python dependencies
├── README.md            # Project description (this file)
└── .gitignore           # Files to ignore in Git

## 🔧 Setup Instructions

Clone the repository:

git clone https://github.com/HanaMekonen/tabular-data-classification-problem.git
cd tabular-data-classification-problem

## Install the required packages:

pip install -r requirements.txt

## 🔎 Pipeline Overview

### 1. Data Preprocessing

Removed ID column for training

Outlier Clipping

Imputed missing values using mean strategy

Standardized features with StandardScaler

### 2. Feature Engineering

Applied dimension reduction using PCA (Principal Component Analysis) 

### 3. Model Training

Trained and evaluated multiple classifiers:

Logistic Regression (liblinear, C tuned)

SVM (Linear kernel, tuned C)

Random Forest (tuned n_estimators, max_depth, and max_features)


### 4. Evaluation Metrics

Reported for each model:

Accuracy

AUROC

Recall (Sensitivity)

Specificity

F1-Score

### 5. Prediction Output

Generated predicted class probabilities for test and blind_test

Saved as CSVs with format:

ID, Class_0_Prob, Class_1_Prob

## 📄 Deliverables

test_predictions.csv and blind_test_predictions.csv

methodology_report.pdf

Source code to reproduce results (src/train_models.py)

## 🚀 Future Improvements

Add class balancing with SMOTE or weights

Explore deep learning models (MLP, CNN for imaging)

Use SHAP for model explainability
