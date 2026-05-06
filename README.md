# Fraud Click Detection System

## Features

* **Developed an ensemble fraud click detection system** using:
  * Random Forest Classifier
  * XGBoost Classifier
  * Neural Network (Multi-Layer Perceptron)

* **Achieved high predictive performance**:
  * Random Forest: **92.9% Accuracy**, **0.714 F1-Score**
  * XGBoost: **95.6% Accuracy**, **0.846 F1-Score**
  * Neural Network: **97.5% Accuracy**, **0.917 F1-Score**
  * Ensemble Model: **97.2% Accuracy**, **0.906 F1-Score**

* **Ensemble Learning**:
  * Combined predictions using soft and hard voting methods
  * Improved fraud detection on imbalanced datasets

* **Interactive Hyperparameter Tuning**:
  * Random Forest: number of trees, max depth, min samples split
  * XGBoost: number of estimators, max depth, learning rate
  * Neural Network: hidden layers, activation function, learning rate, iterations

* **Advanced Evaluation Metrics**:
  * Accuracy, Precision, Recall, F1-Score
  * Confusion Matrices
  * ROC Curves with AUC scores
  * Precision-Recall Curves
  * Feature Importance Analysis

## Methodology

### Data Preprocessing

* Median imputation for missing values
* Categorical variable encoding
* Feature standardization using StandardScaler
* Train-test split

### Model Training

* Trained three independent ML models
* Hyperparameter optimization for each model

### Evaluation

* Multi-metric performance analysis
* Visualization for interpretability

### Ensemble Optimization

* Voting-based prediction aggregation
* Enhanced detection robustness

## Results

### Random Forest Model

* **Accuracy:** 0.929
* **Precision:** 0.930
* **Recall:** 0.579
* **F1-Score:** 0.714

### XGBoost Model

* **Accuracy:** 0.956
* **Precision:** 0.921
* **Recall:** 0.783
* **F1-Score:** 0.846

### Neural Network Model

* **Accuracy:** 0.975
* **Precision:** 0.925
* **Recall:** 0.909
* **F1-Score:** 0.917

### Ensemble Model

* **Accuracy:** 0.972
* **Precision:** 0.939
* **Recall:** 0.874
* **F1-Score:** 0.906

## Tech Stack

* **Python**
* **Streamlit**
* **Pandas**
* **NumPy**
* **Scikit-learn**
* **XGBoost**
* **Plotly**

## References

* [MDPI Journal - Click Fraud Detection](https://www.mdpi.com/2224-2708/12/1/4)
* [Kaggle Dataset 1 - User Behavior & Ad Clickstream Logs](https://www.kaggle.com/datasets/programmer3/fraud-detection-dataset)
* [Kaggle Dataset 2 - Behavioral & Device Data](https://www.kaggle.com/datasets/ziya07/fraud-detection-dataset)
