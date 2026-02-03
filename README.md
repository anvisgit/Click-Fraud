# Fraud Click Detection 

##  Features

- **Multiple ML Models**: Compare performance across three different algorithms
  -  Random Forest Classifier
  -  XGBoost Classifier
  -  Neural Network (Multi-Layer Perceptron)
  
- **Ensemble Learning**: Combine predictions from all models using voting mechanisms for superior accuracy

- **Interactive Hyperparameter Tuning**: Adjust model parameters in real-time and see immediate results
  - Random Forest: number of trees, max depth, min samples split
  - XGBoost: number of estimators, max depth, learning rate
  - Neural Network: hidden layers, activation function, learning rate, iterations
  
- **Evaluation Metrics**:
  - Accuracy, Precision, Recall, F1-Score
  - Confusion Matrices
  - ROC Curves with AUC scores
  - Precision-Recall Curves
  - Feature Importance Analysis
  

##  Methodology

1. **Data Preprocessing**:
   - Handle missing values using median imputation
   - Encode categorical variables
   - Standardize features using StandardScaler
   - Split data into training and testing sets

2. **Model Training**:
   - Train three independent models with customizable hyperparameters

3. **Evaluation**:
   - Calculate multiple performance metrics
   - Generate visualizations for model interpretation

4. **Ensemble**:
   - Combine predictions using soft or hard voting
   - Leverage the strengths of all models

##  Performance Metrics

- **Accuracy**
- **Precision**
- **Recall**
- **F1-Score**
- **ROC**
- **Confusion Matrix**

##  Model Details

### Random Forest
- Ensemble of decision trees
- Works well with non-linear relationships

### XGBoost
- Gradient boosting framework
- Handles missing data well

### Neural Network
- Multi-layer perceptron architecture
- Complex non-linear patterns
- For high-dimensional data

### Ensemble
- Voting classifier combining all three models
- Reduces variance and bias
- More robust predictions

##  TechStack

- **Streamlit**
- **Pandas**
- **NumPy**
- **Scikit-learn**
- **XGBoost**
- **Plotly**

##  References

- [MDPI Journal - Click Fraud Detection](https://www.mdpi.com/2224-2708/12/1/4)
- [Kaggle Dataset 1 - User Behavior & Ad Clickstream Logs](https://www.kaggle.com/datasets/programmer3/fraud-detection-dataset)
- [Kaggle Dataset 2 - Behavioral & Device Data](https://www.kaggle.com/datasets/ziya07/fraud-detection-dataset)

