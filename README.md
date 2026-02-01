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
  
- **Comprehensive Evaluation Metrics**:
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
   - Use stratified sampling to handle class imbalance

3. **Evaluation**:
   - Calculate multiple performance metrics
   - Generate visualizations for model interpretation
   - Compare feature importances

4. **Ensemble**:
   - Combine predictions using soft or hard voting
   - Leverage the strengths of all models

##  Performance Metrics

The application evaluates models using:

- **Accuracy**: Overall correctness of predictions
- **Precision**: Ratio of true positives to all positive predictions
- **Recall**: Ratio of true positives to all actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve
- **Confusion Matrix**: Detailed breakdown of predictions

##  Model Details

### Random Forest
- Ensemble of decision trees
- Robust to overfitting
- Provides feature importance rankings
- Works well with non-linear relationships

### XGBoost
- Gradient boosting framework
- High performance and efficiency
- Handles missing data well
- Often achieves state-of-the-art results

### Neural Network
- Multi-layer perceptron architecture
- Learns complex non-linear patterns
- Flexible architecture through hidden layers
- Good for high-dimensional data

### Ensemble
- Voting classifier combining all three models
- Reduces variance and bias
- More robust predictions
- Usually outperforms individual models

##  TechStack

- **Streamlit**: Web application framework
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Scikit-learn**: Machine learning algorithms
- **XGBoost**: Gradient boosting library
- **Plotly**: Interactive visualizations

##  References

- [MDPI Journal - Click Fraud Detection](https://www.mdpi.com/2224-2708/12/1/4)
- [Kaggle Dataset 1 - User Behavior & Ad Clickstream Logs](https://www.kaggle.com/datasets/programmer3/fraud-detection-dataset)
- [Kaggle Dataset 2 - Behavioral & Device Data](https://www.kaggle.com/datasets/ziya07/fraud-detection-dataset)

