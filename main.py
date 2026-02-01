import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, f1_score, accuracy_score, precision_score, recall_score
import xgboost as xgb
from sklearn.neural_network import MLPClassifier
import warnings
import os
warnings.filterwarnings('ignore')

st.set_page_config(page_title="Fraud Click Detection System", page_icon="F", layout="wide")

# Session state
if 'current_dataset' not in st.session_state:
    st.session_state.current_dataset = "fraud_dataset_1.csv"

@st.cache_data
def load_data(file_path):
    return pd.read_csv(file_path)

@st.cache_data
def preprocess_data(df, target_column='is_fraud'):
    data = df.copy()
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if target_column in numeric_cols:
        numeric_cols.remove(target_column)
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
    categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
    for col in categorical_cols:
        data[col] = pd.Categorical(data[col]).codes
    if target_column in data.columns:
        X = data.drop(target_column, axis=1)
        y = data[target_column]
    else:
        X = data
        y = np.random.randint(0, 2, size=len(data))
    return X, y

def create_confusion_matrix_plot(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    fig = go.Figure(data=go.Heatmap(z=cm, x=['Predicted Legitimate', 'Predicted Fraud'],
                                     y=['Actual Legitimate', 'Actual Fraud'], colorscale='Purples',
                                     text=cm, texttemplate='%{text}', textfont={"size": 16}))
    fig.update_layout(title=title, xaxis_title="Predicted", yaxis_title="Actual", height=400)
    return fig

def create_roc_curve_plot(y_true, y_pred_proba, title):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC={auc_score:.3f})', line=dict(color='#667eea', width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(color='gray', width=2, dash='dash')))
    fig.update_layout(title=title, xaxis_title='FPR', yaxis_title='TPR', height=400)
    return fig

def main():
    st.title("Fraud Click Detection System")
    st.write("Advanced Machine Learning for Click Fraud Prevention")
    
    # Sidebar
    with st.sidebar:
        st.header("Data Configuration")
        dataset_choice = st.selectbox("Choose Dataset:", 
                                      ["fraud_dataset_1.csv", "fraud_dataset_2.csv", "fraud_dataset_3.csv"])
        st.session_state.current_dataset = dataset_choice
        st.success(f"Using {dataset_choice}")
        
        st.markdown("---")
        st.header("Model Settings")
        test_size = st.slider("Test Set Size", 0.1, 0.5, 0.3, 0.05)
        random_state = st.number_input("Random State", value=42, min_value=0)
    
    # Load data
    if os.path.exists(st.session_state.current_dataset):
        df = load_data(st.session_state.current_dataset)
        fraud_cols = [col for col in df.columns if 'fraud' in col.lower() or 'label' in col.lower()]
        target_col = fraud_cols[0] if fraud_cols else df.columns[-1]
    else:
        st.error("Dataset not found. Please run: python generate_datasets.py")
        return
    
    # Tabs
    tab_home, tab_rf, tab_xgb, tab_nn, tab_ensemble = st.tabs(
        ["Home", "Random Forest", "XGBoost", "Neural Network", "Ensemble"])
    
    # HOME TAB
    with tab_home:
        st.header("Project Overview")
        st.write("""
        This application uses machine learning to detect fraudulent clicks in online advertising.
        
        **Features:**
        - Multiple ML models (Random Forest, XGBoost, Neural Network)
        - Ensemble learning for best performance
        - Interactive hyperparameter tuning
        - Comprehensive evaluation metrics
        """)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.subheader("Dataset Statistics")
            st.metric("Total Records", f"{len(df):,}")
            st.metric("Features", df.shape[1] - 1)
            fraud_rate = (df[target_col].sum() / len(df)) * 100
            st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
        with col2:
            st.subheader("Data Preview")
            st.dataframe(df.head(), use_container_width=True)
    
    # RANDOM FOREST TAB
    with tab_rf:
        st.header("Random Forest Model")
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("Random Forest Parameters")
        rf_n_estimators = st.sidebar.slider("Number of Trees", 50, 500, 100, 50)
        rf_max_depth = st.sidebar.slider("Max Depth", 5, 50, 10, 5)
        rf_min_samples_split = st.sidebar.slider("Min Samples Split", 2, 20, 2)
        
        if st.button("Train Random Forest Model", key="train_rf"):
            with st.spinner("Training..."):
                X, y = preprocess_data(df, target_col)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                rf_model = RandomForestClassifier(n_estimators=rf_n_estimators, max_depth=rf_max_depth,
                                                 min_samples_split=rf_min_samples_split, random_state=random_state, n_jobs=-1)
                rf_model.fit(X_train_scaled, y_train)
                
                y_pred = rf_model.predict(X_test_scaled)
                y_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
                
                st.session_state.rf_model = rf_model
                st.session_state.rf_y_test = y_test
                st.session_state.rf_y_pred = y_pred
                st.session_state.rf_y_pred_proba = y_pred_proba
                st.session_state.rf_X = X
                st.success("✅ Model trained!")
        
        if 'rf_model' in st.session_state:
            y_test = st.session_state.rf_y_test
            y_pred = st.session_state.rf_y_pred
            y_pred_proba = st.session_state.rf_y_pred_proba
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
            col2.metric("Precision", f"{precision_score(y_test, y_pred):.3f}")
            col3.metric("Recall", f"{recall_score(y_test, y_pred):.3f}")
            col4.metric("F1-Score", f"{f1_score(y_test, y_pred):.3f}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_confusion_matrix_plot(y_test, y_pred, "Confusion Matrix"), use_container_width=True)
            with col2:
                st.plotly_chart(create_roc_curve_plot(y_test, y_pred_proba, "ROC Curve"), use_container_width=True)
    
    # XGBOOST TAB
    with tab_xgb:
        st.header("XGBoost Model")
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("XGBoost Parameters")
        xgb_n_estimators = st.sidebar.slider("Number of Estimators", 50, 500, 100, 50, key="xgb_n")
        xgb_max_depth = st.sidebar.slider("Max Depth", 3, 20, 6, key="xgb_d")
        xgb_learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.3, 0.1, 0.01, key="xgb_lr")
        
        if st.button("Train XGBoost Model", key="train_xgb"):
            with st.spinner("Training..."):
                X, y = preprocess_data(df, target_col)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                xgb_model = xgb.XGBClassifier(n_estimators=xgb_n_estimators, max_depth=xgb_max_depth,
                                            learning_rate=xgb_learning_rate, random_state=random_state,
                                            use_label_encoder=False, eval_metric='logloss')
                xgb_model.fit(X_train_scaled, y_train)
                
                y_pred = xgb_model.predict(X_test_scaled)
                y_pred_proba = xgb_model.predict_proba(X_test_scaled)[:, 1]
                
                st.session_state.xgb_model = xgb_model
                st.session_state.xgb_y_test = y_test
                st.session_state.xgb_y_pred = y_pred
                st.session_state.xgb_y_pred_proba = y_pred_proba
                st.success("✅ Model trained!")
        
        if 'xgb_model' in st.session_state:
            y_test = st.session_state.xgb_y_test
            y_pred = st.session_state.xgb_y_pred
            y_pred_proba = st.session_state.xgb_y_pred_proba
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
            col2.metric("Precision", f"{precision_score(y_test, y_pred):.3f}")
            col3.metric("Recall", f"{recall_score(y_test, y_pred):.3f}")
            col4.metric("F1-Score", f"{f1_score(y_test, y_pred):.3f}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_confusion_matrix_plot(y_test, y_pred, "Confusion Matrix"), use_container_width=True)
            with col2:
                st.plotly_chart(create_roc_curve_plot(y_test, y_pred_proba, "ROC Curve"), use_container_width=True)
    
    # NEURAL NETWORK TAB
    with tab_nn:
        st.header("Neural Network Model")
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("Neural Network Parameters")
        nn_hidden_layers = st.sidebar.text_input("Hidden Layers (comma separated)", "100,50", key="nn_layers")
        nn_activation = st.sidebar.selectbox("Activation", ['relu', 'tanh', 'logistic'], key="nn_act")
        nn_learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.001, 0.0001, key="nn_lr")
        nn_max_iter = st.sidebar.slider("Max Iterations", 100, 1000, 200, 100, key="nn_iter")
        
        if st.button("Train Neural Network Model", key="train_nn"):
            with st.spinner("Training..."):
                X, y = preprocess_data(df, target_col)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                hidden_layer_sizes = tuple([int(x.strip()) for x in nn_hidden_layers.split(',')])
                nn_model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes, activation=nn_activation,
                                       learning_rate_init=nn_learning_rate, max_iter=nn_max_iter, random_state=random_state)
                nn_model.fit(X_train_scaled, y_train)
                
                y_pred = nn_model.predict(X_test_scaled)
                y_pred_proba = nn_model.predict_proba(X_test_scaled)[:, 1]
                
                st.session_state.nn_model = nn_model
                st.session_state.nn_y_test = y_test
                st.session_state.nn_y_pred = y_pred
                st.session_state.nn_y_pred_proba = y_pred_proba
                st.success("✅ Model trained!")
        
        if 'nn_model' in st.session_state:
            y_test = st.session_state.nn_y_test
            y_pred = st.session_state.nn_y_pred
            y_pred_proba = st.session_state.nn_y_pred_proba
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
            col2.metric("Precision", f"{precision_score(y_test, y_pred):.3f}")
            col3.metric("Recall", f"{recall_score(y_test, y_pred):.3f}")
            col4.metric("F1-Score", f"{f1_score(y_test, y_pred):.3f}")
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_confusion_matrix_plot(y_test, y_pred, "Confusion Matrix"), use_container_width=True)
            with col2:
                st.plotly_chart(create_roc_curve_plot(y_test, y_pred_proba, "ROC Curve"), use_container_width=True)
    
    # ENSEMBLE TAB
    with tab_ensemble:
        st.header("Ensemble Model")
        st.write("Combines all three models using voting for superior performance")
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("Ensemble Parameters")
        voting_type = st.sidebar.selectbox("Voting Type", ['soft', 'hard'])
        
        if st.button("Train Ensemble Model", key="train_ensemble"):
            with st.spinner("Training..."):
                X, y = preprocess_data(df, target_col)
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                rf_model = RandomForestClassifier(n_estimators=100, random_state=random_state)
                xgb_model = xgb.XGBClassifier(n_estimators=100, random_state=random_state, use_label_encoder=False, eval_metric='logloss')
                nn_model = MLPClassifier(hidden_layer_sizes=(100, 50), random_state=random_state, max_iter=200)
                
                ensemble_model = VotingClassifier(estimators=[('rf', rf_model), ('xgb', xgb_model), ('nn', nn_model)], voting=voting_type)
                ensemble_model.fit(X_train_scaled, y_train)
                
                y_pred = ensemble_model.predict(X_test_scaled)
                if voting_type == 'soft':
                    y_pred_proba = ensemble_model.predict_proba(X_test_scaled)[:, 1]
                else:
                    y_pred_proba = np.mean([m.predict_proba(X_test_scaled)[:, 1] for _, m in ensemble_model.named_estimators_.items()], axis=0)
                
                st.session_state.ensemble_model = ensemble_model
                st.session_state.ensemble_y_test = y_test
                st.session_state.ensemble_y_pred = y_pred
                st.session_state.ensemble_y_pred_proba = y_pred_proba
                st.success("Ensemble model trained successfully!")
        
        if 'ensemble_model' in st.session_state:
            y_test = st.session_state.ensemble_y_test
            y_pred = st.session_state.ensemble_y_pred
            y_pred_proba = st.session_state.ensemble_y_pred_proba
            
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
            col2.metric("Precision", f"{precision_score(y_test, y_pred):.3f}")
            col3.metric("Recall", f"{recall_score(y_test, y_pred):.3f}")
            col4.metric("F1-Score", f"{f1_score(y_test, y_pred):.3f}")
            
            # Model Comparison
            st.subheader("Model Comparison")
            comparison_data = []
            if 'rf_y_pred' in st.session_state and len(st.session_state.rf_y_test) == len(y_test):
                comparison_data.append({'Model': 'Random Forest', 'Accuracy': accuracy_score(st.session_state.rf_y_test, st.session_state.rf_y_pred),
                                       'F1-Score': f1_score(st.session_state.rf_y_test, st.session_state.rf_y_pred)})
            if 'xgb_y_pred' in st.session_state and len(st.session_state.xgb_y_test) == len(y_test):
                comparison_data.append({'Model': 'XGBoost', 'Accuracy': accuracy_score(st.session_state.xgb_y_test, st.session_state.xgb_y_pred),
                                       'F1-Score': f1_score(st.session_state.xgb_y_test, st.session_state.xgb_y_pred)})
            if 'nn_y_pred' in st.session_state and len(st.session_state.nn_y_test) == len(y_test):
                comparison_data.append({'Model': 'Neural Network', 'Accuracy': accuracy_score(st.session_state.nn_y_test, st.session_state.nn_y_pred),
                                       'F1-Score': f1_score(st.session_state.nn_y_test, st.session_state.nn_y_pred)})
            comparison_data.append({'Model': 'Ensemble', 'Accuracy': accuracy_score(y_test, y_pred), 'F1-Score': f1_score(y_test, y_pred)})
            
            comparison_df = pd.DataFrame(comparison_data)
            fig = px.bar(comparison_df, x='Model', y=['Accuracy', 'F1-Score'], barmode='group', title='Model Performance Comparison')
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(create_confusion_matrix_plot(y_test, y_pred, "Confusion Matrix"), use_container_width=True)
            with col2:
                st.plotly_chart(create_roc_curve_plot(y_test, y_pred_proba, "ROC Curve"), use_container_width=True)

if __name__ == "__main__":
    main()
