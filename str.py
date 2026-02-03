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

st.set_page_config(page_title="Fraud Click Detection", page_icon="F", layout="wide")

if 'current_dataset' not in st.session_state:
    st.session_state.current_dataset = "fraud_dataset_1.csv"

@st.cache_data
def load(file_path):
    return pd.read_csv(file_path)

@st.cache_data
def preprocess(df, target='is_fraud'):
    data = df.copy()
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    if target in numeric_cols:
        numeric_cols.remove(target)
    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].median())
    categorical = data.select_dtypes(include=['object']).columns.tolist()
    for col in categorical:
        data[col] = pd.Categorical(data[col]).codes
    if target in data.columns:
        x = data.drop(target, axis=1)
        y = data[target]
    else:
        x = data
        y = np.random.randint(0, 2, size=len(data))
    return x, y

def ConfusionMatrix(ytrue, ypred, title):
    cm = confusion_matrix(ytrue, ypred)
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted Legitimate', 'Predicted Fraud'],
        y=['Actual Legitimate', 'Actual Fraud'],
        colorscale='Purples',
        text=cm,
        texttemplate='%{text}',
        textfont={"size": 16}
    ))
    fig.update_layout(title=title, xtitle="Predicted", ytitle="Actual", height=400)
    return fig

def rocCurve(ytrue, ypredP, title):
    fpr, tpr, _ = roc_curve(ytrue, ypredP)
    auc = roc_auc_score(ytrue, ypredP)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC (AUC={auc:.3f})'))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash')))
    fig.update_layout(title=title, xtitle='FPR', ytitle='TPR', height=400)
    return fig

def main():
    st.title("Fraud Click Detection System")
    st.write("Advanced Machine Learning for Click Fraud Prevention")

    with st.sidebar:
        st.header("Data Configuration")
        dataset_choice = st.selectbox(
            "Choose Dataset:",
            ["fraud_dataset_1.csv", "fraud_dataset_2.csv", "fraud_dataset_3.csv"]
        )
        st.session_state.current_dataset = dataset_choice
        st.success(f"Using {dataset_choice}")

        st.markdown("---")
        st.header("Model Settings")
        test_size = st.slider("Test Set Size", 0.1, 0.5, 0.3, 0.05)
        random_state = st.number_input("Random State", value=42, min_value=0)

    if os.path.exists(st.session_state.current_dataset):
        df = load(st.session_state.current_dataset)
        fraud_cols = [col for col in df.columns if 'fraud' in col.lower() or 'label' in col.lower()]
        target = fraud_cols[0] if fraud_cols else df.columns[-1]
    else:
        st.error("Dataset not found. Please run: python generate_datasets.py")
        return

    tab0, tab1, tab2, tab3, tab4 = st.tabs(
        ["Home", "Random Forest", "XGBoost", "Neural Network", "Ensemble"]
    )

    with tab0:
        st.header("Project Overview")
        st.write("""
        This application uses machine learning to detect fraudulent clicks in online advertising.
        """)
        col1, col2 = st.columns([2, 1])
        with col1:
            st.metric("Total Records", f"{len(df):,}")
            st.metric("Features", df.shape[1] - 1)
            fraud_rate = (df[target].sum() / len(df)) * 100
            st.metric("Fraud Rate", f"{fraud_rate:.2f}%")
        with col2:
            st.dataframe(df.head(), use_container_width=True)

    with tab1:
        st.header("Random Forest Model")

        estimators = st.sidebar.slider("Number of Trees", 50, 500, 100, 50)
        maxdepth = st.sidebar.slider("Max Depth", 5, 50, 10, 5)
        samplesplit = st.sidebar.slider("Min Samples Split", 2, 20, 2)

        if st.button("Train Random Forest Model", key="train_rf"):
            x, y = preprocess(df, target)
            xtrain, xtest, ytrain, ytest = train_test_split(
                x, y, test_size=test_size, random_state=random_state, stratify=y
            )
            scaler = StandardScaler()
            xtrainS = scaler.fit_transform(xtrain)
            xtestS = scaler.transform(xtest)

            model1 = RandomForestClassifier(
                n_estimators=estimators,
                max_depth=maxdepth,
                min_samples_split=samplesplit,
                random_state=random_state,
                n_jobs=-1
            )
            model1.fit(xtrainS, ytrain)

            ypred = model1.predict(xtestS)
            ypredP = model1.predict_proba(xtestS)[:, 1]

            st.session_state.ytest = ytest
            st.session_state.ypred = ypred
            st.session_state.ypredP = ypredP
            st.success("BYY NEPTUNEE!")

        if 'ypred' in st.session_state:
            ytest = st.session_state.ytest
            ypred = st.session_state.ypred
            ypredP = st.session_state.ypredP

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy", f"{accuracy_score(ytest, ypred):.3f}")
            col2.metric("Precision", f"{precision_score(ytest, ypred):.3f}")
            col3.metric("Recall", f"{recall_score(ytest, ypred):.3f}")
            col4.metric("F1-Score", f"{f1_score(ytest, ypred):.3f}")

            col1, col2 = st.columns(2)
            with col1:
                st.plotly_chart(ConfusionMatrix(ytest, ypred, "Confusion Matrix"), use_container_width=True)
            with col2:
                st.plotly_chart(rocCurve(ytest, ypredP, "ROC Curve"), use_container_width=True)

    with tab2:
        st.header("XGBoost Model")

        estimators = st.sidebar.slider("Number of Estimators", 50, 500, 100, 50, key="xgb_n")
        maxdepth = st.sidebar.slider("Max Depth", 3, 20, 6, key="xgb_d")
        learningrate = st.sidebar.slider("Learning Rate", 0.01, 0.3, 0.1, 0.01, key="xgb_lr")

        if st.button("Train XGBoost Model", key="train_xgb"):
            x, y = preprocess(df, target)
            xtrain, xtest, ytrain, ytest = train_test_split(
                x, y, test_size=test_size, random_state=random_state, stratify=y
            )
            scaler = StandardScaler()
            xtrainS = scaler.fit_transform(xtrain)
            xtestS = scaler.transform(xtest)

            model2 = xgb.XGBClassifier(
                n_estimators=estimators,
                max_depth=maxdepth,
                learning_rate=learningrate,
                random_state=random_state,
                use_label_encoder=False,
                eval_metric='logloss'
            )
            model2.fit(xtrainS, ytrain)

            ypred = model2.predict(xtestS)
            ypredP = model2.predict_proba(xtestS)[:, 1]

            st.session_state.ytest = ytest
            st.session_state.ypred = ypred
            st.session_state.ypredP = ypredP
            st.success("BYY NEPTUNE!")

    with tab3:
        st.header("Neural Network Model")

        nn_hidden_layers = st.sidebar.text_input("Hidden Layers (comma separated)", "100,50")
        nn_activation = st.sidebar.selectbox("Activation", ['relu', 'tanh', 'logistic'])
        nn_learning_rate = st.sidebar.slider("Learning Rate", 0.0001, 0.01, 0.001, 0.0001)
        nn_max_iter = st.sidebar.slider("Max Iterations", 100, 1000, 200, 100)

        if st.button("Train Neural Network Model", key="train_nn"):
            x, y = preprocess(df, target)
            xtrain, xtest, ytrain, ytest = train_test_split(
                x, y, test_size=test_size, random_state=random_state, stratify=y
            )
            scaler = StandardScaler()
            xtrainS = scaler.fit_transform(xtrain)
            xtestS = scaler.transform(xtest)

            hidden_layer_sizes = tuple(int(v.strip()) for v in nn_hidden_layers.split(','))
            nn_model = MLPClassifier(
                hidden_layer_sizes=hidden_layer_sizes,
                activation=nn_activation,
                learning_rate_init=nn_learning_rate,
                max_iter=nn_max_iter,
                random_state=random_state
            )
            nn_model.fit(xtrainS, ytrain)

            ypred = nn_model.predict(xtestS)
            ypredP = nn_model.predict_proba(xtestS)[:, 1]

            st.session_state.ytest = ytest
            st.session_state.ypred = ypred
            st.session_state.ypredP = ypredP
            st.success("BYY NEPTUNEE!")

    with tab4:
        st.header("Ensemble Model")

        voting_type = st.sidebar.selectbox("Voting Type", ['soft', 'hard'])

        if st.button("Train Ensemble Model", key="train_ensemble"):
            x, y = preprocess(df, target)
            xtrain, xtest, ytrain, ytest = train_test_split(
                x, y, test_size=test_size, random_state=random_state, stratify=y
            )
            scaler = StandardScaler()
            xtrainS = scaler.fit_transform(xtrain)
            xtestS = scaler.transform(xtest)

            rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
            xg = xgb.XGBClassifier(n_estimators=100, random_state=random_state, use_label_encoder=False, eval_metric='logloss')
            nn = MLPClassifier(hidden_layer_sizes=(100, 50), random_state=random_state, max_iter=200)

            ensemble = VotingClassifier(
                estimators=[('rf', rf), ('xgb', xg), ('nn', nn)],
                voting=voting_type
            )
            ensemble.fit(xtrainS, ytrain)

            ypred = ensemble.predict(xtestS)
            ypredP = ensemble.predict_proba(xtestS)[:, 1]

            st.session_state.ytest = ytest
            st.session_state.ypred = ypred
            st.session_state.ypredP = ypredP
            st.success("Ensemble model trained successfully!")

if __name__ == "__main__":
    main()
