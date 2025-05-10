# %% [markdown]
# AF3005 – Machine Learning Application
# **Author:** Your Name  
# **Universal ML Processor**  

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import yfinance as yf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, r2_score, mean_absolute_error,
    confusion_matrix, classification_report
)
import plotly.express as px
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
import shap

# Configure app
st.set_page_config(
    page_title="Universal ML Processor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'target' not in st.session_state:
    st.session_state.target = None
if 'problem_type' not in st.session_state:
    st.session_state.problem_type = None

# %% [markdown]
## 1. Universal Data Loader

def load_data():
    st.sidebar.header("Data Input")
    source = st.sidebar.radio("Data Source", ["Upload File", "Yahoo Finance"])
    
    if source == "Upload File":
        file = st.sidebar.file_uploader("Upload Dataset", type=["csv", "xlsx"])
        if file:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            else:
                df = pd.read_excel(file)
            return df
    else:
        ticker = st.sidebar.text_input("Enter Ticker Symbol", "AAPL")
        days = st.sidebar.slider("Historical Days", 30, 365*5, 365)
        if ticker:
            @st.cache(ttl=3600)
            def fetch_yfinance(ticker, days):
                df = yf.download(ticker, period=f"{days}d")
                df.reset_index(inplace=True)
                return df
            return fetch_yfinance(ticker, days)
    return None

# %% [markdown]
## 2. Smart Preprocessing

def auto_preprocess(df):
    st.header("Data Preprocessing")
    
    # Display raw data
    with st.expander("Raw Data Preview"):
        st.dataframe(df.head())
        st.write(f"Shape: {df.shape}")
    
    # Auto-detect column types
    numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    
    # Missing value treatment
    st.subheader("Missing Values Handling")
    missing = df.isna().sum()
    st.write("Missing Values Count:", missing[missing > 0])
    
    # Numeric preprocessing
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    
    # Categorical preprocessing
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, cat_cols)])
    
    with st.spinner("Preprocessing data..."):
        processed_data = preprocessor.fit_transform(df)
    
    st.success("Preprocessing completed!")
    return preprocessor, processed_data

# %% [markdown]
## 3. Adaptive Model Training

def train_model(X, y, problem_type):
    st.sidebar.header("Model Configuration")
    
    model_type = st.sidebar.selectbox("Select Model Type",
        ["Random Forest", "Gradient Boosting", "Logistic Regression", "Linear Regression"])
    
    params = {}
    if "Forest" in model_type or "Boosting" in model_type:
        params['n_estimators'] = st.sidebar.slider("Number of Trees", 50, 500, 100)
        params['max_depth'] = st.sidebar.slider("Max Depth", 2, 20, 5)
    
    # Model selection
    if problem_type == "Regression":
        models = {
            "Random Forest": RandomForestRegressor(**params),
            "Gradient Boosting": GradientBoostingRegressor(**params),
            "Linear Regression": LinearRegression()
        }
    else:
        models = {
            "Random Forest": RandomForestClassifier(**params),
            "Gradient Boosting": GradientBoostingClassifier(**params),
            "Logistic Regression": LogisticRegression(max_iter=1000)
        }
    
    model = models[model_type]
    
    with st.spinner(f"Training {model_type}..."):
        model.fit(X, y)
    
    return model

# %% [markdown]
## 4. Intelligent Evaluation

def evaluate_model(model, X_test, y_test, problem_type):
    st.header("Model Evaluation")
    
    y_pred = model.predict(X_test)
    
    if problem_type == "Regression":
        col1, col2, col3 = st.columns(3)
        col1.metric("MSE", f"{mean_squared_error(y_test, y_pred):.2f}")
        col2.metric("R² Score", f"{r2_score(y_test, y_pred):.2f}")
        col3.metric("MAE", f"{mean_absolute_error(y_test, y_pred):.2f}")
        
        fig = px.scatter(x=y_test, y=y_pred, 
                        labels={'x': 'Actual', 'y': 'Predicted'},
                        title="Actual vs Predicted Values")
        st.plotly_chart(fig)
    else:
        st.subheader("Classification Metrics")
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted')
        rec = recall_score(y_test, y_pred, average='weighted')
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{acc:.2%}")
        col2.metric("Precision", f"{prec:.2%}")
        col3.metric("Recall", f"{rec:.2%}")
        
        st.subheader("Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig = px.imshow(cm, text_auto=True, 
                       labels=dict(x="Predicted", y="Actual"),
                       title="Confusion Matrix")
        st.plotly_chart(fig)
    
    # Feature Importance
    if hasattr(model, 'feature_importances_'):
        st.subheader("Feature Importance")
        importance = pd.DataFrame({
            'Feature': X_test.columns,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        fig = px.bar(importance, x='Importance', y='Feature', 
                     orientation='h', title="Feature Importance")
        st.plotly_chart(fig)
    
    # SHAP Explanations
    st.subheader("Model Explanations (SHAP)")
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    fig, ax = plt.subplots()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    st.pyplot(fig)

# %% [markdown]
## 5. Main Application Flow

def main():
    st.title("Universal Machine Learning Processor")
    
    # Step 1: Load Data
    df = load_data()
    if df is not None:
        st.session_state.data = df
    
    if st.session_state.data is not None:
        # Step 2: Preprocessing
        preprocessor, processed_data = auto_preprocess(st.session_state.data)
        
        # Step 3: Target Selection
        target = st.selectbox("Select Target Variable", 
                             st.session_state.data.columns)
        st.session_state.target = target
        
        # Detect problem type
        if pd.api.types.is_numeric_dtype(st.session_state.data[target]):
            unique_values = st.session_state.data[target].nunique()
            st.session_state.problem_type = "Regression" if unique_values > 10 else "Classification"
        else:
            st.session_state.problem_type = "Classification"
        
        # Step 4: Train/Test Split
        test_size = st.slider("Test Size Ratio", 0.1, 0.5, 0.2)
        X_train, X_test, y_train, y_test = train_test_split(
            processed_data, st.session_state.data[target],
            test_size=test_size, random_state=42
        )
        
        # Step 5: Model Training
        if st.button("Train Model"):
            model = train_model(X_train, y_train, st.session_state.problem_type)
            st.session_state.model = model
        
        # Step 6: Evaluation
        if st.session_state.model is not None:
            evaluate_model(st.session_state.model, X_test, y_test, 
                          st.session_state.problem_type)
            
            # Step 7: Save Model
            st.download_button(
                label="Download Trained Model",
                data=joblib.dump(st.session_state.model),
                file_name="trained_model.pkl",
                mime="application/octet-stream"
            )

if __name__ == "__main__":
    main()
