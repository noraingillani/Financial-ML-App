from streamlit.runtime.scriptrunner import add_script_run_ctx
import streamlit as st
st.set_page_config(
    page_title="💰 Smart Budget Tracker | AF3005",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded"
)
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error
import plotly.express as px
import matplotlib.pyplot as pl

# %% [markdown]
## 2. Sidebar: Data Input & Settings

st.sidebar.title("Data Input & Settings")
# File uploader for Kragle dataset
dataset_file = st.sidebar.file_uploader("Upload Kragle dataset", type=["csv", "xlsx"])
# Ticker input
ticker = st.sidebar.text_input("Yahoo Finance Ticker", value="AAPL")
# Fetch button
fetch_data = st.sidebar.button("Fetch Data")
# Auto-refresh interval (in seconds)
update_freq = st.sidebar.slider("Auto-refresh interval (sec)", 30, 600, 300, 30)

# Session state initialization
if 'data' not in st.session_state:
    st.session_state.data = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'features' not in st.session_state:
    st.session_state.features = []
if 'model_type' not in st.session_state:
    st.session_state.model_type = None

# %% [markdown]
## 3. Step-by-Step ML Pipeline

st.header("Machine Learning Pipeline")

# 1. Load Data
if st.button("1. Load Data"):
    # From uploaded file
    if dataset_file:
        if dataset_file.name.endswith('.csv'):
            df = pd.read_csv(dataset_file)
        else:
            df = pd.read_excel(dataset_file)
        st.success("Kragle dataset loaded!")
    # Or fetch via yfinance with TTL cache
    else:
        @st.cache(ttl=update_freq)
        def load_yf(ticker):
            df = yf.download(ticker, period="1y", interval="1d")
            df.reset_index(inplace=True)
            return df
        df = load_yf(ticker)
        st.success(f"Yahoo Finance data for {ticker} loaded!")
    st.session_state.data = df
    st.dataframe(df.head())

# 2. Preprocessing
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Display missing value statistics
    missing = df.isnull().sum()
    st.write("Missing values per column:", missing)
    # Drop missing rows for simplicity
    df_clean = df.dropna()
    return df_clean

if st.button("2. Preprocessing"):
    if st.session_state.data is None:
        st.error("Load data first.")
    else:
        df_clean = preprocess(st.session_state.data)
        st.session_state.data = df_clean
        st.success("Preprocessing complete: missing values dropped.")
        st.dataframe(df_clean.head())

# 3. Feature Engineering
if st.button("3. Feature Engineering"):
    if st.session_state.data is None:
        st.error("Run preprocessing first.")
    else:
        df = st.session_state.data.copy()
        # Create target: 1 if next-day close > today
        df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
        # Simple features: 10-day MA and daily return
        df['MA10'] = df['Close'].rolling(window=10).mean()
        df['Return'] = df['Close'].pct_change()
        df = df.dropna()
        st.session_state.features = ['MA10', 'Return']
        st.session_state.data = df
        st.success("Features engineered: MA10, Return, Target.")
        st.dataframe(df[['MA10', 'Return', 'Target']].head())

# 4. Train/Test Split
if st.button("4. Train/Test Split"):
    df = st.session_state.data
    if df is None or 'Target' not in df.columns:
        st.error("Engineer features first.")
    else:
        X = df[st.session_state.features]
        y = df['Target']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        st.session_state.X_train = X_train
        st.session_state.X_test = X_test
        st.session_state.y_train = y_train
        st.session_state.y_test = y_test
        fig = px.pie(names=['Train', 'Test'], values=[len(X_train), len(X_test)], title='Train/Test Split')
        st.plotly_chart(fig)

# 5. Model Training
if st.button("5. Model Training"):
    if 'X_train' not in st.session_state:
        st.error("Split data first.")
    else:
        # Dynamic model selection
        model_choice = st.sidebar.selectbox("Choose Model", ['Logistic Regression', 'Linear Regression', 'K-Means Clustering'])
        st.session_state.model_type = model_choice
        if model_choice == 'Logistic Regression':
            model = LogisticRegression()
        elif model_choice == 'Linear Regression':
            model = LinearRegression()
        else:
            model = KMeans(n_clusters=2)
        # Fit model
        if model_choice == 'K-Means Clustering':
            model.fit(st.session_state.X_train)
        else:
            model.fit(st.session_state.X_train, st.session_state.y_train)
        st.session_state.model = model
        st.success(f"{model_choice} trained!")

# 6. Evaluation
if st.button("6. Evaluation"):
    model = st.session_state.model
    mtype = st.session_state.model_type
    if model is None:
        st.error("Train a model first.")
    else:
        if mtype in ['Logistic Regression', 'Linear Regression']:
            X_test = st.session_state.X_test
            y_test = st.session_state.y_test
            y_pred = model.predict(X_test)
            if mtype == 'Logistic Regression':
                # Classification metrics
                st.write(classification_report(y_test, y_pred))
                cm = confusion_matrix(y_test, y_pred)
                fig = px.imshow(cm, text_auto=True, title='Confusion Matrix')
                st.plotly_chart(fig)
                st.write("Accuracy:", accuracy_score(y_test, y_pred))
            else:
                # Regression metrics
                mse = mean_squared_error(y_test, y_pred)
                st.write(f"Mean Squared Error: {mse:.4f}")
                fig = px.scatter(x=y_test, y=y_pred,
                                 labels={'x':'Actual','y':'Predicted'},
                                 title='Actual vs Predicted')
                st.plotly_chart(fig)
        else:
            # K-Means visualization
            labels = model.labels_
            df_vis = st.session_state.X_test.copy()
            df_vis['Cluster'] = labels
            fig = px.scatter(df_vis, x=st.session_state.features[0], y=st.session_state.features[1],
                             color='Cluster', title='Cluster Visualization')
            st.plotly_chart(fig)

# 7. Results Visualization & Downloads
if st.button("7. Results Visualization"):
    model = st.session_state.model
    mtype = st.session_state.model_type
    X_test = st.session_state.X_test
    if model is None:
        st.error("Train and evaluate a model first.")
    else:
        if mtype in ['Logistic Regression', 'Linear Regression']:
            y_pred = model.predict(X_test)
            results_df = X_test.copy()
            results_df['Prediction'] = y_pred
            st.write(results_df.head())
            # Download predictions
            csv = results_df.to_csv(index=False).encode()
            st.download_button("Download Predictions as CSV", data=csv, file_name="predictions.csv")
            # Feature importance
            coef = model.coef_[0] if mtype=='Logistic Regression' else model.coef_
            fig = px.bar(x=st.session_state.features, y=coef, title='Feature Importance')
            st.plotly_chart(fig)
        else:
            # K-Means download
            labels = model.labels_
            dl_df = X_test.copy()
            dl_df['Cluster'] = labels
            st.write(dl_df.head())
            csv = dl_df.to_csv(index=False).encode()
            st.download_button("Download Clusters as CSV", data=csv, file_name="clusters.csv")
        # Download trained model
        import io
        buf = io.BytesIO()
        joblib.dump(model, buf)
        buf.seek(0)
        st.download_button("Download Trained Model (.pkl)", data=buf, file_name="trained_model.pkl")

# %% [markdown]
## 4. Themed GIFs and Images

# Display a finance-themed GIF at start/end
st.image("https://media.giphy.com/media/xT9IgG50Fb7Mi0prBC/giphy.gif", use_column_width=True)
