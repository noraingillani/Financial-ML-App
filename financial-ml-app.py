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
dataset_file = st.sidebar.file_uploader("Upload Kaggle dataset", type=["csv", "xlsx"])
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

# Enhanced Preprocessing
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    with st.expander("Preprocessing Details", expanded=True):
        st.subheader("Data Cleaning Pipeline")
        
        # Missing values handling
        missing = df.isnull().sum()
        st.write("Missing Values Before Treatment:", missing)
        
        # Numeric imputation
        num_cols = df.select_dtypes(include=np.number).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        
        # Categorical handling
        cat_cols = df.select_dtypes(exclude=np.number).columns
        if not cat_cols.empty:
            df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
            st.write("Created dummy variables for:", list(cat_cols))
        
        # Outlier detection
        st.write("Outlier Analysis (Z-scores > 3):")
        z = np.abs((df[num_cols] - df[num_cols].mean())/df[num_cols].std())
        st.write(z[z > 3].count())
        
        st.success("Preprocessing complete!")
        return df

# 3. Feature Engineering for Mental Health Dataset
if st.button("3. Feature Engineering"):
    if st.session_state.data is None:
        st.error("Run preprocessing first.")
    else:
        df = st.session_state.data.copy()
        
        # Check for required columns
        required_columns = {'mental_health_score', 'stress_level', 'sleep_quality'}
        missing_columns = required_columns - set(df.columns)
        
        if missing_columns:
            st.error(f"Missing required columns: {', '.join(missing_columns)}")
            st.write("Available columns:", list(df.columns))
            st.stop()
            
        try:
            # Create target: Mental health score (regression)
            df['target'] = df['mental_health_score']
            
            # Feature 1: Total screen time
            df['total_screen_time'] = (
                df['daily_screen_time_hours'] + 
                df['phone_usage_hours'] + 
                df['laptop_usage_hours'] +
                df['tablet_usage_hours'] +
                df['tv_usage_hours']
            )
            
            # Feature 2: Sleep efficiency ratio
            df['sleep_efficiency'] = df['sleep_quality'] / df['sleep_duration_hours']
            
            # Feature 3: Stress-sleep interaction
            df['stress_sleep_interaction'] = df['stress_level'] * df['sleep_quality']
            
            # Feature 4: Healthy lifestyle score
            df['healthy_lifestyle'] = (
                df['physical_activity_hours_per_week'] +
                df['mindfulness_minutes_per_day']/60 +
                df['eats_healthy'] * 2
            )
            
            # Select final features
            st.session_state.features = [
                'total_screen_time',
                'sleep_efficiency',
                'stress_sleep_interaction',
                'healthy_lifestyle'
            ]
            
            st.session_state.data = df.dropna()
            st.success(f"Features engineered: {', '.join(st.session_state.features)}")
            st.dataframe(df[st.session_state.features + ['target']].head())
            
        except Exception as e:
            st.error(f"Error in feature engineering: {str(e)}")
            st.stop()
# 4. Updated Train/Test Split
if st.button("4. Train/Test Split"):
    if st.session_state.data is None:
        st.error("Preprocess data first.")
    else:
        df = st.session_state.data
        # Check for required components
        if 'target' not in df.columns or not st.session_state.features:
            st.error("Run feature engineering first.")
            st.write("Current dataframe columns:", list(df.columns))
            st.write("Available features:", st.session_state.features)
            st.stop()
            
        try:
            X = df[st.session_state.features]
            y = df['target']
            
            if X.empty or y.empty:
                st.error("Features/target are empty - check feature engineering")
                st.stop()
                
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=0.2, 
                random_state=42,
                stratify=y if st.session_state.model_type == 'Logistic Regression' else None
            )
            
            st.session_state.X_train = X_train
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_test = y_test
            
            fig = px.pie(names=['Train', 'Test'], 
                        values=[len(X_train), len(X_test)],
                        title=f'Train/Test Split ({len(X_train)}/{len(X_test)})')
            st.plotly_chart(fig)
            
        except Exception as e:
            st.error(f"Split failed: {str(e)}")
            st.stop()

# Enhanced Model Training
if st.button("5. Model Training"):
    if 'X_train' not in st.session_state:
        st.error("Split data first.")
    else:
        model_choice = st.sidebar.selectbox("Choose Model", [
            'Linear Regression', 
            'Random Forest', 
            'Gradient Boosting',
            'Support Vector Regression'
        ])
        
        # Model configuration
        with st.sidebar.expander("Hyperparameters"):
            if model_choice == 'Random Forest':
                n_estimators = st.slider("Trees", 50, 500, 100)
                max_depth = st.slider("Max Depth", 2, 20, 5)
                
        # Model initialization
        model_configs = {
            'Linear Regression': LinearRegression(),
            'Random Forest': RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=42
            ),
            'Gradient Boosting': GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                random_state=42
            ),
            'Support Vector Regression': SVR(kernel='rbf', C=100)
        }
        
        # Training with progress
        progress_bar = st.progress(0)
        model = model_configs[model_choice]
        
        with st.spinner(f"Training {model_choice}..."):
            model.fit(st.session_state.X_train, st.session_state.y_train)
            progress_bar.progress(100)
            
        st.session_state.model = model
        st.success(f"{model_choice} trained successfully!")

# Enhanced Evaluation
if st.button("6. Evaluation"):
    if st.session_state.model is None:
        st.error("Train a model first.")
    else:
        model = st.session_state.model
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        
        y_pred = model.predict(X_test)
        
        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        col1, col2, col3 = st.columns(3)
        col1.metric("MSE", f"{mse:.2f}")
        col2.metric("R² Score", f"{r2:.2f}")
        col3.metric("MAE", f"{mae:.2f}")
        
        # Visualization
        fig = px.scatter(
            x=y_test, y=y_pred,
            trendline="ols",
            labels={'x': 'Actual', 'y': 'Predicted'},
            title="Actual vs Predicted Values"
        )
        st.plotly_chart(fig)
        
        # Feature Importance
        if hasattr(model, 'feature_importances_'):
            importance = pd.DataFrame({
                'Feature': st.session_state.features,
                'Importance': model.feature_importances_
            }).sort_values('Importance', ascending=False)
            
            fig = px.bar(
                importance, 
                x='Importance', 
                y='Feature', 
                orientation='h',
                title="Feature Importance Analysis"
            )
            st.plotly_chart(fig)


# Enhanced Results Visualization
if st.button("7. Results Visualization"):
    if st.session_state.model is None:
        st.error("Train and evaluate first.")
    else:
        # Interactive prediction explorer
        st.subheader("Prediction Explorer")
        
        col1, col2 = st.columns(2)
        with col1:
            feature_inputs = {}
            for feature in st.session_state.features:
                data_min = st.session_state.X_train[feature].min()
                data_max = st.session_state.X_train[feature].max()
                feature_inputs[feature] = st.slider(
                    f"{feature}",
                    min_value=float(data_min),
                    max_value=float(data_max),
                    value=float(st.session_state.X_train[feature].median())
                )
        
        with col2:
            input_df = pd.DataFrame([feature_inputs])
            prediction = st.session_state.model.predict(input_df)[0]
            st.metric("Predicted Mental Health Score", 
                      f"{prediction:.1f}",
                      help="Higher scores indicate better mental health")
            
            # Explanation
            if hasattr(st.session_state.model, 'predict_proba'):
                explainer = shap.Explainer(st.session_state.model)
                shap_values = explainer(input_df)
                fig, ax = plt.subplots()
                shap.plots.waterfall(shap_values[0], show=False)
                st.pyplot(fig)
        
        # Downloadable report
        report = f"""
        Model Evaluation Report
        ------------------------
        Model Type: {st.session_state.model_type}
        Features Used: {', '.join(st.session_state.features)}
        
        Performance Metrics:
        - Mean Squared Error: {mse:.2f}
        - R² Score: {r2:.2f}
        - Mean Absolute Error: {mae:.2f}
        """
        st.download_button(
            "Download Evaluation Report",
            data=report,
            file_name="mental_health_model_report.txt"
        )
# %% [markdown]
## 4. Themed GIFs and Images

# Display a finance-themed GIF at start/end
st.image("https://media.giphy.com/media/xT9IgG50Fb7Mi0prBC/giphy.gif", use_column_width=True)
