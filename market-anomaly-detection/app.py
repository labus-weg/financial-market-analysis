import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from model import MarketAnomalyDetector
import time

# Page configuration
st.set_page_config(
    page_title="Market Sentinel - Anomaly Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

def load_and_prepare_data(file):
    """Load and prepare data with proper error handling"""
    try:
        # Read the file
        if file.name.endswith('.csv'):
            df = pd.read_csv(file)
        elif file.name.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file)
        else:
            raise ValueError("Unsupported file format")
        
        # Convert date column if it exists
        date_columns = df.filter(like='date').columns
        if len(date_columns) > 0:
            df[date_columns[0]] = pd.to_datetime(df[date_columns[0]])
            df.set_index(date_columns[0], inplace=True)
        
        # Ensure we have numeric data
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            raise ValueError("No numeric columns found in the data")
        
        # Basic feature engineering
        features = pd.DataFrame()
        
        # If we have typical financial data columns
        if 'Close' in df.columns:
            features['returns'] = df['Close'].pct_change()
            features['volatility'] = features['returns'].rolling(window=20).std()
        else:
            # Use the first numeric column as the main feature
            main_col = numeric_df.columns[0]
            features['value'] = numeric_df[main_col]
            features['change'] = features['value'].pct_change()
            features['volatility'] = features['change'].rolling(window=20).std()
        
        # Add any additional numeric columns as features
        for col in numeric_df.columns:
            if col not in ['Close']:
                features[f'{col}_raw'] = numeric_df[col]
        
        # Drop any NaN values
        features = features.dropna()
        
        return features, df
        
    except Exception as e:
        raise Exception(f"Error processing data: {str(e)}")

def plot_anomalies(data, predictions, title="Anomaly Detection Results"):
    """Create plotly visualization of anomalies"""
    fig = go.Figure()
    
    # Get the main value column (Close or first numeric column)
    value_col = 'Close' if 'Close' in data.columns else data.select_dtypes(include=[np.number]).columns[0]
    
    # Add main line
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data[value_col],
        name='Value',
        line=dict(color='blue', width=1)
    ))
    
    # Add anomaly points
    anomaly_indices = predictions == 1
    fig.add_trace(go.Scatter(
        x=data.index[anomaly_indices],
        y=data[value_col][anomaly_indices],
        mode='markers',
        name='Anomalies',
        marker=dict(color='red', size=10, symbol='x')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time',
        yaxis_title='Value',
        height=500
    )
    
    return fig

# Sidebar
with st.sidebar:
    st.title("Market Sentinel")
    st.markdown("---")
    
    uploaded_file = st.file_uploader(
        "📊 Upload Market Data",
        type=["csv", "xlsx", "xls"],
        help="Upload your market data file (CSV or Excel)"
    )
    
    if uploaded_file:
        st.success("File uploaded successfully!")
    
    st.markdown("### ⚙️ Settings")
    contamination = st.slider(
        "Anomaly Sensitivity",
        min_value=0.01,
        max_value=0.2,
        value=0.1,
        help="Adjust the sensitivity of anomaly detection"
    )

# Main content
st.title("🎯 Market Anomaly Detection System")

if uploaded_file is not None:
    try:
        # Process data
        with st.spinner("Processing data..."):
            features, original_data = load_and_prepare_data(uploaded_file)
            
            # Initialize and train model
            model = MarketAnomalyDetector(contamination=contamination)
            model.fit(features)
            
            # Get predictions
            predictions = model.predict(features)
            scores = model.anomaly_scores(features)
        
        # Display results in tabs
        tab1, tab2 = st.tabs(["📈 Analysis", "📊 Details"])
        
        with tab1:
            # Summary metrics
            col1, col2 = st.columns(2)
            with col1:
                anomaly_count = sum(predictions == 1)
                st.metric("Anomalies Detected", anomaly_count)
            with col2:
                anomaly_ratio = (anomaly_count / len(predictions)) * 100
                st.metric("Anomaly Percentage", f"{anomaly_ratio:.1f}%")
            
            # Plot
            st.plotly_chart(plot_anomalies(original_data, predictions), use_container_width=True)
            
            # Market status and advice
            status = "🚨 Anomaly Detected" if predictions[-1] == 1 else "✅ Normal Conditions"
            st.markdown(f"### Current Status: {status}")
            
            if predictions[-1] == 1:
                st.markdown("""
                **Recommended Actions:**
                - Review portfolio risk exposure
                - Consider reducing position sizes
                - Monitor market conditions closely
                """)
            else:
                st.markdown("""
                **Recommended Actions:**
                - Maintain normal trading operations
                - Continue regular market monitoring
                - Review portfolio as planned
                """)
        
        with tab2:
            # Detailed view
            results_df = pd.DataFrame({
                'Date': original_data.index,
                'Anomaly': predictions,
                'Anomaly Score': scores
            })
            st.dataframe(results_df)
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please check your data format and try again.")

else:
    # Welcome message
    st.markdown("""
    ## 👋 Welcome to Market Sentinel!
    
    Upload your market data to:
    - 🎯 Detect market anomalies
    - 📊 Visualize patterns
    - 📈 Get market insights
    
    ### Supported Data Formats:
    - CSV files
    - Excel files (xls, xlsx)
    
    ### Required Columns:
    - Date/Time column
    - At least one numeric column (e.g., Close price, value)
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Market Sentinel - Advanced Market Anomaly Detection</p>
    </div>
    """,
    unsafe_allow_html=True
)