import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="Market Sentinel - Anomaly Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
    }
    div.stButton > button:hover {
        background-color: #FF6B6B;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://via.placeholder.com/150?text=Market+Sentinel", width=150)
    st.title("Market Sentinel")
    st.markdown("---")
    
    # File upload
    uploaded_file = st.file_uploader("📊 Upload Market Data", type=["csv", "xlsx", "xls"], help="Upload your market data file (CSV or Excel)")
    
    # Settings
    contamination = st.slider("Anomaly Sensitivity", min_value=0.01, max_value=0.2, value=0.1, help="Adjust the sensitivity of anomaly detection")

# Load data function
@st.cache_data
def load_data(file):
    if file.name.endswith('.csv'):
        data = pd.read_csv(file)
    elif file.name.endswith(('.xls', '.xlsx')):
        data = pd.read_excel(file)
    else:
        raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")
    
    return data

# Train model function
@st.cache_resource
def train_anomaly_detector(data, contamination):
    # Handle missing values
    data = data.fillna(data.mean())
    
    # Standardizing the data (works for any numeric dataset)
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.select_dtypes(include=[np.number]))  # Only numerical columns
    
    # Train IsolationForest for anomaly detection
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(scaled_data)
    
    return model, scaler

# Predict anomalies
def predict_anomalies(model, scaler, data):
    # Scaling the data before prediction
    scaled_data = scaler.transform(data.select_dtypes(include=[np.number]))  # Only numerical columns
    return model.predict(scaled_data) == -1

# Plot anomalies
def plot_anomaly_timeline(data, anomalies):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data.iloc[:, 0], name='Data', line=dict(color='blue', width=1)))  # Plots the first column as data
    fig.add_trace(go.Scatter(x=data.index[anomalies], y=data.iloc[:, 0][anomalies], mode='markers', name='Anomalies', marker=dict(color='red', size=8)))
    fig.update_layout(title='Anomaly Detection', xaxis_title='Index', yaxis_title='Value', height=500)
    return fig

# Plot decision function (anomaly score)
def plot_decision_function(data, model, scaler):
    # Get the anomaly scores
    scaled_data = scaler.transform(data.select_dtypes(include=[np.number]))  # Only numerical columns
    scores = model.decision_function(scaled_data)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=scores, mode='lines', name='Anomaly Score', line=dict(color='purple', width=2)))
    fig.update_layout(title="Anomaly Score", xaxis_title='Index', yaxis_title='Anomaly Score', height=400)
    return fig

# Main content
st.title("🎯 Market Anomaly Detection System")

if uploaded_file is not None:
    try:
        with st.spinner("Processing data..."):
            # Load the dataset
            data = load_data(uploaded_file)
            
            # Display basic summary of the dataset
            st.write("Dataset Summary:")
            st.write(data.describe())
            
            # Train the anomaly detection model
            model, scaler = train_anomaly_detector(data, contamination)
            
            # Predict anomalies in the data
            anomalies = predict_anomalies(model, scaler, data)
            
            # Display the anomaly timeline plot
            st.plotly_chart(plot_anomaly_timeline(data, anomalies), use_container_width=True)
            
            # Display the decision function plot
            st.plotly_chart(plot_decision_function(data, model, scaler), use_container_width=True)
            
            # Show anomalies
            st.write(f"Number of anomalies detected: {anomalies.sum()}")
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.markdown("""
    ## 👋 Welcome to Market Sentinel!
    Upload your market data to:
    - 🎯 Detect market anomalies
    - 📊 Visualize market patterns
    - 🤖 Get AI-powered advice
    - 📈 Monitor market health
    """)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center'><p>Market Sentinel - Advanced Market Anomaly Detection</p><p style='color: #666;'>Built with Streamlit</p></div>", unsafe_allow_html=True)
