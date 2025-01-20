import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from advisor_bot import get_advice

# Page configuration
st.set_page_config(
    page_title="Market Anomaly Detection",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI
st.markdown("""
    <style>
    /* Body and background styling */
    body {
        background-color: #121212;
        color: #EAEAEA;
        font-family: 'Arial', sans-serif;
        margin: 0;
        padding: 0;
    }

    /* Title */
    .stTitle {
        color: #00FFFF;  /* Neon Cyan */
        font-size: 42px;
        font-weight: bold;
        text-align: center;
        text-shadow: 0 0 15px rgba(0, 255, 255, 0.8), 0 0 30px rgba(0, 255, 255, 0.7);
        margin-top: 50px;
    }

    /* Sidebar Styling */
    .sidebar .sidebar-content {
        background-color: #1D1D1D;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.4);
    }

    /* Button Styling */
    .stButton>button {
        background-color: #6200EA;
        color: white;
        width: 100%;
        border-radius: 8px;
        padding: 14px;
        font-size: 18px;
        font-weight: bold;
        transition: 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #3700B3;
    }

    /* Input Field Styling */
    .stTextInput>div>input {
        padding: 14px;
        font-size: 16px;
        width: 80%;
        border-radius: 12px;
        border: 1px solid #3700B3;
        box-shadow: 0px 2px 5px rgba(0, 0, 0, 0.3);
        background-color: #121212;
        color: white;
    }

    .stTextInput>div>input:focus {
        border-color: #00FFFF;
    }

    /* Anomaly Plot Styling */
    .stPlotlyChart {
        background-color: #1D1D1D;
        border-radius: 12px;
        padding: 15px;
        margin-top: 30px;
    }

    /* Chat Styling */
    .chat-container {
        display: flex;
        flex-direction: column;
        width: 100%;
        height: 70vh;
        overflow-y: auto;
        background-color: #2C2C2C;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.4);
    }

    .chat-message {
        padding: 12px;
        margin: 5px 0;
        border-radius: 10px;
        font-size: 16px;
        line-height: 1.6;
    }

    .user-message {
        background-color: #BB86FC;
        color: white;
        align-self: flex-end;
    }

    .bot-message {
        background-color: #03DAC6;
        color: black;
        align-self: flex-start;
    }

    </style>
""", unsafe_allow_html=True)

# Sidebar with file uploader and settings
with st.sidebar:
    st.image("https://via.placeholder.com/150?text=Anomaly+Detection", width=150)
    st.title("Anomaly Detection")
    st.markdown("---")

    # File upload
    uploaded_file = st.file_uploader("📊 Upload Market Data", type=["csv", "xlsx", "xls"], help="Upload your market data file (CSV or Excel)")

    # Sensitivity slider
    contamination = st.slider("Anomaly Sensitivity", min_value=0.01, max_value=0.2, value=0.1, help="Adjust the sensitivity of anomaly detection")

# Function to load data
@st.cache_data
def load_data(file):
    if file.name.endswith('.csv'):
        data = pd.read_csv(file)
    elif file.name.endswith(('.xls', '.xlsx')):
        data = pd.read_excel(file)
    else:
        raise ValueError("Unsupported file type. Please upload a CSV or Excel file.")
    
    return data

# Function to train anomaly detection model
@st.cache_resource
def train_anomaly_detector(data, contamination, model_type='IsolationForest'):
    data = data.fillna(data.mean())  # Handle missing values
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data.select_dtypes(include=[np.number]))  # Standardizing numerical data

    if model_type == 'IsolationForest':
        model = IsolationForest(contamination=contamination, random_state=42)
    elif model_type == 'OneClassSVM':
        model = OneClassSVM(nu=contamination, kernel="rbf", gamma='auto')
    elif model_type == 'LOF':
        model = LocalOutlierFactor(n_neighbors=20, contamination=contamination)

    model.fit(scaled_data)
    return model, scaler

# Function to predict anomalies
def predict_anomalies(model, scaler, data):
    scaled_data = scaler.transform(data.select_dtypes(include=[np.number]))  # Standardizing before prediction
    if isinstance(model, LocalOutlierFactor):
        return model.fit_predict(scaled_data) == -1
    return model.predict(scaled_data) == -1

# Function to plot anomaly detection timeline
def plot_anomaly_timeline(data, anomalies, model_name):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data.iloc[:, 0], name='Data', line=dict(color='blue', width=1)))  # Plotting the first column
    fig.add_trace(go.Scatter(x=data.index[anomalies], y=data.iloc[:, 0][anomalies], mode='markers', name='Anomalies', marker=dict(color='red', size=8)))
    fig.update_layout(title=f'{model_name} Anomaly Detection Timeline', xaxis_title='Index', yaxis_title='Value', height=500)
    return fig

# Function to plot anomaly score vs index
def plot_decision_function(data, model, scaler, model_name):
    scaled_data = scaler.transform(data.select_dtypes(include=[np.number]))  # Standardizing before prediction
    if isinstance(model, LocalOutlierFactor):
        scores = model.negative_outlier_factor_
    else:
        scores = model.decision_function(scaled_data)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=scores, mode='lines', name=f'{model_name} Anomaly Score', line=dict(color='purple', width=2)))
    fig.update_layout(title=f"{model_name} Anomaly Score vs Index", xaxis_title='Index', yaxis_title='Anomaly Score', height=400)
    return fig

# Main Content
st.title("🚀 Market Anomaly Detection")

if uploaded_file is not None:
    try:
        with st.spinner("Processing data..."):
            # Load the dataset
            data = load_data(uploaded_file)
            
            # Train the anomaly detection model
            model, scaler = train_anomaly_detector(data, contamination)
            
            # Predict anomalies in the data
            anomalies = predict_anomalies(model, scaler, data)
            anomaly_scores = model.decision_function(scaler.transform(data.select_dtypes(include=[np.number])))
            
            # Get investment strategy suggestions based on the detected anomalies
            investment_strategies = suggest_investment_strategy(anomalies, anomaly_scores, data)
            
            # Display the strategies in the Streamlit app
            st.write("### Investment Strategy Suggestions")
            for strategy in investment_strategies:
                st.write(strategy)
            
            # Display the anomaly timeline plot
            st.plotly_chart(plot_anomaly_timeline(data, anomalies), use_container_width=True)
            
            # Display the decision function plot
            st.plotly_chart(plot_decision_function(data, model, scaler), use_container_width=True)
            
            # Show anomalies count
            st.write(f"Number of anomalies detected: {anomalies.sum()}")
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
else:
    st.markdown("""
    ## 👋 Welcome to Market Anomaly Detection!
    Upload your market data to:
    - 🎯 Detect market anomalies
    - 📊 Visualize market patterns
    - 🤖 Get AI-powered investment advice
    - 📈 Monitor market health
    """)

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center'><p>Market Anomaly Detection - Powered by Streamlit</p><p style='color: #666;'>Built with ❤️</p></div>", unsafe_allow_html=True)
from model import MarketAnomalyDetector, load_data

# Sidebar: Model selection
with st.sidebar:
    model_name = st.selectbox("Select Model", ["IsolationForest", "OneClassSVM", "LOF"])
    contamination = st.slider("Anomaly Sensitivity", min_value=0.01, max_value=0.2, value=0.1)

# Load data and initialize the detector with the selected model
if uploaded_file is not None:
    data = load_data(uploaded_file)
    anomaly_detector = MarketAnomalyDetector(contamination=contamination, model_type=model_name)
    
def load_data(uploaded_file):
    """Load market data from a file"""
    # Get the file name
    file_name = uploaded_file.name

    # Check file type based on the extension
    if file_name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    elif file_name.endswith(('.xls', '.xlsx')):
        data = pd.read_excel(uploaded_file)
    else:
        raise ValueError("Unsupported file format")

    # Automatically fill missing values with the column mean
    data = data.fillna(data.mean())

    return data


    # Fit the model
    anomaly_detector.fit(data)

    # Predict anomalies and get anomaly scores
    anomalies = anomaly_detector.predict(data)
    anomaly_scores = anomaly_detector.anomaly_scores(data)

    # Visualize anomalies and scores
    st.plotly_chart(plot_anomaly_timeline(data, anomalies), use_container_width=True)
    st.plotly_chart(plot_decision_function(data, anomaly_scores), use_container_width=True)


# Chatbot Section
st.title("🤖 AI Financial Advisor")

# Text input for user to chat with the bot
user_input = st.text_input("Ask your financial advisor:", "")

if user_input:
    # Get advice from the chatbot
    try:
        bot_response = get_advice(user_input)
        st.write(f"**Advisor's Response:** {bot_response}")
    except Exception as e:
        st.error(f"An error occurred while fetching advice: {e}")
else:
    st.write("Ask me anything related to market analysis and anomaly detection!")

