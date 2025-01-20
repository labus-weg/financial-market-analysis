import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from model import MarketAnomalyDetector, load_data
import joblib
from datetime import datetime
import time

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
    .stProgress .st-bo {
        background-color: #FF4B4B;
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
    uploaded_file = st.file_uploader(
        "📊 Upload Market Data",
        type=["csv", "xlsx", "xls"],
        help="Upload your market data file (CSV or Excel)"
    )
    
    if uploaded_file:
        st.success("File uploaded successfully!")
        
    # Settings
    st.markdown("### ⚙️ Settings")
    contamination = st.slider(
        "Anomaly Sensitivity",
        min_value=0.01,
        max_value=0.2,
        value=0.1,
        help="Adjust the sensitivity of anomaly detection"
    )
    
    with st.expander("🔧 Advanced Settings"):
        window_size = st.slider("Analysis Window", 5, 50, 20)
        confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.95, 0.7)

# Main content
st.title("🎯 Market Anomaly Detection System")

# Initialize session state
if 'model' not in st.session_state:
    st.session_state.model = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
if 'data' not in st.session_state:
    st.session_state.data = None

def get_bot_advice(prediction, confidence, market_data):
    """Generate bot advice based on market conditions"""
    if prediction == 1:  # Anomaly detected
        advice = {
            'status': '🚨 Market Anomaly Detected',
            'risk_level': 'High Risk',
            'actions': [
                "Consider reducing position sizes",
                "Review stop-loss levels",
                "Increase cash holdings",
                "Monitor market volatility closely"
            ],
            'explanation': f"Unusual market behavior detected with {confidence:.1%} confidence. "
                         "This could indicate potential market instability."
        }
    else:
        advice = {
            'status': '✅ Normal Market Conditions',
            'risk_level': 'Standard Risk',
            'actions': [
                "Maintain regular position sizes",
                "Continue planned investments",
                "Monitor market conditions",
                "Review portfolio allocation"
            ],
            'explanation': f"Market conditions appear normal with {confidence:.1%} confidence. "
                         "Continue following your investment strategy."
        }
    return advice

def plot_anomaly_timeline(data, predictions, scores):
    """Create an interactive timeline plot of anomalies"""
    fig = go.Figure()
    
    # Add price line
    fig.add_trace(go.Scatter(
        x=data.index,
        y=data['Close'] if 'Close' in data.columns else data.iloc[:, 0],
        name='Price',
        line=dict(color='blue', width=1)
    ))
    
    # Add anomaly points
    anomaly_points = data[predictions == 1]
    fig.add_trace(go.Scatter(
        x=anomaly_points.index,
        y=anomaly_points['Close'] if 'Close' in anomaly_points.columns else anomaly_points.iloc[:, 0],
        mode='markers',
        name='Anomalies',
        marker=dict(color='red', size=10, symbol='x')
    ))
    
    fig.update_layout(
        title='Market Anomaly Timeline',
        xaxis_title='Date',
        yaxis_title='Price',
        height=500
    )
    
    return fig

# Main app logic
if uploaded_file is not None:
    try:
        # Load and process data
        with st.spinner("Processing data..."):
            X, y, original_data = load_data(uploaded_file)
            
            # Train or load model
            if st.session_state.model is None:
                st.session_state.model = MarketAnomalyDetector(contamination=contamination)
                st.session_state.model.fit(X)
            
            # Make predictions
            predictions = st.session_state.model.predict(X)
            anomaly_scores = st.session_state.model.anomaly_scores(X)
            
            # Store results
            st.session_state.predictions = predictions
            st.session_state.data = original_data
            
            # Calculate confidence scores
            confidence_scores = 1 / (1 + np.exp(-anomaly_scores))  # Sigmoid transformation
        
        # Create tabs for different views
        tab1, tab2, tab3 = st.tabs(["📈 Analysis", "🤖 AI Advisor", "📊 Details"])
        
        # Tab 1: Analysis
        with tab1:
            col1, col2, col3 = st.columns(3)
            with col1:
                anomaly_count = sum(predictions)
                st.metric("Anomalies Detected", anomaly_count)
            with col2:
                anomaly_ratio = (anomaly_count / len(predictions)) * 100
                st.metric("Anomaly Ratio", f"{anomaly_ratio:.1f}%")
            with col3:
                avg_confidence = np.mean(confidence_scores)
                st.metric("Average Confidence", f"{avg_confidence:.1%}")
            
            # Plot timeline
            st.plotly_chart(plot_anomaly_timeline(original_data, predictions, anomaly_scores),
                          use_container_width=True)
        
        # Tab 2: AI Advisor
        with tab2:
            # Get latest prediction and confidence
            latest_pred = predictions[-1]
            latest_conf = confidence_scores[-1]
            
            # Get bot advice
            advice = get_bot_advice(latest_pred, latest_conf, original_data.iloc[-1])
            
            # Display advice
            st.header(advice['status'])
            st.markdown(f"**Risk Level:** {advice['risk_level']}")
            st.markdown(f"**Analysis:** {advice['explanation']}")
            
            st.subheader("📋 Recommended Actions:")
            for action in advice['actions']:
                st.markdown(f"- {action}")
            
            # Add interactive chat-like interface
            with st.expander("💬 Ask Market Sentinel"):
                user_question = st.text_input("Ask about current market conditions...")
                if user_question:
                    with st.spinner("Analyzing..."):
                        time.sleep(1)  # Simulate processing
                        st.write("🤖 Based on current analysis:")
                        st.write(advice['explanation'])
                        st.write("Consider the recommended actions above.")
        
        # Tab 3: Details
        with tab3:
            st.dataframe(
                pd.DataFrame({
                    'Date': original_data.index,
                    'Anomaly': predictions,
                    'Confidence': confidence_scores,
                    'Score': anomaly_scores
                }).sort_values('Score', ascending=False)
            )
            
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.write("Please ensure your data is in the correct format and try again.")

else:
    # Welcome message and instructions
    st.markdown("""
    ## 👋 Welcome to Market Sentinel!
    
    Upload your market data to:
    - 🎯 Detect market anomalies
    - 📊 Visualize market patterns
    - 🤖 Get AI-powered advice
    - 📈 Monitor market health
    
    ### Data Format Requirements:
    - CSV or Excel file
    - Must include: Date, Close/Price columns
    - Optional: Volume, Open, High, Low
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <p>Market Sentinel - Advanced Market Anomaly Detection</p>
        <p style='color: #666;'>Built with ❤️ by NNL</p>
    </div>
    """,
    unsafe_allow_html=True
)