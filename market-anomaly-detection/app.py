import streamlit as st
import pandas as pd
from model import load_data, train_model, evaluate_model, preprocess_data
from strategy import generate_strategy
from advisor_bot import get_advice
import joblib

# Streamlit app setup
st.title("Market Anomaly Detection and Investment Strategy")

# File upload section
uploaded_file = st.sidebar.file_uploader("Upload Financial Data (CSV)", type=["csv"])

if uploaded_file:
    # Load and display the data
    data = pd.read_csv(uploaded_file)
    st.write("Data Preview:", data.head())
    
    # Preprocess the data
    X, y, original_data = load_data(uploaded_file)
    
    # Preprocess data (imputation and scaling)
    X_scaled, scaler, imputer = preprocess_data(X)

    # Train the model
    st.sidebar.text("Training Model...")
    model = train_model(X_scaled, y)
    
    # Model evaluation
    st.sidebar.text("Evaluating Model...")
    report = evaluate_model(model, X_scaled, y)
    st.text("Model Evaluation:\n" + report)

    # Make prediction (for demo, predicting on the first row)
    prediction = model.predict([X.iloc[0]])[0]
    
    # Generate strategy
    strategy = generate_strategy(prediction)
    st.write(f"**Predicted Market Condition:** {'Crash' if prediction == 1 else 'Stable'}")
    st.write(f"**Suggested Strategy:** {strategy}")
    
    # AI Bot response (for investment strategy)
    input_data = original_data.iloc[0].to_dict()
    advice = get_advice(prediction, input_data)
    st.write(f"**AI Bot Advice:** {advice}")

else:
    st.info("Please upload a dataset to start the analysis.")
