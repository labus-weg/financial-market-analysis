import streamlit as st
import pandas as pd
from model import load_data, train_model, evaluate_model, preprocess_data
from strategy import generate_strategy
from advisor_bot import get_advice
import joblib
import openai
import os

# Streamlit app setup
st.title("Market Anomaly Detection and Investment Strategy")

# File upload section
uploaded_file = st.sidebar.file_uploader("Upload Financial Data (CSV, Excel, or other formats)", type=["csv", "xlsx", "xls", "json", "txt"])

if uploaded_file:
    # Handle file reading dynamically based on type
    file_extension = uploaded_file.name.split('.')[-1].lower()
    
    if file_extension == 'csv':
        data = pd.read_csv(uploaded_file, encoding='latin1')
    elif file_extension in ['xls', 'xlsx']:
        data = pd.read_excel(uploaded_file)
    elif file_extension == 'json':
        data = pd.read_json(uploaded_file)
    elif file_extension == 'txt':
        data = pd.read_csv(uploaded_file, sep='\t', encoding='latin1')  # For tab-separated values
    else:
        st.error(f"Unsupported file format: {file_extension}")
        st.stop()

    st.write("Data Preview:", data.head())

        # Preprocess the data
        X, y, original_data = load_data(uploaded_file)

        # Preprocess data (imputation and scaling)
        X_scaled, scaler, imputer = preprocess_data(X)

        # Load trained model (or train if needed)
        try:
            model = joblib.load('xgb_model.pkl')  # Load pre-trained model if exists
            st.sidebar.text("Loaded pre-trained model")
        except FileNotFoundError:
            model = train_model(X_scaled, y)  # Train model if not found
            joblib.dump(model, 'xgb_model.pkl')
            st.sidebar.text("Trained new model")

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

    except Exception as e:
        st.error(f"Error loading the file: {e}")
else:
    st.info("Please upload a dataset to start the analysis.")
