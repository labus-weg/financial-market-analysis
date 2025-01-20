# advisor_bot.py

def get_advice(user_input):
    """
    Given a user's query, this function will return an appropriate response.
    It uses a simple keyword-based approach, but you can improve it with NLP models if needed.
    """
    user_input = user_input.lower()
    
    if "anomaly" in user_input:
        return "Anomalies are outliers in the market data, which we detected using our model. Red markers in the chart represent anomalies."
    elif "investment strategy" in user_input:
        return "Investment strategies can vary depending on detected anomalies. Hedging and position adjustments are typical approaches."
    elif "model" in user_input:
        return "We are using the Isolation Forest model for anomaly detection, which is effective in identifying outliers in high-dimensional data."
    elif "data" in user_input:
        return "You can upload market data in CSV or Excel format to detect anomalies and explore investment strategies."
    else:
        return "I'm here to help with anomalies and market strategies. Feel free to ask about any related topics!"

