def generate_strategy(predictions):
    if predictions == 1:  # Anomaly predicted (market crash)
        return "Sell: The market is expected to crash. Minimize your losses."
    else:  # No anomaly (stable market)
        return "Buy: The market is stable. Consider investing in growth assets."
