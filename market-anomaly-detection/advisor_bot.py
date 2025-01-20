# advisor_bot.py
import openai
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file

openai.api_key = os.getenv("OPENAI_API_KEY")


def get_advice(user_input):
    """
    This function generates a bot response based on user input using OpenAI's GPT model.
    """
    try:
        # Send the prompt to OpenAI's GPT model using the latest API (ChatCompletion)
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # You can replace with gpt-4 if required
            messages=[
                {"role": "user", "content": user_input}
            ]
        )
        
        # Extract the response message content
        bot_response = response['choices'][0]['message']['content'].strip()
        return bot_response

    except openai.error.OpenAIError as e:
        # Handle any OpenAI API specific errors
        return f"An error occurred with the OpenAI API: {str(e)}"
    except Exception as e:
        # Handle any other unexpected errors
        return f"An unexpected error occurred: {str(e)}"


def suggest_investment_strategy(anomalies, anomaly_scores, data):
    """
    Suggest investment strategies based on detected anomalies in the dataset.
    
    :param anomalies: A boolean array indicating where anomalies occur.
    :param anomaly_scores: The anomaly scores for each data point.
    :param data: The original data (price or market data).
    :return: Suggested investment strategies.
    """
    
    strategies = []
    
    for i in range(1, len(data)):
        if anomalies[i]:
            # Check previous trend and anomaly score
            prev_data = data[i - 1]
            current_data = data[i]
            anomaly_score = anomaly_scores[i]
            
            if current_data > prev_data and anomaly_score < -0.5:  # price increase with a negative anomaly score
                strategies.append(f"SELL at index {i}: Market is potentially overvalued.")
            elif current_data < prev_data and anomaly_score > 0.5:  # price decrease with a positive anomaly score
                strategies.append(f"BUY at index {i}: Market is potentially undervalued.")
            elif current_data < prev_data and anomaly_score < -0.5:  # sharp decline, strong anomaly
                strategies.append(f"SHORT SELL at index {i}: Market shows bearish signals.")
            else:
                strategies.append(f"HOLD at index {i}: Market is stable, no action needed.")
    
    if not strategies:
        strategies.append("No significant anomalies detected.")
    
    return strategies
