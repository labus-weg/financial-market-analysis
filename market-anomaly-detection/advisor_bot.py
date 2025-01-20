# advisor_bot.py

import openai

openai.api_key = "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"


def get_advice(user_input):
    """
    This function generates a bot response based on user input using OpenAI's GPT model (updated API).
    """
    # Send a prompt to OpenAI's GPT model
    response = openai.chat_completions.create(
        model="gpt-3.5-turbo",  # You can replace with the desired GPT model version
        messages=[
            {"role": "user", "content": user_input}
        ]
    )
    
    # Get the response text
    bot_response = response['choices'][0]['message']['content'].strip()
    
    return bot_response
