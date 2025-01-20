# advisor_bot.py

import openai

openai.api_key = "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"


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
