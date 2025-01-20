import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def get_advice(predictions, input_data):
    if predictions == 1:
        prompt = f"Given the following market data {input_data}, propose a strategy to minimize losses during a predicted market crash."
    else:
        prompt = f"Given the following market data {input_data}, propose an investment strategy to maximize returns in a stable market."

    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )

    return response.choices[0].text.strip()
