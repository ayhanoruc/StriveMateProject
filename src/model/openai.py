import os 
import json 
from dotenv import load_dotenv
import openai 
import requests

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")



def seq_gpt_call(instruction: str, prompt: str, model_name: str = "gpt-3.5-turbo", max_tokens: int = 300, timeout_duration: int = 70) -> str:
    openai.api_key = openai_api_key

    url = "https://api.openai.com/v1/chat/completions"

    headers = {
        'Authorization': f'Bearer {openai_api_key}',
        'Content-Type': 'application/json',
    }

    payload = {
        "model": model_name,
        "messages": [
            {"role": "system", "content": instruction},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens
            }

    try:
        response = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout_duration)
        #response.raise_for_status()  # Raises an HTTPError if the HTTP request returned an unsuccessful status code
        if response.status_code == 429:
            print("Rate limit exceeded. Please try again later.")
            return None, None
        data = response.json()
        #print(data)
        
        text = data["choices"][0]["message"]["content"]
        metrics = data["usage"]

        return text, metrics
    except requests.exceptions.Timeout:
        print(f"The request timed out after {timeout_duration} seconds.")
        return None, None
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        return None, None
    except requests.exceptions.RequestException as err:
        print(f"Other error occurred: {err}")
        return None, None


