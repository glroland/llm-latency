import time
import openai
from openai import OpenAI

def get_timestamp():
    """
    Get the current timestamp in milliseconds.

    Returns:
        int: Current timestamp in milliseconds.
    """
    return int(round(time.time() * 1000))

def execute_inference(test_name, url, token, model, prompt, max_tokens):
    """
    Execute inference using the OpenAI API.

    Args:
        test_name (str): The name of the current test.
        url (str): The URL of the OpenAI API.
        token (str): The API token for authentication.
        model (str): The model to use for inference.
        prompt (str): The input prompt for the model.
        max_tokens (int): Maximum number of tokens to respond with.

    Returns:
        str: The response from the model.
    """
    # configure OpenAI API client
    client = OpenAI(base_url=url,
                    api_key=token)

    ts_start = get_timestamp()

    # invoke the inference API
    stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
#            temperature=0.7,
#            max_completion_tokens=max_tokens,  
            stream=True)

    # Collect the streamed response
    full_response = ""
    first_chunk = True
    ts_ttft = None
    for chunk in stream:
        content = chunk.choices[0].delta.content
        if content:
            if first_chunk:
                ts_ttft = get_timestamp()
                first_chunk = False
            full_response += content
 
    ts_end = get_timestamp()

    # Build response
    response = {
        "test_name": test_name,
        "prompt": prompt,
        "response": full_response,
        "ts_start": ts_start,
        "ts_end": ts_end,
        "ts_ttft": ts_ttft,
        "duration": ts_end - ts_start,
        "ttft": ts_ttft - ts_start if ts_ttft else None
    }
    return response
