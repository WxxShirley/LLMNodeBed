from openai import OpenAI
from openai import AzureOpenAI
import json
import requests
from api_keys import MY_KEYS


def invoke_llm_api(model_name, content):
    if model_name == "deepseek-chat":
        prediction = invoke_deepseek(content)
    elif model_name == "gpt-4o":
        prediction = invoke_gpt(content)
    elif model_name in ["Mistral-7B-Instruct-v0.2", "Meta-Llama-3.1-8B-Instruct"]:
        prediction = invoke_opensource_llm(model_name, content)

    return prediction


def invoke_deepseek(content):
    client = OpenAI(api_key=MY_KEYS["deepseek_key"], base_url="https://api.deepseek.com")
    
    response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {
                    'role': 'user',
                    'content': content
                }
            ],
            stream=False
        )
    prediction = response.choices[0].message.content
    return prediction


def invoke_gpt(content):
    # TODO: adjust the OpenAI API information based on your actual invocation requirements
    client = AzureOpenAI(
        azure_endpoint="AZURE_ENDPOINT",
        api_key=MY_KEYS["openai_key"],
        api_version="2024-06-01",
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
             {"role": "system", "content": "You are a helpful assistant"},
             {"role": "user", "content": content},
        ]
    )
    
    prediction = response.choices[0].message.content
    return prediction


def invoke_opensource_llm(model_name, content):
    headers = { "Content-Type": 'application/json'}
    
    # TODO: adjust the url based on your actual deployment port
    url = f"http://127.0.0.1:8008/v1/chat/completions"

    payload = json.dumps({
        "model": model_name,
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "temperature": 0.2,
        "top_p": 0.1,
        "frequency_penalty": 0,
        "presence_penalty": 1.05,
        "max_tokens": 4096,
        "stream": False,
        "stop": None
    })
    
    response = requests.post(url, headers=headers, data=payload, timeout=300)
    
    resp = response.json()
    prediction = resp["choices"][0]["message"]["content"]
    return  prediction
