import torch 
import argparse
import os
import sys
import json 
import time
from openai import AzureOpenAI
sys.path.append("../../")
from common import load_graph_dataset_for_tape, EXPLANATION_PROMPTS as prompt_dict, get_cur_time
import random 


GPT4_RESOURCE = {
    "endpoint": "https://gcrgpt4aoai5c.openai.azure.com/", 
    "keys": ["52f7817a4f964a4db1dc57b7be25427e", "b5ebbaefbfa94fa795d0e478413ae947"],
    "model": "gpt-4",
    "version": "2024-02-15-preview"  
}

GPT4o_MINI_RESOURCE = {
    "endpoint": "https://gcrgpt4aoai5.openai.azure.com/",
    'keys': ["eb5c8ad7dda2424aa0996e4cf6ce637e", "6199c81a6a894d45aa4b7a2d4840e732"],
    "model": "gpt-4o-mini", # or "gpt-4o"
    "version": "2024-06-01"
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--model_name", type=str, default="gpt-4o-mini", choices=["gpt-4", "gpt-4o-mini", "gpt-4o"])

    args = parser.parse_args()
    device = torch.device(args.device)
    print('= ' * 20)
    print('## Starting Time:', get_cur_time(), flush=True)
    print(args, "\n")
    
    graph_data, _, raw_texts = load_graph_dataset_for_tape(args.dataset, device)
    prompt = prompt_dict[args.dataset]
    prompt += "Please limit your output categories and response in a concise manner.\nEven if there is limited text, try your best to predict the categories."

    current_llm = {"gpt-4": GPT4_RESOURCE, "gpt-4o-mini": GPT4o_MINI_RESOURCE}[args.model_name]
    client = AzureOpenAI(  
        azure_endpoint = current_llm["endpoint"], 
        api_key = random.choice(current_llm["keys"]),
        api_version = current_llm["version"]
    )
    
    write_dir = f"../../results/TAPE/{args.model_name}"
    os.makedirs(write_dir, exist_ok=True) 
    writefile = f"{write_dir}/{args.dataset}.json"
    has_inferenced = []
    if os.path.exists(writefile):
        for line in open(writefile, 'r'):
            content = json.loads(line)
            has_inferenced.append(content["idx"])
    
    for node_idx, node_description in enumerate(raw_texts):
        if node_idx in has_inferenced:
            continue
        
        query = f"{node_description}\n{prompt}"
        try:
            response = client.chat.completions.create(
                model = current_llm["model"],
                messages = [
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": query},
                ]
            )
            explanation = response.choices[0].message.content
            # print(explanation, "\n\n")
        except Exception as e:
            explanation = None
            print(f"[ERROR] Data index {node_idx}, Error {e}")
            time.sleep(random.randint(20, 50))
        
        if explanation:
            with open(writefile, 'a+') as file:
                file.write(json.dumps({"idx": node_idx, "answer": explanation}) + "\n")
                file.flush()
        
    print('\n## Finishing Time:', get_cur_time(), flush=True)
    print('= ' * 20)
    print("Done!")
