import torch 
import argparse
import os
import sys
import json
sys.path.append("../../")
from common import load_graph_dataset_for_tape, EXPLANATION_PROMPTS as prompt_dict, get_cur_time
from openai import OpenAI, AzureOpenAI
import time 


# TODO: fill in other LLMs
LLM_RESOURCES = {
    "gpt-4": {
        "endpoint": "YOUR_GPT4_ENDPOINT",
        "key": "YOUR_GPT4-KEY", 
        "version": "2024-02-15-preview"  
    },
    "gpt-4o-mini": {
        "endpoint": "YOUR_GPT4o-mini_ENDPOINT",
        "key": "YOUR_GPT4O-KEY", 
        "version": "2024-06-01"
    },
    "deepseek-chat": {
        "endpoint": "https://api.deepseek.com",
        "key": "YOUR DEEPSEEK KEY"
    }
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--llm_name", type=str, default="gpt-4o-mini", choices=["gpt-4", "gpt-4o-mini", "deepseek-chat"])

    args = parser.parse_args()
    device = torch.device(args.device)
    print('= ' * 20)
    print('## Starting Time:', get_cur_time(), flush=True)
    print(args, "\n")
    
    graph_data, _, raw_texts = load_graph_dataset_for_tape(args.dataset, device)
    prompt = prompt_dict[args.dataset]
    prompt += "Please limit your output categories and response in a concise manner.\nEven if there is limited text, try your best to predict the categories."
    
    llm_resource = LLM_RESOURCES[args.llm_name]
    if args.llm_name in ["gpt-4", "gpt-4o-mini"]:
        client = AzureOpenAI(
            azure_endpoint=llm_resource["endpoint"],
            api_key=llm_resource["key"],
            api_version=llm_resource["version"]
        )
    else:
        client = OpenAI(api_key=llm_resource["key"], base_url=llm_resource["endpoint"])

    write_dir = f"../../results/LLMEncoder/TAPE/{args.llm_name}"
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
                model=args.llm_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": query},
                ]
            )
            explanation = response.choices[0].message.content
            # print(explanation, "\n\n")
        except Exception as e:
            explanation = None
            print(f"[ERROR] Data index {node_idx}, Error {e}")
            # time.sleep(10)
        
        if explanation:
            with open(writefile, 'a+') as file:
                file.write(json.dumps({"idx": node_idx, "answer": explanation}) + "\n")
                file.flush()
        
    print('\n## Finishing Time:', get_cur_time(), flush=True)
    print('= ' * 20)
    print("Done!")
