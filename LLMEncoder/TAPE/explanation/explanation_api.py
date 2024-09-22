import torch 
import argparse
import os
import sys
import json
sys.path.append("../../")
from common import load_graph_dataset_for_tape, EXPLANATION_PROMPTS as prompt_dict, get_cur_time
from openai import OpenAI


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--device", type=str, default="cuda:0")

    args = parser.parse_args()
    device = torch.device(args.device)
    print('= ' * 20)
    print('## Starting Time:', get_cur_time(), flush=True)
    print(args, "\n")
    
    graph_data, _, raw_texts = load_graph_dataset_for_tape(args.dataset, device)
    prompt = prompt_dict[args.dataset]
    
    if args.dataset in ["cora", "wikics", "arxiv", "citeseer"]:
        prompt += "Please limit your output categories to 3."

    client = OpenAI(api_key="Deep-Seek-API", base_url="https://api.deepseek.com")
    write_dir = f"../../results/LLMEncoder/TAPE/DeepSeek-Chat"
    os.makedirs(write_dir, exist_ok=True)

    for node_idx, node_description in enumerate(raw_texts):
        if args.dataset == "citeseer" and node_idx < 354:
            continue 
        
        query = f"{node_description}\n{prompt}"
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": query},
                ],
                stream=False
            )
            explanation = response.choices[0].message.content
            # print(explanation, "\n\n")
        except Exception as e:
            explanation = None
            print(f"[ERROR] Data index {node_idx}, Error {e}")
        
        if explanation:
            with open(f"{write_dir}/{args.dataset}.json", 'a+') as file:
                file.write(json.dumps({"idx": node_idx, "answer": explanation}) + "\n")
                file.flush()
        
    print('\n## Finishing Time:', get_cur_time(), flush=True)
    print('= ' * 20)
    print("Done!")
