import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import argparse
import os
import sys
from tqdm import tqdm
sys.path.append("../../")
from common import load_graph_dataset_for_tape, EXPLANATION_PROMPTS as prompt_dict, get_cur_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--llm_name", type=str, default="Qwen2.5-3B")
    parser.add_argument("--device", type=str, default="cuda:0")
    
    args = parser.parse_args()
    device = torch.device(args.device)
    print('= ' * 20)
    print('## Starting Time:', get_cur_time(), flush=True)
    print(args, "\n")
    
    graph_data, _, raw_texts = load_graph_dataset_for_tape(args.dataset, device)
    
    param_str = args.llm_name.split("-")[1].replace(".", "___")
    llm_path = f"/root/autodl-tmp/models/qwen/Qwen2___5-{param_str}-Instruct"
    
    llm_tokenizer = AutoTokenizer.from_pretrained(llm_path)
    llm_model = AutoModelForCausalLM.from_pretrained(llm_path, torch_dtype=torch.float16, device_map="cuda:0")
    
    prompt = prompt_dict[args.dataset]
    prompt += "Please limit your output categories and response in a concise manner."

    write_dir = f"../../results/LLMEncoder/TAPE/{args.llm_name}"
    os.makedirs(write_dir, exist_ok=True)

    max_length = 1024 if args.dataset not in ["arxiv"] else 2048
    if not os.path.exists(write_dir):
        os.makedirs(write_dir, exist_ok=True)
    
    writefile = f"{write_dir}/{args.dataset}.json"
    has_inferenced = []
    if os.path.exists(writefile):
        for line in open(writefile, 'r'):
            content = json.loads(line)
            has_inferenced.append(content["idx"])
    
    print(f"{writefile} has {len(has_inferenced)} been inferenced before ... ")
    
    for node_idx in tqdm(range(len(raw_texts)), desc="Node Explanation"):
        if node_idx in has_inferenced:
            continue

        messages = [
            {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
            {"role": "user", "content": f"{raw_texts[node_idx]}\n{prompt}"}
        ]
        text = llm_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Deal with over-length
        text = text[:min(len(text), 2048)]
        model_inputs = llm_tokenizer([text], return_tensors="pt").to(llm_model.device)
        generated_ids = llm_model.generate(**model_inputs, max_new_tokens=max_length)
        generated_ids = [
              output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = llm_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        with open(f"{write_dir}/{args.dataset}.json", 'a+') as file:
            file.write(json.dumps({"idx": node_idx, "answer": response}) + "\n")
            file.flush()
        
        del model_inputs, generated_ids
        torch.cuda.empty_cache()
 
    print('\n## Finishing Time:', get_cur_time(), flush=True)
    print('= ' * 20)
    print("Done!")
