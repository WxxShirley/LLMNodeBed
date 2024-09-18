import torch 
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
import argparse
import os
import sys
sys.path.append("../../")
from common import load_graph_dataset_for_tape, EXPLANATION_PROMPTS as prompt_dict, get_cur_time


llm_path_dict = {
    "Mistral-7B": "/root/autodl-tmp/models/Mistral-7B/snapshots/Mistral-7B-Instruct-v0.2", # 7B
    "Llama3-8B": "/root/autodl-tmp/models/LLM-Research/Meta-Llama-3-8B-Instruct", # 8B
    "Vicuna-13B": "/root/autodl-tmp/models/Vicuna-13B/snapshots/Vicuna-13B-v1.5", # 13B
    "Llama-13B": "/root/autodl-tmp/models/Llama2/Llama-2-13b-chat-hf" # 13B
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--llm_name", type=str, default="Llama3-8B")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--batch_size", type=str, default=16)

    args = parser.parse_args()
    device = torch.device(args.device)
    print('= ' * 20)
    print('## Starting Time:', get_cur_time(), flush=True)
    print(args, "\n")
    
    graph_data, _, raw_texts = load_graph_dataset_for_tape(args.dataset, device)
    
    assert args.llm_name in llm_path_dict.keys()
    llm_path = llm_path_dict[args.llm_name]

    llm_tokenizer = AutoTokenizer.from_pretrained(llm_path)
    llm_tokenizer.pad_token = llm_tokenizer.bos_token
    llm_model = AutoModelForCausalLM.from_pretrained(llm_path, torch_dtype=torch.float16).to(device)
    
    prompt = prompt_dict[args.dataset]

    data_loader = DataLoader(list(enumerate(raw_texts)), batch_size=args.batch_size, sampler=SequentialSampler(list(enumerate(raw_texts))))
    
    write_dir = f"../../results/LLMEncoder/TAPE/{args.llm_name}"
    max_length = 1024 if args.dataset not in ["arxiv"] else 2048
    if not os.path.exists(write_dir):
        os.makedirs(write_dir, exist_ok=True)

    for batch in tqdm(data_loader):
        text_batch, index_batch = batch[1], batch[0]
        batch_prompts = [f"{text}\n{prompt}" for text in text_batch]
        inputs = llm_tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True).to(device)
        copy_inputs = llm_tokenizer(batch_prompts)
        cur_max_length = max([len(one_sample) for one_sample in copy_inputs["input_ids"]])
        outputs = llm_model.generate(**inputs, max_length=max(max_length, cur_max_length+20))
        answers = [llm_tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
        
        for idx, answer in zip(index_batch, answers):
            with open(f"{write_dir}/{args.dataset}.json", 'a+') as file:
                prompt_copy = prompt.replace("\n\nAnswer: ", "")
                answer = answer.replace(prompt_copy, "")
                answer = answer.replace(raw_texts[idx.item()], "")
                
                file.write(json.dumps({"idx": idx.item(), "answer": answer}) + "\n")
                file.flush()
        
        del inputs, outputs, copy_inputs
    
    print('\n## Finishing Time:', get_cur_time(), flush=True)
    print('= ' * 20)
    print("Done!")
