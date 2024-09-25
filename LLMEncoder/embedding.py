"""Use LLM/LM models to generate embeddings"""
import torch 
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from tqdm import tqdm
import os
import argparse
import sys
sys.path.append("../")
from common import get_cur_time


def mean_pooling(model_output, attention_mask):
    # from Sentence-Transformers official code "https://huggingface.co/sentence-transformers/all-mpnet-base-v2"
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def mean_pooling_llm(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def lm_forwad(data):
    text_embeddings = []
    with torch.no_grad():
        for text in tqdm(data.raw_texts, desc="Generating LM Embedding"):
        # for text in data.raw_texts:
            # We can also implement a parallel mode for generating embeddings. However, I've noticed that the speed tends to be somewhat slower:
            # for st_index in trange(0, len(data.raw_texts), batch_size, desc="Generating LM Embedding"):
            # text_batch = data.raw_texts[st_index: st_index + batch_size]
            encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512).to(device)
            output = lm_model(**encoded_input)
            if args.use_cls:
                text_embeddings.append(output.last_hidden_state[:, 0, :].cpu())
            else:
                tmp_embed = mean_pooling(output, encoded_input["attention_mask"])
                text_embeddings.append(tmp_embed)
            del output, encoded_input
            torch.cuda.empty_cache()
    
    return torch.cat(text_embeddings, dim=0)
   

def llm_forward(data):
    text_embeddings = []
    with torch.no_grad():
        for text in tqdm(data.raw_texts, desc="Generating LLM Embedding"):
        # for text in data.raw_texts:
            # fix len(text) == 0 for the 153-th entry in Instagram dataset
            if len(text) == 0:
                text = "Empty text"
            encoded_input = llm_tokenizer(text, add_special_tokens=False, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            outputs = llm_model(**encoded_input, output_hidden_states=True)
            if args.use_cls:
                text_embed = outputs.hidden_states[-1][:, 0, :]
            else:
                text_embed = mean_pooling_llm(outputs.hidden_states[-1], encoded_input["attention_mask"])
            text_embeddings.append(text_embed)

            del outputs, encoded_input
            torch.cuda.empty_cache()
    
    return torch.cat(text_embeddings, dim=0)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="cora", choices=['cora', "pubmed", "citeseer", "wikics", "arxiv", "instagram", "reddit"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--encoder_name", type=str, default="e5-large", choices=[
        "e5-large", "SentenceBert", "MiniLM", "roberta",
        "Qwen-3B", "Mistral-7B", "Vicuna-13B", "Llama3-8B", "Llama-13B"
    ])
    parser.add_argument("--use_cls", type=int, default=1)
    parser.add_argument("--save_emb", type=int, default=1)
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    graph_data = torch.load(f"../datasets/{args.dataset}.pt")

    lm_name_dict = {
        "MiniLM": "sentence-transformers/all-MiniLM-L6-v2", # 22M
        "SentenceBert": "sentence-transformers/multi-qa-distilbert-cos-v1", # 66M
        "roberta": "sentence-transformers/all-roberta-large-v1", # 355M
        "e5-large": "intfloat/e5-large-v2", # 355M
    }
    llm_path_dict = {
        "Qwen-3B": "/root/autodl-tmp/models/qwen/Qwen2___5-3B-Instruct", # 3B
        "Mistral-7B": "/root/autodl-tmp/models/Mistral-7B/snapshots/Mistral-7B-Instruct-v0.2", # 7B
        "Llama3-8B": "/root/autodl-tmp/models/LLM-Research/Meta-Llama-3-8B-Instruct", # 8B
        "Vicuna-13B": "/root/autodl-tmp/models/Vicuna-13B/snapshots/Vicuna-13B-v1.5", # 13B
        "Llama-13B": "/root/autodl-tmp/models/Llama2/Llama-2-13b-chat-hf" # 13B
    }

    print('= ' * 20)
    print('## Starting Time:', get_cur_time(), flush=True)
    print(args, "\n")
    
    if args.encoder_name in lm_name_dict.keys():
        lm_fullname = lm_name_dict[args.encoder_name]
        tokenizer = AutoTokenizer.from_pretrained(lm_fullname)
        lm_model = AutoModel.from_pretrained(lm_fullname).to(device)
        trainable_params = sum(p.numel() for p in lm_model.parameters() if p.requires_grad)
        print(f"[{args.encoder_name}] Number of parameters {trainable_params}")
        generated_node_emb = lm_forwad(graph_data)
    elif args.encoder_name in llm_path_dict.keys():
        llm_path = llm_path_dict[args.encoder_name]

        llm_tokenizer = AutoTokenizer.from_pretrained(llm_path)
        llm_tokenizer.pad_token_id = 0
        llm_tokenizer.padding_side = "right"
        llm_tokenizer.truncation_side = "right"

        llm_model = AutoModelForCausalLM.from_pretrained(llm_path, torch_dtype=torch.float16).to(device)
        generated_node_emb = llm_forward(graph_data)
    else:
        raise NotImplementedError(f"Encoder Name {args.encoder_name} is Not Supported!\nYou can either change the encoder or implement it in LM (or LLM) dictionaries!")
    
    print(f"[{args.dataset}-{args.encoder_name}] Node Embedding Shape {generated_node_emb.shape}")
    if args.save_emb:
        write_dir = f"../datasets/{args.encoder_name}"
        os.makedirs(write_dir, exist_ok=True)
        # suffix = "" if (args.encoder_name in llm_path_dict.keys() or args.use_cls == 1) else "_mean"
        torch.save(generated_node_emb, f"{write_dir}/{args.dataset}.pt")
    
    print('\n## Finishing Time:', get_cur_time(), flush=True)
    print('= ' * 20)
    print("Done!")
