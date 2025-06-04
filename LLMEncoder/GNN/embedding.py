"""Use LLM/LM models to generate embeddings"""
import torch 
from tqdm import tqdm
import os
import argparse
import sys
sys.path.append("../..")
from common import get_cur_time, TextEncoder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="cora", choices=['cora', "pubmed", "citeseer", "wikics", "arxiv", "instagram", "reddit", "computer", "photo", "history", "cornell", "wisconsin", "texas", "washington"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--encoder_name", type=str, default="SentenceBert", choices=["MiniLM", "SentenceBert", "e5-large", "roberta", "Qwen-3B", "Qwen-7B", "Mistral-7B", "Llama-8B"])
    parser.add_argument("--use_cls", type=int, default=1)
    parser.add_argument("--save_emb", type=int, default=1)
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    graph_data = torch.load(f"../../datasets/{args.dataset}.pt")

    print('= ' * 20)
    print('## Starting Time:', get_cur_time(), flush=True)
    print(args, "\n")

    if os.path.exists(f"../../datasets/{args.encoder_name}/{args.dataset}.pt"):
        print(f"[{args.dataset}-{args.encoder_name}] Embedding file already exists, Quit!")
        print('= ' * 20)
        exit()
    
    encoder_type = "LM" if args.encoder_name in ["MiniLM", "SentenceBert", "e5-large", "roberta"] else "LLM"
    text_encoder = TextEncoder(encoder_name=args.encoder_name, encoder_type=encoder_type, device=device)
    
    with torch.no_grad():
        text_embeddings = []
        for text in tqdm(graph_data.raw_texts, desc=f"Generating {encoder_type} Embedding"):
            text_emb = text_encoder(input_text=text, pooling="cls" if args.use_cls else "mean")
            text_embeddings.append(text_emb)
            torch.cuda.empty_cache()
    
    generated_node_emb = torch.cat(text_embeddings, dim=0)
    print(f"[{args.dataset}-{args.encoder_name}] Node Embedding Shape {generated_node_emb.shape}")
    if args.save_emb:
        write_dir = f"../../datasets/{args.encoder_name}"
        os.makedirs(write_dir, exist_ok=True)
        torch.save(generated_node_emb, f"{write_dir}/{args.dataset}.pt")
    
    print('\n## Finishing Time:', get_cur_time(), flush=True)
    print('= ' * 20)
    print("Done!")
