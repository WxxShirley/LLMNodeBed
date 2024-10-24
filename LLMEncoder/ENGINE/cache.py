import argparse
import torch 
import sys
sys.path.append("../..")
from common import get_cur_time, TextEncoder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset", type=str, default="cora", choices=['cora', "pubmed", "citeseer", "wikics", "arxiv", "instagram", "reddit"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--encoder_name", type=str, default="MiniLM", choices=[
        "e5-large", "SentenceBert", "MiniLM", "roberta",
        "Qwen-3B", "Mistral-7B", "Vicuna-13B", "Llama3-8B", "Llama-13B"
    ])
    parser.add_argument("--save_emb", type=int, default=1)
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    graph_data = torch.load(f"../../datasets/{args.dataset}.pt")
    
    print('= ' * 20)
    print('## Starting Time:', get_cur_time(), flush=True)
    print(args, "\n")

    text_encoder = TextEncoder(encoder_name=args.encoder_name, device=device)
    
    layer_hidden_states = text_encoder.engine_forward(graph_data.raw_texts) 
    
    if args.save_emb:
        torch.save(layer_hidden_states, f"../../datasets/{args.encoder_name}/{args.dataset}_cache_emb.pt")
    
    print('\n## Finishing Time:', get_cur_time(), flush=True)
    print('= ' * 20)
    print("Done!")
