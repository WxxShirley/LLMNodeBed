import argparse
import torch 
import os 
import sys
sys.path.append("../..")
from common import get_cur_time, TextEncoder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset", type=str, default="cora", choices=['cora', "pubmed", "citeseer", "wikics", "arxiv", "instagram", "reddit"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--encoder", type=str, default="MiniLM", choices=[
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
    
    if os.path.exists(f"../../datasets/{args.encoder}/{args.dataset}_cache_emb.pt"):
        exit()

    text_encoder = TextEncoder(encoder_name=args.encoder, device=device)
    
    layer_hidden_states = text_encoder.engine_forward(graph_data.raw_texts) 
    # print(len(layer_hidden_states))
    selected_layers_dict = {
        "MiniLM": [0, 3, 6], "SentenceBert": [0, 3, 6],
        "roberta": [0, 12, 24], "e5-large": [0, 12, 24],
        "Qwen-3B": [0, 9, 18, 27, 36], "Mistral-7B": [0, 8, 16, 24, 32], "Llama3-8B": [0, 8, 16, 24, 32],
        "Vicuna-13B": [0, 8, 16, 24, 32, 40], "Llama-13B": [0, 8, 16, 24, 32, 40]
    }
    selected_layers = selected_layers_dict[args.encoder]
    
    if args.save_emb:
        layer_hidden_states = [hidden  for layer, hidden in enumerate(layer_hidden_states) if layer in selected_layers]
        torch.save(layer_hidden_states, f"../../datasets/{args.encoder}/{args.dataset}_cache_emb.pt")
    
    print('\n## Finishing Time:', get_cur_time(), flush=True)
    print('= ' * 20)
    print("Done!")
