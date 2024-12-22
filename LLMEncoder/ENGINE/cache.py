import argparse
import torch 
import os 
import sys
sys.path.append("../..")
from common import get_cur_time, TextEncoder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset", type=str, default="cora", choices=['cora', "pubmed", "citeseer", "wikics", "arxiv", "instagram", "reddit", "computer", "history", "photo"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--encoder", type=str, default="roberta", choices=[
        "e5-large", "SentenceBert", "MiniLM", "roberta",
        "Qwen-3B", "Qwen-7B", "Llama-8B", "Mistral-7B"
    ])
    parser.add_argument("--save_emb", type=int, default=1)
    parser.add_argument("--pool", type=str, default="cls")
    
    args = parser.parse_args()
    device = torch.device(args.device)
    
    graph_data = torch.load(f"../../datasets/{args.dataset}.pt")
    
    print('= ' * 20)
    print('## Starting Time:', get_cur_time(), flush=True)
    print(args, "\n")
    
    if os.path.exists(f"../../datasets/{args.encoder}/{args.dataset}_ENGINE.pt"):
        print(f"{args.encoder}-{args.dataset}-ENGINE file already exists!")
        exit()
    
    encoder_type = "LM" if args.encoder in ["MiniLM", "SentenceBert", "e5-large", "roberta"] else "LLM"
    text_encoder = TextEncoder(encoder_name=args.encoder, encoder_type=encoder_type, device=device)
    
    layer_hidden_states = text_encoder.engine_forward(graph_data.raw_texts, pool=args.pool if encoder_type != "LLM" else "mean") 
    # print(len(layer_hidden_states))

    # TODO: you can customize the selected layers setting
    selected_layers_dict = {
        "MiniLM": [0, 3, 6], "SentenceBert": [0, 3, 6], "roberta": [0, 12, 24], "e5-large": [0, 12, 24],
        "Qwen-3B": [0, 9, 18, 27, 36], "Qwen-7B": [0, 9, 18, 27],  "Mistral-7B": [0, 8, 16, 24, 32], "Llama-8B": [0, 8, 16, 24, 32],
    }
    selected_layers = selected_layers_dict[args.encoder]
    
    if args.save_emb:
        layer_hidden_states = [hidden  for layer, hidden in enumerate(layer_hidden_states) if layer in selected_layers]
        torch.save(layer_hidden_states, f"../../datasets/{args.encoder}/{args.dataset}_ENGINE.pt")
    
    print('\n## Finishing Time:', get_cur_time(), flush=True)
    print('= ' * 20)
    print("Done!")
