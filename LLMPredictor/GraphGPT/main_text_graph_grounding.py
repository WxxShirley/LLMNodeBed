import argparse 
from torch.utils.data import DataLoader 
import torch 
import argparse
from dataset import TextGraphGroundDataset
from text_graph_clip import TextGraphCLIP
import time
import sys 
sys.path.append("../..")
from common import set_seed, load_graph_dataset_for_llaga, load_graph_dataset_for_zerog


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--lm_name", type=str, default="SentenceBert")
    parser.add_argument("--lm_freeze", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)

    # GNN Configuration 
    parser.add_argument("--gnn_hidden_dim", type=int, default=512)
    parser.add_argument("--gnn_layers", type=int, default=2)
    
    # Training Configuration 
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_epoch", type=int, default=2)
    
    parser.add_argument("--loss_lambda", type=float, default=1)
    parser.add_argument("--num_sampled_neighbors", type=int, default=3)
    
    args = parser.parse_args()
    device = torch.device(args.device)
    print(args, "\n")
    set_seed(args.seed)
    
    lm_dim_dict = {"MiniLM": 384, "SentenceBert": 768, "e5-large": 1024, "roberta": 1024}
    
    # graph_data = load_graph_dataset_for_llaga(dataset_name=args.dataset, device=device, encoder=args.lm_name)
    graph_data = load_graph_dataset_for_zerog(dataset_name=args.dataset, device=device)
    
    graph_dataset = TextGraphGroundDataset(graph_data, num_sampled_neighbors=args.num_sampled_neighbors)
    dataloader = DataLoader(graph_dataset, batch_size=args.batch_size, shuffle=True)
    
    model = TextGraphCLIP(gnn_input_dim=graph_data.x.shape[1],
                          gnn_hidden_dim=args.gnn_hidden_dim, 
                          gnn_layers=args.gnn_layers, 
                          projector_dim=lm_dim_dict[args.lm_name],
                          text_model=args.lm_name, 
                          device=device,
                          lm_freeze=args.lm_freeze,
                          num_samples=args.num_sampled_neighbors).to(device)
    optimizer = torch.optim.Adam([p for _, p in model.named_parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)
    for epoch in range(args.num_epoch):
        st_time = time.time()
        model.train()
        epoch_loss = 0.0 
        for batch_idx, batch_data in enumerate(dataloader):
            node_loss, node_neighbor_loss, text2text_loss = model(batch_data, graph_data)
            final_loss = node_loss + args.loss_lambda * node_neighbor_loss + args.loss_lambda * text2text_loss
        
            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step() 
            epoch_loss += final_loss.item()
            
            if batch_idx % 100 == 0:
                print(f"(temporary) Epoch {epoch+1} - Batch {batch_idx+1} Loss {final_loss:.4f}")
            
        epoch_loss /= len(dataloader)
        print(f"Epoch {epoch+1}/{args.num_epoch} Training Loss {epoch_loss:.4f}  Cost Time {time.time()-st_time:.3f}")
    
    # TODO: Save the well-trained graph embedding for further LLM instruction tuning
    