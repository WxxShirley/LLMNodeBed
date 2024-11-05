import torch 
from tqdm import tqdm 
import argparse
import os 
import sys
sys.path.append("../..")
from common import get_cur_time, load_graph_dataset_for_tape, GNNEncoder, compute_acc_and_f1, array_mean_std, set_seed
import sampling
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_mean_pool
import json
import time
import numpy as np


def prepare_dataloader(g_data): 
    g_data = g_data.to("cpu")
    train_idx = g_data.train_mask.nonzero().squeeze()
    val_idx = g_data.val_mask.nonzero().squeeze()
    test_idx = g_data.test_mask.nonzero().squeeze()
    kwargs = {'batch_size': args.batch_size, 'num_workers': 4, 'persistent_workers': True}
    
    if args.sampler == "random_walk" and args.dataset != "arxiv":
        train_graphs = sampling.collect_subgraphs(train_idx, g_data, walk_steps=args.walk_steps, restart_ratio=args.restart)
        val_graphs = sampling.collect_subgraphs(val_idx, g_data, walk_steps=args.walk_steps, restart_ratio=args.restart)
        test_graphs = sampling.collect_subgraphs(test_idx, g_data, walk_steps=args.walk_steps, restart_ratio=args.restart)
    else:
        save_folder = f"../../datasets/subgraphs"
        os.makedirs(save_folder, exist_ok=True)
        if os.path.exists(f"{save_folder}/{args.dataset}_train.pt"):
            train_graphs = torch.load(f"{save_folder}/{args.dataset}_train.pt")
            val_graphs = torch.load(f"{save_folder}/{args.dataset}_val.pt")
            test_graphs = torch.load(f"{save_folder}/{args.dataset}_test.pt")
        else:
            train_graphs = sampling.ego_graphs_sampler(train_idx, g_data, hop=args.k)
            val_graphs = sampling.ego_graphs_sampler(val_idx, g_data, hop=args.k)
            test_graphs = sampling.ego_graphs_sampler(test_idx, g_data, hop=args.k)
    
    train_loader = DataLoader(train_graphs, shuffle=True, **kwargs)
    val_loader = DataLoader(val_graphs, **kwargs)
    test_loader = DataLoader(test_graphs, **kwargs)
    
    g_data = g_data.to(device)
    return train_loader, val_loader, test_loader
    

def train_eval(train_loader, val_loader, test_loader, xs, model_list, prog_list, alpha_list, optimizer): 
    patience, cnt = args.patience, 0
    best_test_acc, best_test_f1, best_val_acc = 0, 0, 0 
    criterion = torch.nn.CrossEntropyLoss()
    for epoch in range(args.epoch):
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            last = None 
            total_loss = 0 
            
            for i, m in enumerate(model_list):
                m.train()
                prog_list[i].train()
                if i == 0: 
                    out = m(prog_list[i]((xs[i][data.original_idx.cpu()]).to(device)), data.edge_index)
                else:
                    a = torch.nn.functional.sigmoid(alpha_list[i] / T)
                    x = prog_list[i]((xs[i][data.original_idx.cpu()]).to(device))*a + last*(1-a)
                    out = m(x, data.edge_index)
                
                last = out 
                hid_out = torch.cat([last[data.root_n_index], global_mean_pool(last, data.batch)], dim=1)
                hid_logits = classifier(hid_out)
                total_loss += criterion(hid_logits, data.y)
            
            total_loss.backward(retain_graph=True)
            optimizer.step()
            
        val_acc, val_f1 = efficient_eval(val_loader, xs, model_list, prog_list, alpha_list)
        test_acc, test_f1 = efficient_eval(test_loader, xs, model_list, prog_list, alpha_list)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            cnt = 0
            best_test_acc, best_test_f1 = test_acc, test_f1
        else:
            cnt += 1
        if cnt >= patience:
            print(f'early stop at epoch {epoch}')
            return best_test_acc, best_test_f1
        
        if epoch % 20 == 0:
            print(f"Epoch {epoch:3d} | Val acc {val_acc:.3f} Val f1 {val_f1:.3f} Test Acc {test_acc:.3f} Test F1 {test_f1:.3f}")
    
    return best_test_acc, best_test_f1


def efficient_eval(test_loader, xs, model_list, prog_list, alpha_list):
    all_pred, all_label = None, None 
    for data in test_loader:
        data = data.to(device)
        
        last = 0
        for i, m in enumerate(model_list):
            m.eval()
            prog_list[i].eval()
           
            if i == 0:
                out = m(prog_list[i]((xs[i][data.original_idx.cpu()]).to(device)), data.edge_index)
            else:
                a = torch.nn.functional.sigmoid(alpha_list[i]/T)
                x = prog_list[i]((xs[i][data.original_idx.cpu()]).to(device))*a + last*(1-a)
                
                out = m(x, data.edge_index)
            last = out
        
        out = torch.cat([out[data.root_n_index], global_mean_pool(out, data.batch)], dim=1)
        out = classifier(out)
        pred = out.argmax(dim=1)
       
        all_pred = pred if all_pred is None else torch.cat([all_pred, pred], dim=0)
        all_label = data.y if all_label is None else torch.cat([all_label, data.y], dim=0)

    acc, f1 = compute_acc_and_f1(all_pred.cpu().numpy(), all_label.cpu().numpy())
    return acc, f1
        
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset", type=str, default="cora", choices=['cora', "pubmed", "citeseer", "wikics", "arxiv", "instagram", "reddit"])
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--encoder", type=str, default="MiniLM", choices=["e5-large", "SentenceBert", "MiniLM", "roberta", "Qwen-3B", "Mistral-7B", "Vicuna-13B", "Llama3-8B", "Llama-13B"])
    parser.add_argument("--re_split", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--run_times", type=int, default=3)
    parser.add_argument("--write_result", type=int, default=1)
    
    parser.add_argument("--r", type=int, default=32)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--gnn_type", type=str, default="GCN")
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--gnn_dropout", type=float, default=0.5)
    parser.add_argument("--T", type=float, default=0.1)
    parser.add_argument("--batch_norm", type=int, default=0)
    
    # Sampling 
    parser.add_argument('--restart', type=float, help="the restart ratio of random walking", default=0.5)
    parser.add_argument('--walk_steps', type=int, help="the steps of random walking", default=64)
    parser.add_argument('--k', type=int, help="the hop of neighboors", default=1)
    parser.add_argument("--sampler", type=str, default="random_walk", choices=["random_walk", "khop"])
    
    # Model Training 
    parser.add_argument("--lr", type=float, default=0.0005)
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--epoch", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    
    args = parser.parse_args()
    device = torch.device(args.device)
    print(args)
    
    r_dict = {
        "MiniLM": 1, "SentenceBert": 2, 
        "roberta": 4, "e5-large": 4, 
        "Qwen-3B": 8, "Mistral-7B": 16, "Llama3-8B": 16, 
        "Vicuna-13B": 16, "Llama-13B": 16
    }
    args.r = r_dict[args.encoder]
    
    cache_file_path = f"../../datasets/{args.encoder}/{args.dataset}_cache_emb.pt"
    if not os.path.exists(cache_file_path):
        raise FileNotFoundError(f"No cache file found! Please use `python cache.py --dataset DATASET --encoder_name ENCODER` to generate it.")
    hidden_states = torch.load(cache_file_path)
    hidden_states = [x for x in hidden_states]
    layer_select = list(range(len(hidden_states)))
 
    input_dim, r, hidden = hidden_states[0].shape[1], args.r, args.hidden 
    reduced_input_dim = int(input_dim / r) # k in EGINE's original paper
    
    final_acc_list, final_f1_list, cost_times = [], [], []
    
    for seed in range(args.run_times): 
        set_seed(seed) 
        graph_data, num_classes, texts = load_graph_dataset_for_tape(args.dataset, device=device, use_gpt=False, re_split=args.re_split)
        train_loader, val_loader, test_loader = prepare_dataloader(graph_data)
    
        start_time = time.time()
        prog_list = [torch.nn.Sequential(
            torch.nn.Linear(input_dim, reduced_input_dim), 
            torch.nn.LayerNorm(reduced_input_dim), 
            torch.nn.ReLU(), 
            torch.nn.Linear(reduced_input_dim, reduced_input_dim)).to(device) for l in layer_select]
        model_list = [GNNEncoder(
            # TODO: specify normalization as ENGINE
            reduced_input_dim, 
            hidden_dim=hidden, 
            output_dim=reduced_input_dim, 
            n_layers=args.n_layers,
            gnn_type=args.gnn_type, 
            dropout=args.gnn_dropout, 
            batch_norm=args.batch_norm).to(device) for l in layer_select]
        
        alpha_list = [torch.nn.Parameter(torch.tensor(0.0), requires_grad=True) for l in layer_select]

        classifier = torch.nn.Linear(reduced_input_dim * 2, num_classes).to(device)
        T, lr, weight_decay = args.T, args.lr, args.weight_decay 
    
        params, xs_list = [], []
        for i, l in enumerate(layer_select):
            params.append({'params': model_list[i].parameters(), 'lr': lr, 'weight_decay': weight_decay})
            params.append({'params': prog_list[i].parameters(), 'lr': lr, 'weight_decay': weight_decay}) 
            params.append({'params': alpha_list[i], 'lr': lr, 'weight_decay': weight_decay})
            xs_list.append(hidden_states[l])
        params.append({'params': classifier.parameters(), 'lr': lr, 'weight_decay': weight_decay})
        
        optimizer = torch.optim.AdamW(params)
    
        acc, f1 = train_eval(train_loader, val_loader, test_loader, xs_list, model_list, prog_list, alpha_list, optimizer)
        print(f"Timer {seed+1} {args.encoder}-{args.dataset} Best Acc {acc:.3f} Best F1 {f1:.3f}")
        final_acc_list.append(acc)
        final_f1_list.append(f1)
        cost_times.append(time.time() - start_time)
        
    acc_mean, acc_std = array_mean_std(final_acc_list)
    f1_mean, f1_std = array_mean_std(final_f1_list)
    print(f"\n[Final] Acc {acc_mean}±{acc_std}  F1 {f1_mean}±{f1_std}")
    
    result_dict = {
        "encoder": args.encoder, "dataset": args.dataset, 
        "test-acc": f"{acc_mean:.2f}+-{acc_std:.2f}", "test-f1": f"{f1_mean:.2f}+-{f1_std:.2f}",
        "time": f"{np.round(np.mean(np.array(cost_times)), 2)}s",
        "gnn-config": f"gnn_type {args.gnn_type}; #layer {args.n_layers}; hidden {args.hidden}; reduced_input {reduced_input_dim}; sampler {args.sampler}",
        "train-config": f"lr {args.lr}"
    }
    if args.write_result:
        os.makedirs("../../results/LLMEncoder/ENGINE", exist_ok=True)
        with open("../../results/LLMEncoder/ENGINE/result.txt", "a+") as file:
            file.write(json.dumps(result_dict) + "\n")
    print("\n\n")
        