import argparse
import time 
import torch 
import os
from gnn import GNNEncoder
import torch.nn.functional as F
from sklearn.metrics import f1_score
import numpy as np 
import random


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True


def gnn_train():
    gnn_model.train()
    optimizer.zero_grad()
    # print(graph_data.x)
    output = gnn_model(graph_data.x, graph_data.edge_index)
    loss = F.cross_entropy(output[graph_data.train_mask], graph_data.y[graph_data.train_mask])
    loss.backward()
    optimizer.step()
    return float(loss)


@torch.no_grad()
def gnn_test():
    gnn_model.eval()
    pred = gnn_model(graph_data.x, graph_data.edge_index).argmax(dim=1)

    accuracy, f1_scores = [], []
    for mask in [graph_data.train_mask, graph_data.val_mask, graph_data.test_mask]:
        acc = int((pred[mask] == graph_data.y[mask]).sum()) / int(mask.sum())
        accuracy.append(acc * 100.0)
        
        f1 = f1_score(graph_data.y[mask].cpu().numpy(), pred[mask].cpu().numpy(), average="macro")
        f1_scores.append(f1 * 100.0)
    return accuracy, f1_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="cora")
    # Encoder
    parser.add_argument("--emb_type", type=str, default="shallow", choices=["LM", "LLM", "shallow"])
    parser.add_argument("--emb_model", type=str, default="e5-large")

    # GNN configuration
    parser.add_argument("--gnn_type", type=str, default="GCN", choices=["GCN", "GAT", "SAGE", "GIN", "TransformerConv"])
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.5)

    # Learning configuration
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()
    print(args)
    # set_seed(args.seed)

    device = torch.device(args.device)

    graph_data = torch.load(f"../datasets/{args.dataset}.pt").to(device)
    if args.emb_type != "shallow":
        assert os.path.exists(f"../datasets/{args.emb_model}/{args.dataset}.pt")
        node_feat = torch.load(f"../datasets/{args.emb_model}/{args.dataset}.pt").to(device).type(torch.float)
        graph_data.x = node_feat
    
    if len(graph_data.train_mask) == 10:
        graph_data.train_mask, graph_data.val_mask, graph_data.test_mask = graph_data.train_mask[0], graph_data.val_mask[0], graph_data.test_mask[0]
    
    gnn_model = GNNEncoder(
        input_dim=graph_data.x.shape[1],
        hidden_dim=args.hidden_dim, 
        output_dim=len(graph_data.label_name),
        n_layers=args.n_layers,
        gnn_type=args.gnn_type,
        dropout=args.dropout
    ).to(device)
    print(gnn_model)

    optimizer = torch.optim.Adam(gnn_model.parameters(), lr=args.learning_rate)
    
    best_eval_acc = best_test_acc = 0.0
    best_eval_f1 = best_test_f1 = 0.0
    timer, counter = [], 0

    for epoch in range(1, args.epochs+1):
        st_time = time.time()
        cur_loss = gnn_train()
        train_acc, val_acc, test_acc = gnn_test()[0]
        train_f1, val_f1, test_f1 = gnn_test()[1]

        if val_acc > best_eval_acc:
            best_eval_acc = val_acc
            best_test_acc = test_acc
            counter = 0
        else:
            counter += 1
        if val_f1 > best_eval_f1:
            best_eval_f1 = val_f1
            best_test_f1 = test_f1
        
        print(f"Epoch {epoch:03d}    Train acc {train_acc:.3f} Val acc {val_acc:.3f} Test acc {test_acc:.3f}  Train F1 {train_f1:.3f} Val F1 {val_f1:.3f} Test F1 {test_f1:.4f}")
        timer.append(time.time() - st_time)
        if counter >= args.patience:
            break
    
    print(f'[Final] Test Acc {best_test_acc:.3f}  Test F1 {best_test_f1:.3f} Avg Time/Epoch {sum(timer)/len(timer):.4f}s')
