import argparse
import time 
import torch 
import os
import sys
sys.path.append("../")
from common import load_graph_dataset, GNNEncoder, array_mean_std, compute_acc_and_f1
import torch.nn.functional as F


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
        acc, f1 = compute_acc_and_f1(pred[mask].cpu().numpy(), graph_data.y[mask].cpu().numpy())
        accuracy.append(acc)
        f1_scores.append(f1)
        
    return accuracy, f1_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="cora")
    # Encoder
    parser.add_argument("--encoder_name", type=str, default="", choices=["", "shallow", "e5-large", "SentenceBert", "MiniLM", "roberta", "Qwen-3B", "Mistral-7B", "Vicuna-13B", "Llama3-8B", "Llama-13B"])
    
    # GNN configuration
    parser.add_argument("--gnn_type", type=str, default="GCN", choices=["GCN", "GAT", "SAGE", "TransformerConv"])
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--use_softmax", type=int, default=0)
    parser.add_argument("--residual_conn", type=int, default=0)
    parser.add_argument("--jump_knowledge", type=int, default=0)
    parser.add_argument("--batch_norm", type=int, default=0)

    # Learning configuration
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--run_times", type=int, default=5)
    parser.add_argument("--print_freq", type=int, default=50)

    parser.add_argument("--write_result", type=int, default=0)

    args = parser.parse_args()
    print(args)

    device = torch.device(args.device)
    graph_data = load_graph_dataset(args.dataset, device, args.encoder_name if len(args.encoder_name) else "shallow")
    print(graph_data.x.shape)

    final_acc_list, final_f1_list, timer_list = [], [], []

    if args.write_result:
        os.makedirs("../results/LLMEncoder", exist_ok=True)
        write_file = open(f"../results/LLMEncoder/{args.dataset}.csv", mode='a', newline='')
        
    for i in range(1, args.run_times+1):
        gnn_model = GNNEncoder(
            input_dim=graph_data.x.shape[1],
            hidden_dim=args.hidden_dim, 
            output_dim=len(graph_data.label_name),
            n_layers=args.n_layers,
            gnn_type=args.gnn_type,
            dropout=args.dropout,
            use_softmax=args.use_softmax,
            batch_norm=args.batch_norm,
            residual_conn=args.residual_conn,
            jump_knowledge=args.jump_knowledge
        ).to(device)
        if i == 1:
            trainable_params = sum(p.numel() for p in gnn_model.parameters() if p.requires_grad)
            print(f"[GNN] Number of parameters {trainable_params}")

        optimizer = torch.optim.Adam(gnn_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
        best_eval_acc = best_test_acc = 0.0
        best_eval_f1 = best_test_f1 = 0.0
        st_time, counter = time.time(), 0

        for epoch in range(1, args.epochs+1):
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
            
            if epoch % args.print_freq == 0:
                print(f"Epoch {epoch:03d}   Train acc {train_acc:.3f} Val acc {val_acc:.3f} Test acc {test_acc:.3f}  Train F1 {train_f1:.3f} Val F1 {val_f1:.3f} Test F1 {test_f1:.4f}")
            
            # Early stopping
            if counter >= args.patience:
                break
        
        timer_list.append(round(time.time() - st_time, 3))
        print(f'[Times {i}] Test Acc {best_test_acc:.2f}  Test F1 {best_test_f1:.2f} Time {timer_list[-1]:.3f}s\n')
        final_acc_list.append(best_test_acc)
        final_f1_list.append(best_test_f1)
    
    acc_mean, acc_std = array_mean_std(final_acc_list)
    f1_mean, f1_std = array_mean_std(final_f1_list)
    print(f"\n[Final] Acc {acc_mean}±{acc_std}  F1 {f1_mean}±{f1_std}")

    if args.write_result:
        import csv
        writer = csv.writer(write_file)
        writer.writerow([args.gnn_type, args.n_layers, args.hidden_dim, args.dropout, args.encoder_name, args.batch_norm, args.residual_conn, f"{acc_mean:.2f}±{acc_std:.2f}", f"{f1_mean:.2f}±{f1_std:.2f}", trainable_params, f"{sum(timer_list)/len(timer_list):.3f}s"])
