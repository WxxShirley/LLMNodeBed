import argparse 
import time 
import torch 
import os 
import sys 
sys.path.append("../..")
from common import load_graph_dataset_for_gnn, array_mean_std, compute_acc_and_f1, set_seed
import torch.nn.functional as F 
from mlp import MLP


def mlp_train():
    mlp_model.train()
    optimizer.zero_grad()
    output = mlp_model(graph_data.x)
    loss = F.cross_entropy(output[graph_data.train_mask], graph_data.y[graph_data.train_mask])
    loss.backward()
    optimizer.step() 
    return float(loss) 


@torch.no_grad()
def mlp_test():
    mlp_model.eval()
    pred = mlp_model(graph_data.x).argmax(dim=1)
    
    accuracy, macro_f1_scores, micro_f1_scores = [], [], []
    for mask in [graph_data.train_mask, graph_data.val_mask, graph_data.test_mask]:
        acc, macro_f1, micro_f1 = compute_acc_and_f1(pred[mask].cpu().numpy(), graph_data.y[mask].cpu().numpy())
        accuracy.append(acc)
        macro_f1_scores.append(macro_f1)
        micro_f1_scores.append(micro_f1)
        
    return accuracy, macro_f1_scores, micro_f1_scores


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--encoder_name", type=str, default="", choices=["", "shallow", "e5-large", "SentenceBert", "MiniLM", "roberta", "Qwen-3B", "Mistral-7B", "Qwen-7B", "Llama-8B"])
    
    # MLP configuration
    parser.add_argument("--n_layers", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--use_softmax", type=int, default=1)
    
    # Learning configuration
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run_times", type=int, default=5)
    parser.add_argument("--print_freq", type=int, default=50)
    
    # Train setting 
    parser.add_argument("--re_split", type=int, default=0)
    parser.add_argument("--write_result", type=int, default=1)
    
    args = parser.parse_args()
    print(args)
    
    device = torch.device(args.device)
    
    final_acc_list, final_macro_f1_list, final_weight_f1_list, timer_list = [], [], [], []

    if args.write_result:
        os.makedirs("../../results/MLP", exist_ok=True)
        write_file = open(f"../../results/MLP/summary{'' if not args.re_split else '_s'}.csv", mode='a', newline='')
    
    for i in range(args.run_times):
        set_seed(i)
        graph_data = load_graph_dataset_for_gnn(dataset_name=args.dataset,
                                                device=device,
                                                emb_model=args.encoder_name if len(args.encoder_name) else "shallow",
                                                re_split=args.re_split)
        print(graph_data.x.shape, graph_data.edge_index.shape)
        
        mlp_model = MLP(input_dim=graph_data.x.shape[1],
                        hidden_dim=args.hidden_dim, 
                        output_dim=len(graph_data.label_name),
                        n_layers=args.n_layers,
                        dropout=args.dropout,
                        use_softmax=args.use_softmax).to(device)
        if i == 0:
            trainable_params = sum(p.numel() for p in mlp_model.parameters() if p.requires_grad)
            print(f"[MLP] Number of parameters {trainable_params}")
        
        optimizer = torch.optim.Adam(mlp_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        
        best_eval_acc = best_test_acc = 0.0
        best_eval_mac_f1 = best_test_mac_f1 = 0.0
        best_eval_weight_f1 = best_test_weight_f1 = 0.0
        st_time, counter = time.time(), 0
        
        for epoch in range(1, args.epochs+1):
            cur_loss = mlp_train()
            [train_acc, val_acc, test_acc], [train_mac_f1, val_mac_f1, test_mac_f1], [train_weight_f1, val_weight_f1, test_weight_f1] = mlp_test()
            
            if val_acc > best_eval_acc:
                best_eval_acc, best_test_acc = val_acc, test_acc
                best_eval_mac_f1, best_test_mac_f1 = val_mac_f1, test_mac_f1
                best_eval_weight_f1, best_test_weight_f1 = val_weight_f1, test_weight_f1
                counter = 0
            else:
                counter += 1
            
            if epoch % args.print_freq == 0:
                print(f"Epoch {epoch:03d}   Train acc {train_acc:.3f} Val acc {val_acc:.3f} Test acc {test_acc:.3f}  Train F1 {train_mac_f1:.3f} Val F1 {val_mac_f1:.3f} Test F1 {test_mac_f1:.3f}")
            
            # Early stopping
            if counter >= args.patience:
                break
        
        timer_list.append(round(time.time() - st_time, 3))
        print(f'[Times {i}] Test Acc {best_test_acc:.2f}  Test Macro-F1 {best_test_mac_f1:.2f}  Test Micro-F1 {best_test_weight_f1:.2f} Time {timer_list[-1]:.3f}s\n')
        final_acc_list.append(best_test_acc)
        final_macro_f1_list.append(best_test_mac_f1)
        final_weight_f1_list.append(best_test_weight_f1)
    
    acc_mean, acc_std = array_mean_std(final_acc_list)
    macrof1_mean, macrof1_std = array_mean_std(final_macro_f1_list)
    weightf1_mean, weightf1_std = array_mean_std(final_weight_f1_list)
    print(f"\n[Final] Acc {acc_mean}±{acc_std}  Macro-F1 {macrof1_mean}±{macrof1_std}  Weight-F1 {weightf1_mean}±{weightf1_std}")

    if args.write_result:
        import csv
        writer = csv.writer(write_file)
        writer.writerow([args.dataset, args.encoder_name, args.hidden_dim, args.n_layers,
                         f"{acc_mean:.2f}±{acc_std:.2f}", f"{macrof1_mean:.2f}±{macrof1_std:.2f}", f"{weightf1_mean:.2f}±{weightf1_std:.2f}", 
                         trainable_params, f"{sum(timer_list)/len(timer_list):.3f}s"])
        