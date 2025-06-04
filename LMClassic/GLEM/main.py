import argparse
import os
import time
import torch
from torch_geometric.loader import DataLoader, NeighborLoader
from model import GLEM
import sys 
import copy
import csv
sys.path.append('../..')
from common import set_seed, load_graph_dataset, GNNEncoder, compute_acc_and_f1, MODEL_PATHs
from dataset import TextDataset


def evaluate(pred_logits):
    pred = pred_logits.argmax(dim=-1)
    accuracy, macro_f1_scores = [], []
    for mask in [graph_data.train_mask, graph_data.val_mask, graph_data.test_mask]:
        acc, macro_f1, _ = compute_acc_and_f1(pred[mask].cpu().numpy(), graph_data.y[mask].cpu().numpy())
        accuracy.append(acc)
        macro_f1_scores.append(macro_f1)
        
    return accuracy, macro_f1_scores


def load_model(cur_phase):
    print(f"Move {cur_phase} model from CPU memory")
    if cur_phase == 'lm':
        model.lm = model.lm.to(device, non_blocking=True)
        model.gnn = model.gnn.to("cpu", non_blocking=True)
        optimizer = torch.optim.Adam(model.lm.parameters(), lr=args.lm_lr)
    if cur_phase == 'gnn':
        model.gnn = model.gnn.to(device, non_blocking=True)
        model.lm = model.lm.to("cpu", non_blocking=True)
        optimizer = torch.optim.Adam(model.gnn.parameters(), lr=args.gnn_lr)
    
    return optimizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Default configurations
    parser.add_argument('--device', type=str, default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--re_split', type=int, default=0)
    
    # GLEM specific configurations
    parser.add_argument('--pl_ratio', type=float, default=0.5, help='Pseudo labels ratio')
    parser.add_argument('--num_em_iters', type=int, default=1, help='Number of EM iterations')
    
    # GLEM GNN model configurations
    parser.add_argument('--gnn_model', type=str, default='GCN')
    parser.add_argument('--gnn_hidden_channels', type=int, default=256)
    parser.add_argument('--gnn_num_layers', type=int, default=3)
    parser.add_argument('--gnn_batch_size', type=int, default=1024)
    parser.add_argument('--gnn_epochs', type=int, default=200)
    parser.add_argument('--gnn_lr', type=float, default=0.002)
    parser.add_argument('--gnn_batch_norm', type=int, default=0) # 0 is a default setting 
    
    # TODO: In GLEM original paper, the LM model is set as 'prajjwal1/bert-tiny'
    parser.add_argument('--lm_model', type=str, default='SentenceBert')
    parser.add_argument('--lm_batch_size', type=int, default=32)
    parser.add_argument('--lm_epochs', type=int, default=20)
    parser.add_argument('--lm_lr', type=float, default=2e-4)
    parser.add_argument('--lm_use_lora', action='store_true', help='use Lora to fine-tune model or not')
    
    parser.add_argument('--alpha', type=float, default=0.5, help='pseudo label weight in E-step')
    parser.add_argument('--beta', type=float, default=0.5, help='pseudo label weight in M-step')
    
    parser.add_argument('--patience', type=int, default=10, help='Patience for early stopping')
    parser.add_argument('--verbose', action='store_true', help='show progress bar during training or not')
    parser.add_argument('--em_order', type=str, default='lm', help='decide train LM first or GNN first') # for arxiv/pubmed/computer, em_order = 'lm', otherwise 'gnn'
    parser.add_argument('--out_dir', type=str, default='../../results/GLEM', help='output directory')

    args = parser.parse_args()
    print(args)
    set_seed(args.seed)
    lm_shortname = copy.deepcopy(args.lm_model)
    args.lm_model = MODEL_PATHs[args.lm_model]
    device = torch.device(args.device) 

    # =========================== Prepare Graph and Text Data ===============================
    st_time = time.time()
    graph_data = load_graph_dataset(args.dataset, device=torch.device("cpu"), re_split=args.re_split, path_prefix='../..')
    # TODO: check wheter it needs original shallow embedding or lm_encoded embeddings 
    # graph_data = load_graph_dataset_for_gnn(args.dataset, device=torch.device("cpu"), re_split=args.re_split, path_prefix='..', emb_model=lm_shortname)
    test_idx = graph_data.test_mask.nonzero(as_tuple=False).squeeze()
    gold_idx = graph_data.train_mask.nonzero(as_tuple=False).squeeze()
    num_pseudo_labels = int(args.pl_ratio * gold_idx.numel())
    idx_to_select = torch.randperm(test_idx.numel())[:num_pseudo_labels]
    pseudo_labels_idx = test_idx[idx_to_select]
    train_idx = torch.cat((gold_idx, pseudo_labels_idx))
    print(f'[Data Augmentation] # Train {train_idx.size(0)}  # Pseudo Labels {num_pseudo_labels}')

    # Text Data
    text_dataset = TextDataset(tokenizer_name=args.lm_model, 
                               raw_texts=graph_data.raw_texts, 
                               labels=graph_data.y,
                               train_mask=graph_data.train_mask)
    gold_dataset = torch.utils.data.Subset(dataset=text_dataset, indices=gold_idx)
    train_dataset = torch.utils.data.Subset(dataset=text_dataset, indices=train_idx)

    text_pretrain_loader = DataLoader(gold_dataset, batch_size=args.lm_batch_size, shuffle=True, drop_last=False)
    text_train_loader = DataLoader(train_dataset, batch_size=args.lm_batch_size, shuffle=True, drop_last=False)
    text_test_loader = DataLoader(text_dataset, batch_size=args.lm_batch_size*4, drop_last=False, shuffle=False)
    print(f"[Text Data] # TrainLoader {len(text_train_loader)} # TestLoader {len(text_test_loader)}")
    
    # Graph Data
    graph_data = graph_data.to(device)
    num_classes = len(graph_data.label_name)
    for key in graph_data.keys(): # searialize all tensor data
        if isinstance(graph_data[key], list):
            try:
                graph_data[key] = torch.tensor(graph_data[key])
            except:
                delattr(graph_data, key)
    graph_data.is_gold = graph_data.train_mask
    print(f"[Graph Data] {graph_data}")

    graph_pretrain_loader = NeighborLoader(graph_data, input_nodes=gold_idx, num_neighbors=[15, 10, 5], batch_size=args.gnn_batch_size, shuffle=True)
    graph_train_loader = NeighborLoader(graph_data, input_nodes=train_idx, num_neighbors=[15, 10, 5], batch_size=args.gnn_batch_size, shuffle=True)
    subgraph_loader = NeighborLoader(graph_data, input_nodes=None, num_neighbors=[-1], batch_size=args.gnn_batch_size*4)

    # =========================== Prepare GLEM Model ===============================
    gnn_model = GNNEncoder(input_dim=graph_data.x.size(1), 
                           hidden_dim=args.gnn_hidden_channels,
                           output_dim=num_classes,
                           n_layers=args.gnn_num_layers,
                           gnn_type=args.gnn_model, 
                           batch_norm=args.gnn_batch_norm).to(device)
    model = GLEM(lm_to_use=args.lm_model, 
                 gnn_to_use=gnn_model, 
                 out_channels=num_classes,
                 lm_use_lora=args.lm_use_lora,
                 alpha=args.alpha,
                 beta=args.beta,
                 device=device)
    
    gnn_opt = torch.optim.Adam(gnn_model.parameters(), lr=args.gnn_lr)
    lm_opt = torch.optim.Adam(model.lm.parameters(), lr=args.lm_lr)
    
    # =========================== Pre-training and EM Iterations ==========================
    if args.em_order == 'lm':
        model.lm = model.lm.to(device)
        pretrain_num_epochs = args.lm_epochs
        pretrain_loader, test_loader, pretrain_opt = text_pretrain_loader, text_test_loader, lm_opt
    else:
        pretrain_num_epochs = args.gnn_epochs 
        pretrain_loader, test_loader, pretrain_opt = graph_pretrain_loader, subgraph_loader, gnn_opt
        
    early_stopping, best_val_acc = 0, 0.0 
    for epoch in range(1, pretrain_num_epochs+1):
        acc, loss = model.train(args.em_order, pretrain_loader, pretrain_opt, None, epoch, False, args.verbose) 

        if epoch >= 5 or epoch == pretrain_num_epochs:
            pretrain_preds = model.inference(args.em_order, test_loader, verbose=args.verbose)
            acc_list, f1_list = evaluate(pretrain_preds)
            print(f'[Pre-train Epoch {epoch}] Train Acc {acc_list[0]} |  Val Acc {acc_list[1]} |  Train F1 {f1_list[0]} | Val F1 {f1_list[1]}')

            if acc_list[1] <= best_val_acc:
                early_stopping += 1 
                if early_stopping >= args.patience:
                    print(f'[Pre-train] Early Stopping at Epoch {epoch}')
                    break
            else:
                best_val_acc = acc_list[1]
    
    preds = model.inference(args.em_order, test_loader, verbose=args.verbose)
    acc_list, f1_list = evaluate(preds)
    print('============================')
    print(f"Pre-training Train-Acc {acc_list[0]} Eval-Acc {acc_list[1]} Test-Acc {acc_list[2]}")
    print(f"Pre-training Train-F1 {f1_list[0]} Eval-F1 {f1_list[1]} Test-F1 {f1_list[2]}\n")

    # =========================== Formal EM Iterations ==========================
    # We run E-step(LM training) and M-Step(GNN training) alternatively in each em iterations, 
    # so the total number of iterations is num_em_iter * 2 and we switch the em_phase at end of each iteration in following loop
    
    torch.cuda.empty_cache()
    em_phase = args.em_order 
    gnn_val_acc = lm_val_acc = 0.0 
    for em_it in range(1, args.num_em_iters*2+1):
        pseudo_labels = preds.argmax(dim=-1)
        best_val_acc = 0.0 

        optimizer = load_model(em_phase) 
        if em_phase == 'gnn':
            train_loader, num_epochs, test_loader = graph_train_loader, args.gnn_epochs, subgraph_loader
        else:
            train_loader, num_epochs, test_loader = text_train_loader, args.lm_epochs, text_test_loader

        for epoch in range(1, num_epochs+1):
            acc, loss = model.train(em_phase, train_loader, optimizer, pseudo_labels, epoch, True, args.verbose)

            if epoch >= 5 or epoch == num_epochs:
                cur_preds = model.inference(em_phase, test_loader, verbose=args.verbose)

                acc_list, f1_list = evaluate(cur_preds)
                print(f'[EMIter {em_it} Phase {em_phase.upper()} Epoch {epoch}] Train Acc {acc_list[0]} Val Acc {acc_list[1]} Test Acc {acc_list[2]}')
                print(f'[EMIter {em_it} Phase {em_phase.upper()} Epoch {epoch}] Train F1 {f1_list[0]} Val F1 {f1_list[1]} Test F1 {f1_list[2]}')

                if acc_list[1] <= best_val_acc:
                    early_stopping += 1 
                    if early_stopping > args.patience:
                        print(f'[EMIter {em_it} Phase {em_phase.upper()}] Early Stopping at Epoch {epoch}, Best Acc {best_val_acc}')
                        break 
                else:
                    best_val_acc = acc_list[1]
        
        preds = model.inference(em_phase, test_loader, verbose=args.verbose)
        if em_phase == 'gnn':
            gnn_val_acc = max(gnn_val_acc, best_val_acc)
            em_phase = 'lm'
        else:
            lm_val_acc = max(lm_val_acc, best_val_acc)
            em_phase = 'gnn'
        torch.cuda.empty_cache()

    print('============================')
    print(f'Best GNN Valication Acc {gnn_val_acc};  LM Validation Acc {lm_val_acc}')
    if gnn_val_acc > lm_val_acc:
        em_phase = 'gnn'
        model.gnn = model.gnn.to(device, non_blocking=True)
        test_loader = subgraph_loader
    else:
        em_phase = 'lm'
        model.lm = model.lm.to(device, non_blocking=True)
        test_loader = text_test_loader
    
    # =========================== Final Predictions ==========================
    test_preds = model.inference(em_phase, test_loader, verbose=args.verbose)
    acc_list, f1_list = evaluate(test_preds)
    end_time = time.time()
    print(f"\n[Final] Test Acc {acc_list[2]} |  Test F1 {f1_list[2]} with Mode {em_phase.upper()}")
    print(f"Total Time {end_time - st_time:.2f} Seconds ~ {(end_time - st_time) / 60:.2f} Minutes ~ {(end_time - st_time) / 3600:.3f} Hours")
    
    # TODO: write performance into Summary file
    os.makedirs(args.out_dir, exist_ok=True)
    written_file = f"Summary{'_s' if args.re_split else ''}.csv"
    writer = csv.writer(open(f"{args.out_dir}/{written_file}", 'a'))
    writer.writerow([args.dataset, args.seed, lm_shortname, args.em_order, args.gnn_model, acc_list[2], f1_list[2], str(round((end_time - st_time) /60, 3))+" Min"])
    