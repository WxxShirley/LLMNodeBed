import os
import time
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.loader import NeighborLoader
from tqdm import tqdm
from torch_geometric.loader import RandomNodeLoader
import sys 
sys.path.append("../..")
from common import set_seed, load_graph_dataset_for_gnn, compute_acc_and_f1
from common import GNNEncoder


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="ogbn-products")
    parser.add_argument("--encoder_name", type=str, default="")
    parser.add_argument("--gnn_type", type=str, default="GCN")
    
    # GNN hyperparameters
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--use_softmax", type=int, default=0)
    parser.add_argument("--batch_norm", type=int, default=1)
    parser.add_argument("--residual_conn", type=int, default=0)
    parser.add_argument("--jump_knowledge", type=int, default=0)
    
    # Training hyperparameters
    parser.add_argument("--run_times", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-2)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--print_freq", type=int, default=1)
    
    # Multi-GPU and batch settings
    parser.add_argument("--gpus", type=str, default="2")
    parser.add_argument("--batch_size", type=int, default=8192)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # Train setting 
    parser.add_argument("--re_split", type=int, default=0)
    parser.add_argument("--write_result", type=int, default=1)
    
    return parser.parse_args()


def train(model, loader, optimizer, devices):
    """Train the model using batches distributed across multiple GPUs"""
    model.train()
    total_loss = 0
    total_nodes = 0
    
    # Use primary device for model
    primary_device = devices[0]
    model = model.to(primary_device)
    
    # Distribute batches across GPUs using round-robin
    for i, batch in enumerate(loader):
        # Move batch to primary device (instead of round-robin)
        batch = batch.to(primary_device)
        optimizer.zero_grad()
        
        # Forward pass
        out = model(batch.x, batch.edge_index)
        # Make sure y is on the same device as out
        loss = F.cross_entropy(out[batch.train_mask], batch.y[batch.train_mask])
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track loss
        batch_size = batch.train_mask.sum().item()
        total_loss += loss.item() * batch_size
        total_nodes += batch_size
    
    return total_loss / total_nodes if total_nodes > 0 else 0


def test(model, loader, devices):
    """Evaluate the model on all splits using batches"""
    model.eval()
    masks = ['train_mask', 'val_mask', 'test_mask']
    accs, macro_f1s = [], []
    
    # Use the first GPU for evaluation
    device = devices[0]
    model = model.to(device)  # Ensure model is on primary device
    
    with torch.no_grad():
        for mask in masks:
            correct = 0
            total = 0
            y_true, y_pred = [], []
            
            for batch in loader:
                batch = batch.to(device)  # Move batch to primary device
                # Only process nodes belonging to the current mask
                if hasattr(batch, mask) and batch[mask].sum() > 0:
                    out = model(batch.x, batch.edge_index)
                    pred = out.argmax(dim=-1)
                    
                    # Compute accuracy
                    mask_tensor = batch[mask]

                    # Store for F1 calculation
                    y_true.append(batch.y[mask_tensor].cpu())
                    y_pred.append(pred[mask_tensor].cpu())
            
            # Convert lists to tensors for F1 calculation
            if y_true and y_pred:
                y_true = torch.cat(y_true)
                y_pred = torch.cat(y_pred)
                
                acc, macro_f1, _ = compute_acc_and_f1(y_pred.cpu().detach().numpy(), y_true.cpu().detach().numpy())
                accs.append(acc)
                macro_f1s.append(macro_f1)
                print(mask, y_true.shape)
    
    return accs, macro_f1s


def prepare_data_for_ogbn_products(graph_data):
    """
    Prepare ogbn-products graph data specifically for NeighborLoader
    """
    from torch_geometric.data import Data
    
    # Create a new clean Data object with only tensor attributes
    clean_data = Data(
        x=graph_data.x,
        edge_index=graph_data.edge_index,
        y=graph_data.y if graph_data.y.dim() == 1 else graph_data.y.squeeze(),
        train_mask=graph_data.train_mask,
        val_mask=graph_data.val_mask,
        test_mask=graph_data.test_mask
    )
    
    # Store the number of classes
    num_classes = len(graph_data.label_name) if hasattr(graph_data, 'label_name') else graph_data.y.max().item() + 1
    
    return clean_data, num_classes


def main():
    args = parse_args()
    print(args)
    
    # Set up GPUs
    gpu_ids = [int(i) for i in args.gpus.split(',')]
    devices = [torch.device(f'cuda:{i}') for i in gpu_ids]
    # devices = [torch.device("cpu")]
    print(f"Using GPUs: {gpu_ids}")
    
    # For tracking results across runs
    final_acc_list, final_macro_f1_list, timer_list = [], [], []
    
    # Setup output file
    if args.write_result:
        os.makedirs("../../results/GNN", exist_ok=True)
        write_file = open(f"../../results/GNN/{args.dataset}{'' if not args.re_split else '_s'}.csv", 
                         mode='a', newline='')
    
    # Run multiple times for stability
    for run in range(args.run_times):
        set_seed(run)
        primary_device = devices[0]  # Use first GPU as primary
        
        # Load graph data (on CPU to avoid OOM)
        print(f"Loading graph dataset (run {run+1}/{args.run_times})...")
        graph_data = load_graph_dataset_for_gnn(
            dataset_name=args.dataset,
            device=torch.device('cpu'),  # Load to CPU first
            emb_model=args.encoder_name if args.encoder_name else "shallow",
            re_split=args.re_split
        )
        print(f"Graph loaded: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges, features: {graph_data.x.shape}")
        loader_data, _ = prepare_data_for_ogbn_products(graph_data)

        # Create data loaders for batched processing
        print("Creating data loaders...")
        loader = RandomNodeLoader(loader_data, num_parts=10, shuffle=True)
        
        # Create model (on primary GPU)
        print("Creating model...")
        gnn_model = GNNEncoder(
            input_dim=graph_data.x.shape[1],
            hidden_dim=args.hidden_dim,
            output_dim=len(graph_data.label_name) if hasattr(graph_data, 'label_name') else graph_data.y.max().item() + 1,
            n_layers=args.n_layers,
            gnn_type=args.gnn_type,
            dropout=args.dropout,
            use_softmax=args.use_softmax,
            batch_norm=args.batch_norm,
            residual_conn=args.residual_conn,
            jump_knowledge=args.jump_knowledge
        ).to(primary_device)

        if len(devices) > 1:
            print(f"Using DataParallel with {len(devices)} GPUs")
            # Use the list of GPU IDs to specify which GPUs to use
            gnn_model = torch.nn.DataParallel(gnn_model, device_ids=[i for i in range(len(devices))])
        
        if run == 0:
            trainable_params = sum(p.numel() for p in gnn_model.parameters() if p.requires_grad)
            print(f"[GNN] Number of parameters: {trainable_params}")
        
        # Optimizer
        optimizer = torch.optim.Adam(
            gnn_model.parameters(), 
            lr=args.learning_rate, 
            weight_decay=args.weight_decay
        )
        
        # Training loop
        best_eval_acc = best_test_acc = 0.0
        best_eval_mac_f1 = best_test_mac_f1 = 0.0
        st_time, counter = time.time(), 0
        
        for epoch in range(1, args.epochs+1):
            # Train
            start_time = time.time()
            train_loss = train(gnn_model, loader, optimizer, devices)
            train_time = str(round((time.time() - start_time) / 60, 2)) + " Min"
            
            # Evaluate
            start_eval = time.time()
            accs, macro_f1s = test(gnn_model, loader, devices)
            eval_time = str(round((time.time() - start_eval) / 60, 2)) + " Min"
            
            train_acc, val_acc, test_acc = accs
            train_mac_f1, val_mac_f1, test_mac_f1 = macro_f1s
            
            # Check for improvement
            if val_acc > best_eval_acc:
                best_eval_acc, best_test_acc = val_acc, test_acc
                best_eval_mac_f1, best_test_mac_f1 = val_mac_f1, test_mac_f1
                counter = 0
            else:
                counter += 1
            
            # Print progress
            if epoch % args.print_freq == 0:
                print(f"Epoch {epoch:03d}   Loss: {train_loss:.4f}   "
                      f"Train acc: {train_acc:.3f}   Val acc: {val_acc:.3f}   Test acc: {test_acc:.3f}   "
                      f"Train F1: {train_mac_f1:.3f}   Val F1: {val_mac_f1:.3f}   Test F1: {test_mac_f1:.3f} "
                      f"Train Time: {train_time}   Eval Time: {eval_time}")
            
            # Early stopping
            if counter >= args.patience:
                print(f"Early stopping after {epoch} epochs")
                break
        
        # Record results
        total_time = round(time.time() - st_time, 3)
        timer_list.append(total_time)
        print(f'[Run {run+1}/{args.run_times}] '
              f'Test Acc: {best_test_acc:.2f}   '
              f'Test Macro-F1: {best_test_mac_f1:.2f}   '
              f'Time: {total_time:.3f}s  \n'
             )
        
        final_acc_list.append(best_test_acc)
        final_macro_f1_list.append(best_test_mac_f1)
       
        # Write results
        if args.write_result:
            write_file.write(f"{args.dataset},{run},{args.encoder_name},{args.batch_norm},"
                            f"{args.gnn_type},{best_test_acc},"
                            f"{best_test_mac_f1},{timer_list[-1]/60:.3f} Min\n")
    
    # Report final statistics
    print("\n" + "="*50)
    print(f"Final results for {args.dataset} with {args.gnn_type}:")
    print(f"Test Accuracy: {np.mean(final_acc_list):.2f} ± {np.std(final_acc_list):.2f}")
    print(f"Test Macro-F1: {np.mean(final_macro_f1_list):.2f} ± {np.std(final_macro_f1_list):.2f}")
    print(f"Average Runtime: {np.mean(timer_list)/60:.2f} minutes")
    print("="*50)
    
    if args.write_result:
        write_file.close()


if __name__ == "__main__":
    main()
