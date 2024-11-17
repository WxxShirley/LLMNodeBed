import os 
import torch 
import json 
import sys 
import csv
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
sys.path.append("../..")
from common import load_graph_dataset_for_llaga, set_seed, get_cur_time, compute_acc_and_f1
from graphgpt_model import GraphGPTModel
from dataset import GraphInstructionTuningDataset, GraphMatchingDataset, classes
import argparse
from ckpt import save_checkpoint, reload_best_model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--re_split", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--llm", type=str, default="Mistral-7B")
    parser.add_argument("--device", type=str, default="cuda:0")
    
    parser.add_argument("--do_stage1", type=int, default=1)
    parser.add_argument("--s1_k_hop", type=int, default=2)
    parser.add_argument("--s1_num_neighbors", type=int, default=5)
    parser.add_argument("--s1_max_txt_length", type=int, default=512)
    parser.add_argument("--s1_max_ans_length", type=int, default=256)
    parser.add_argument("--s1_epoch", type=int, default=2)
    parser.add_argument("--s1_batch_size", type=int, default=2)
    parser.add_argument("--s1_lr", type=float, default=1e-4)

    parser.add_argument("--do_stage2", type=int, default=1)
    parser.add_argument("--s2_num_neighbors", type=int, default=4)
    parser.add_argument("--s2_max_txt_length", type=int, default=256)
    parser.add_argument("--s2_max_ans_length", type=int, default=16)
    parser.add_argument("--s2_epoch", type=int, default=10)
    parser.add_argument("--s2_batch_size", type=int, default=4)
    parser.add_argument("--s2_lr", type=float, default=1e-4)
    parser.add_argument("--s2_patience", type=int, default=2)
    
    parser.add_argument("--output_dim", type=int, default=2048)
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--load_ground_embedding", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="output")
    
    args = parser.parse_args()
    
    print("=" * 20)
    print("## Starting Time:", get_cur_time(), flush=True)
    print(args, "\n")
    
    device = torch.device(args.device)
    set_seed(args.seed)
    
    llm_path = {
        "Qwen-3B": "/root/autodl-tmp/models/qwen/Qwen2___5-3B-Instruct", # 3B
        "Mistral-7B": "/root/autodl-tmp/models/Mistral-7B/snapshots/Mistral-7B-Instruct-v0.2", # 7B
    }[args.llm]
    args.output_dim = {"Qwen-3B": 2048, "Mistral-7B": 4096}[args.llm]
    
    # Prepare Data 
    graph_data = load_graph_dataset_for_llaga(dataset_name=args.dataset, device=torch.device("cpu"), encoder="roberta", re_split=args.re_split)
    graph_embedding = graph_data.x.to(device)
    if args.load_ground_embedding: 
        assert os.path.exists(f"../../datasets/ground_emb/{args.dataset}.pt"), "Please run `main_text_graph_grounding` to generate grounded embedding first!"
        graph_embedding = torch.load(f"../../datasets/ground_emb/{args.dataset}.pt").to(device)
    
    if args.do_stage1: 
        print("Preparing Stage 1 [Graph Matching] ...")
        graph_type = {"cora": "academic_network", "citeseer": "academic_network", "pubmed": "academic_network", "wikics": "academic_network", "arxiv": "academic_network", "reddit": "social_network", "instagram": "social_network"}[args.dataset]
        dataset = GraphMatchingDataset(graph_data=graph_data, k_hop=args.s1_k_hop, num_sampled_neighbors=args.s1_num_neighbors, graph_type=graph_type)
        train_loader = DataLoader(dataset, batch_size=args.s1_batch_size, drop_last=True, shuffle=True)
        
        model = GraphGPTModel(args, llm_path, graph_embedding=graph_embedding, stage="matching")
        
        params = [p for _, p in model.named_parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW([{'params': params, 'lr': args.s1_lr, 'weight_decay': args.wd}])
        
        trainable_params, all_params = model.print_trainable_params()
        print(f"Trainable params {trainable_params} || all params {all_params} || trainable% {100 * trainable_params / all_params:.5f}")
        
        num_training_steps = args.s1_epoch * len(train_loader)
        progress_bar = tqdm(range(num_training_steps))
        
        model.model.gradient_checkpointing_enable()
        for epoch in range(args.s1_epoch):
            model.train()
            
            epoch_loss, accum_loss = 0.0, 0.0
            
            for step, batch in enumerate(train_loader):
                optimizer.zero_grad() 
                
                loss = model(batch)
                loss.backward()
                
                clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)
                optimizer.step() 
                epoch_loss, accum_loss = epoch_loss + loss.item(), accum_loss + loss.item()
                
                if (step+1) % 4 == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    print(f"(Temporary) Step {step} in Epoch {epoch+1} Accum Loss {accum_loss:.4f}")
                    accum_loss = 0.0 
                
                progress_bar.update(1)
            
            print(f"[TRAIN] Epoch {epoch+1}|{args.s1_epoch}: Train Loss (Epoch Mean): {epoch_loss / len(train_loader):.5f}")
        
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
    
    if args.do_stage2: 
        print("Preparing Stage 2 [Instruction Tuning] ...")
        
        if not args.do_stage1: 
            model = GraphGPTModel(args, llm_path, graph_embedding=graph_embedding, stage="matching")
        
            params = [p for _, p in model.named_parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW([{'params': params, 'lr': args.s1_lr, 'weight_decay': args.wd}])
            trainable_params, all_params = model.print_trainable_params()
            print(f"Trainable params {trainable_params} || all params {all_params} || trainable% {100 * trainable_params / all_params:.5f}")
        else:
            print(f"Directly load stage1's pretrained graph projector layer")
            
        train_dataset = GraphInstructionTuningDataset(graph_data=graph_data, maximum_neighbors=args.s2_num_neighbors, dataset_name=args.dataset, data_type="train")
        val_dataset = GraphInstructionTuningDataset(graph_data=graph_data, maximum_neighbors=args.s2_num_neighbors, dataset_name=args.dataset, data_type="val")
        test_dataset = GraphInstructionTuningDataset(graph_data=graph_data, maximum_neighbors=args.s2_num_neighbors, dataset_name=args.dataset, data_type="test")
        
        train_loader = DataLoader(train_dataset, batch_size=args.s2_batch_size, drop_last=False, pin_memory=True, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.s2_batch_size*2, drop_last=False, pin_memory=True, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=args.s2_batch_size*2, drop_last=False, pin_memory=True, shuffle=False)
        
        num_training_steps = args.s2_epoch * len(train_loader)
        progress_bar = tqdm(range(num_training_steps))
        
        best_val_loss = float('inf')
        model.model.gradient_checkpointing_enable()
        
        llm_config_str = f"{args.llm}_Epoch{args.s2_epoch}"
        folder_str = f"{args.output_dir}/{args.dataset}"
        for epoch in range(args.s2_epoch):
        # for epoch in range(0):
            model.train()
            
            epoch_loss, accum_loss = 0.0, 0.0 
            
            for step, batch in enumerate(train_loader):
                optimizer.zero_grad()
                loss = model(batch) 
                loss.backward()
                
                clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)
                optimizer.step() 
                epoch_loss, accum_loss = epoch_loss + loss.item(), accum_loss + loss.item()

                if (step + 1) % 4 == 0:
                    lr = optimizer.param_groups[0]["lr"]
                    accum_loss = 0.

                progress_bar.update(1)
       
            print(f"[TRAIN] Epoch {epoch+1}|{args.s2_epoch}: Train Loss (Epoch Mean): {epoch_loss / len(train_loader):.5f}")
        
            val_loss = 0.0 
            model.eval()
            
            with torch.no_grad():
                for step, batch in enumerate(val_loader):
                    loss = model(batch)
                    val_loss += loss.item()
            print(f"[VAL] Epoch {epoch+1}|{args.s2_epoch}: Val Loss {val_loss / len(val_loader):.5f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss 
                best_epoch = epoch
                save_checkpoint(model, epoch, folder_str, llm_config_str, is_best=True)
            
            if epoch - best_epoch >= args.s2_patience: 
                print(f"[TRAIN] Earlt stop at epoch {epoch+1}")
                break 
        model = reload_best_model(model, folder_str, llm_config_str)
     
            
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    
    model.eval()
    
    os.makedirs(f"prediction/{args.dataset}", exist_ok=True)
    re_split_str = '_s' if args.re_split else ''
    path = f"prediction/{args.dataset}/{args.llm}{re_split_str}_seed{args.seed}.json"
    print(f"\n[Prediction] Write predictions on {path} ...")
    progress_bar = tqdm(range(len(test_loader)))
    
    pred_labels, gt_labels = [], []
    
    with open(path, 'w') as file:
        for step, batch in enumerate(test_loader):
            with torch.no_grad():
                id_list, predictions = model.inference(batch)
                
                for node_idx, llm_pred in zip(id_list, predictions):
                    node_idx = node_idx.item()
                    # print(llm_pred)
                    pred_label = llm_pred[:llm_pred.index("</s>")] if "</s>" in llm_pred  else llm_pred
                    write_obj = {
                        "id": node_idx,
                        "pred": llm_pred,
                        "ground-truth": classes[args.dataset][graph_data.y[node_idx].item()]
                    }
                    pred_labels.append(pred_label) 
                    gt_labels.append(write_obj["ground-truth"])
                    file.write(json.dumps(write_obj) + "\n")
                    file.flush()
                progress_bar.update(1)
    
    acc, f1 = compute_acc_and_f1(pred_labels, gt_labels)
    with open(f"summary{'_semi' if not args.re_split else ''}.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([args.dataset, args.llm, acc, f1, args.re_split, args.seed])
    print(f"Accuracy {acc:.3f}  F1-Score {f1:.3f}")
    print('\n## Finishing Time:', get_cur_time(), flush=True)
    print('= ' * 20)
    print("Done!")               
    