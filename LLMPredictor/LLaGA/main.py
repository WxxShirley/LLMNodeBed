import os 
import csv
from tqdm import tqdm 
import torch 
import json 
import sys
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
sys.path.append("../..")
from common import load_graph_dataset_for_llaga, set_seed, get_cur_time, compute_acc_and_f1
from common import save_checkpoint, reload_best_model, MODEL_PATHs as llm_paths, UNKNOW
from llaga_model import LLaGAModel
import argparse
import time
from dataset import LLaGADataset, build_laplacian_emb, build_hopfield_emb, classes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--re_split", type=int, default=1)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--llm", type=str, default="Mistral-7B")
    parser.add_argument("--lm_encoder", type=str, default="roberta")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--token_counter", type=int, default=1)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_gpus", type=int, default=1)
    
    # Configuration of Neighborhood Encoding
    parser.add_argument("--neighbor_template", default="HO", choices=["ND", "HO"])
    parser.add_argument("--nd_mean", type=int, default=1)
    parser.add_argument("--k_hop", type=int, default=2)
    parser.add_argument("--sample_size", type=int, default=10)
    parser.add_argument("--hopfield", type=int, default=4)
    
    parser.add_argument("--max_txt_length", type=int, default=256)
    parser.add_argument("--max_ans_length", type=int, default=16)
    
    # Configuration of Linear Projection
    parser.add_argument("--n_linear_layer", type=int, default=2)
    parser.add_argument("--hidden_dim", type=int, default=2048)
    parser.add_argument("--output_dim", type=int, default=2048)
    
    # Configuration of Model Training 
    parser.add_argument("--num_epochs", type=int, default=6)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--wd", type=float, default=0.05)
    parser.add_argument("--output_dir", type=str, default="../../results/LLaGA")
    parser.add_argument("--grad_steps", type=int, default=4)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--llm_freeze", type=int, default=1)
    
    args = parser.parse_args() 
    # wandb.init(project="LLaGA", name=f"{args.dataset}_{args.llm}")
    
    print("= " * 20)
    print("## Starting Time:", get_cur_time(), flush=True)
    print(args, "\n")
    
    # device = torch.device(args.device)
    device = torch.device("cuda:"+str(args.gpu_id))
    set_seed(args.seed)
    
    llm_path = llm_paths[args.llm]
    args.output_dim = {"Qwen-3B": 2048, "Qwen-7B": 3584, "Mistral-7B": 4096, "Llama-8B": 4096, "Qwen-14B": 5120, "Qwen-32B": 5120}[args.llm]
    
    # Pre-process Node Classification Training Data 
    graph_data = load_graph_dataset_for_llaga(dataset_name=args.dataset, device=device, encoder=args.lm_encoder, re_split=args.re_split)
    train_dataset = LLaGADataset(args, graph_data=graph_data, data_type="train", repeats=1)
    val_dataset = LLaGADataset(args, graph_data=graph_data, data_type="val", repeats=1)
    test_dataset =LLaGADataset(args, graph_data=graph_data, data_type="test", repeats=1)
    hopfield_emb = build_hopfield_emb(graph_data.x, graph_data.edge_index, n_layers=args.hopfield)
    structure_emb = build_laplacian_emb(args.k_hop, args.sample_size).to(device)
    print(f"[DATA] {'Supervised 6:2:2' if args.re_split else 'Semi-supervised Setting'} # Train {len(train_dataset)} # Val {len(val_dataset)} # Test {len(test_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=False, pin_memory=True, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, drop_last=False, pin_memory=True, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=args.eval_batch_size, drop_last=False, pin_memory=True, shuffle=False)
    
    # Load LLaGA Model 
    model = LLaGAModel(args, 
                       llm_path, 
                       graph_embedding=graph_data.x if args.neighbor_template == "ND" else hopfield_emb, 
                       structure_embedding=structure_emb if args.neighbor_template == "ND" else None)
    
    re_split_str = '_s' if args.re_split else ''
    
    # (Temporary) Token Counter to Decide MAX_ANS_LENGTH & MAX_TXT_LENGTH
    if args.token_counter: 
        input_lengths, txt_lengths, output_lengths = [], [], []
        for sample in train_dataset + val_dataset + test_dataset:
            encoded_query = model.tokenizer(sample["query"])
            encoded_txt = model.tokenizer(sample["origin_txt"])
            encoded_label = model.tokenizer(sample["label"])
            input_lengths.append(len(encoded_query["input_ids"]))
            txt_lengths.append(len(encoded_txt["input_ids"]))
            output_lengths.append(len(encoded_label["input_ids"]))
        print(f"[ANALYSIS] # Avg Input Token {sum(input_lengths)/len(input_lengths):.3f} # Avg txt Token {sum(txt_lengths)/len(txt_lengths):.2f}  # Avg Output Token {sum(output_lengths)/len(output_lengths):.3f}  Max Output Token {max(output_lengths)}")

    params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        [{'params': params, 'lr': args.lr, 'weight_decay': args.wd}]
    )
    
    trainable_params, all_params = model.print_trainable_params()
    print(f"Trainable params {trainable_params} || all params {all_params} || trainable% {100 * trainable_params / all_params:.5f}")
    
    num_training_steps = args.num_epochs * len(train_loader)
    progress_bar = tqdm(range(num_training_steps))
    
    best_val_loss = float('inf')
    
    model.model.gradient_checkpointing_enable()
    llm_config_str = f"{args.llm}_{args.neighbor_template}_Epoch{args.num_epochs}{re_split_str}{'_LoRA' if not args.llm_freeze else ''}"
    folder_str = f"{args.output_dir}/output/{args.dataset}"
    st_time = time.time()
    
    for epoch in range(args.num_epochs):
        model.train() 
        
        epoch_loss, accum_loss = 0.0, 0.0 
        
        for step, batch in enumerate(train_loader):
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            
            clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)
            optimizer.step() 
            epoch_loss, accum_loss = epoch_loss + loss.item(), accum_loss + loss.item()

            if (step + 1) % args.grad_steps == 0:
                lr = optimizer.param_groups[0]["lr"]
                # wandb.log({'Accum Loss': accum_loss / args.grad_steps})
                # wandb.log({'Step Loss': loss.item()})
                accum_loss = 0.

            progress_bar.update(1)
        
        print(f"[TRAIN] Epoch {epoch+1}|{args.num_epochs}: Train Loss (Epoch Mean): {epoch_loss / len(train_loader):.5f}")
        # wandb.log({'Train Loss (Epoch Mean)': epoch_loss / len(train_loader)})
        
        val_loss = 0.0 
        model.eval()
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                loss = model(batch)
                val_loss += loss.item()
            print(f"[VAL] Epoch: {epoch+1}|{args.num_epochs}: Val Loss: {val_loss / len(val_loader):.5f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss 
            best_epoch = epoch 
            save_checkpoint(model, epoch, folder_str, llm_config_str, is_best=True)
            
        if epoch - best_epoch >= args.patience:
            print(f"[TRAIN] Early stop at epoch {epoch+1}")
            break 
    
    train_secs = time.time() - st_time
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    
    model = reload_best_model(model, folder_str, llm_config_str)
    model.eval()
    
    os.makedirs(f"{args.output_dir}/prediction", exist_ok=True)
    path = f"{args.output_dir}/prediction/{args.dataset}_{args.llm}{re_split_str}_seed{args.seed}.json"
    print(f"\n[Prediction] Write predictions on {path} ...")
    progress_bar_test = tqdm(range(len(test_loader)))
    pred_labels, gt_labels = [], []
    st_time = time.time()
    with open(path, 'w') as file:
        for step, batch in enumerate(test_loader):
            with torch.no_grad():
                id_list, predictions = model.inference(batch)
                
                for node_idx, llm_pred in zip(id_list, predictions):
                    node_idx = node_idx.item()
                    # print(llm_pred)
                    pred_label = llm_pred[:llm_pred.index("</s>")] if "</s>" in llm_pred  else llm_pred
                    pred_label = pred_label if pred_label in classes[args.dataset] else UNKNOW
                    write_obj = {
                        "id": node_idx,
                        "pred": llm_pred,
                        "ground-truth": classes[args.dataset][graph_data.y[node_idx].item()]
                    }
                    pred_labels.append(pred_label) 
                    gt_labels.append(write_obj["ground-truth"])
                    file.write(json.dumps(write_obj) + "\n")
                    file.flush()
                progress_bar_test.update(1)
    inference_secs = time.time() - st_time
    
    acc, macro_f1, weight_f1 = compute_acc_and_f1(pred_labels, gt_labels)
    with open(f"{args.output_dir}/summary{'_semi' if not args.re_split else ''}.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([args.dataset, args.llm, acc, macro_f1, weight_f1, 
                         args.neighbor_template, args.hidden_dim, args.lm_encoder, args.num_epochs, args.patience, args.batch_size, args.lr, args.seed,
                         f"Train Minutes-{train_secs/60:.3f}", f"Inference Seconds-{inference_secs:.2f}"])
    print(f"Accuracy {acc:.2f}  Macro F1-Score {macro_f1:.2f}  Weight F1-Score {weight_f1:.2f}")
    print('\n## Finishing Time:', get_cur_time(), flush=True)
    print('= ' * 20)
    print("Done!")
