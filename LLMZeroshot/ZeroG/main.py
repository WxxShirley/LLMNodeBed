from model import TextLoraModel, descriptions
import argparse
import torch 
import torch.nn as nn
from dataset import KHopSubgraphDataset
from torch_geometric.data import DataLoader
import time
import sys 
from torch.utils.data import ConcatDataset
import tqdm
sys.path.append("../../")
from common import load_graph_dataset, get_cur_time, set_seed, save_checkpoint
from common import CLASSES as classes, ZeroG_LABEL_DESC as label_descriptions


def build_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--test_dataset", type=str, default="cora,citeseer")
    parser.add_argument("--model_dir", type=str, default="../../results/ZeroG/output", help="Folder to save model")
    
    parser.add_argument("--epoch", type=int, default=4)
    parser.add_argument("--if_norm", action="store_true", default=True, help="Indicator of normalization")
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--text_encoder", type=str, default="SentenceBert", help="Type of text encoder")
    parser.add_argument("--test_R", type=int, default=11)
    parser.add_argument("--k_hop", type=int, default=2)
    parser.add_argument("--R", type=int, default=0)

    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate of optimizer")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--decay_rate", type=float, default=1, help="Decay rate of learning rate")
    parser.add_argument("--decay_step", type=int, default=100, help="Decay step of learning rate")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="gradient accumulation steps")

    parser.add_argument("--use_lora", type=int, default=1)
    parser.add_argument("--test_re_split", type=int, default=1)
    parser.add_argument("--use_train", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    
    args = parser.parse_args()
    
    if args.use_lora == 0:
        args.lr = 2e-5
    return args 


def eval(test_gdata, test_gname):
    model.eval()
    with torch.no_grad():
        text_features = []
        for text in tqdm.tqdm(test_gdata.raw_texts, desc="[MODEL EVAL] Processing node texts"):
        # for text in test_gdata.raw_texts:
            cur_text_feature = model.text_forward(text).cpu()
            text_features.append(cur_text_feature)
        
        desc = descriptions[test_gname]
        text_features.append(model.text_forward(desc).cpu())
        node_embeds = torch.cat(text_features, dim=0).to(device)

        label_features = []
        for text in tqdm.tqdm(test_gdata.label_name, desc="[MODEL EVAL] Processing label texts"):
        # for text in classes[test_gname]:
            cur_label_feature = model.text_forward(text).cpu()
            label_features.append(cur_label_feature)
        label_embeds = torch.cat(label_features, dim=0).to(device)

        return model.zero_shot_eval(node_embeds, label_embeds, test_gdata)


if __name__ == "__main__":
    args = build_args()
    device = torch.device(args.device)
    
    print('= ' * 20)
    print('## Starting Time:', get_cur_time(), flush=True)
    print(args, "\n")

    # Step 1 - Load Test Graphs
    set_seed(args.seed)
    test_graphs = []
    for test_dataset_name in args.test_dataset.split(","):
        test_graph = load_graph_dataset(dataset_name=test_dataset_name, device=device, re_split=args.test_re_split)
        test_graph.label_name = [f"{label}. {label_descriptions[test_dataset_name].get(label, '')}" for label in test_graph.label_name]
        test_graphs.append(test_graph)
    print(f"[STAGE 1] Loading All Test Graphs {args.test_dataset} ...")
    
    # Step 2 - Load Wrapped (L)LM Encoder Model
    model = TextLoraModel(args)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[STAGE 2] Preparing Text Encoder {args.text_encoder} with # Trainable Parameters {total_params} ...")
   
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.decay_step, gamma=args.decay_rate)
    
    # Step 3 - Preparing Training Data
    train_data = []
    for train_graph in args.dataset.split(","):
        cur_k_hop = 1 if train_graph in ["wikics", "arxiv", "history", "photo", "computer"] else args.k_hop
        graph_data = load_graph_dataset(dataset_name=train_graph, device=device, re_split=True)
        if train_graph == "arxiv": 
            graph_data.label_name = classes[train_graph]
        graph_data.label_name = [f"{label}. {label_descriptions[train_graph].get(label, '')}" for label in graph_data.label_name]
        # print(graph_data.label_name)

        train_corpus = KHopSubgraphDataset(graph_data, num_hops=cur_k_hop, max_nodes=100, dataset_name=train_graph, train_flg=args.use_train)
        train_data.append(train_corpus)
        if train_graph == "citeseer" and "arxiv" in args.dataset:
            train_data.extend([train_corpus for _ in range(9)])
    concat_dataset = ConcatDataset(train_data)
    train_dataloader = DataLoader(concat_dataset, batch_size=args.batch_size, shuffle=True)
    print(f"[STAGE 3] Loading Training Subgraphs from {args.dataset}, forming {len(train_dataloader)} train-loaders ...")
    
    # Step 4 - Model Training 
    best_scores = 0.0
    for i in range(args.epoch):
        model.train()

        start_time = time.time()
        epoch_loss = 0.0 
        for step, batch in enumerate(train_dataloader):
            data = batch[0].to(device)
            loss = model(data)
            epoch_loss += loss.item()
      
            if torch.isnan(loss).any():
                print(loss)
                break 
            
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps 
                loss.backward()
            if (step+1) % args.gradient_accumulation_steps == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()   
            
        print(f"\n[MODEL TRAINING] Epoch {i+1:03d} Loss {epoch_loss/len(train_dataloader):.4f} Cost Times {time.time() - start_time:.3f}s")
        
        total_eval_acc = 0.0
        for cur_graphname, test_graph in zip(args.test_dataset.split(","), test_graphs):
            test_dict = eval(test_graph, cur_graphname)
            print(f"[MODEL EVAL] Epoch {i+1:03d} on {cur_graphname} {test_dict}")
            total_eval_acc += test_dict[0][0]
        avg_eval_acc = total_eval_acc / len(test_graphs)
        if avg_eval_acc > best_scores:
            best_scores = avg_eval_acc 
            dataset_str = args.dataset.replace(",", "-")
            save_checkpoint(model, cur_epoch=i+1, folder_str=f"{args.model_dir}", config_str=f"{dataset_str}-R{args.R}-Epoch{args.epoch}", is_best=True)
         
    print("\n\n")
    print('\n## Finishing Time:', get_cur_time(), flush=True)
    print('= ' * 20)
    print("Done!\n\n")
    