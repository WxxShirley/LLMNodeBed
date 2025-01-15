from model import  Text_Lora
import argparse
import tqdm
from SubgraphDataset import kHopSubgraphDataset, kHopSubgraphDataset_Arxiv
from torch_geometric.data import DataLoader
from torch.utils.data import ConcatDataset
import torch.nn as nn
import os
import sys
import torch
import time
sys.path.append("../../")
from common import load_graph_dataset, load_graph_dataset_for_zerog, get_cur_time, set_seed
from common import ZeroG_DESC as descriptions

def build_args():
    parser = argparse.ArgumentParser(description='UGAD')
    parser.add_argument("--train_dataset", type=str, nargs='+',
                        default=["cora"], help="Pre-train datasets for this model")
    parser.add_argument("--data_dir", type=str,
                        default="./datasets/", help="Data directory")
    parser.add_argument("--model_dir", type=str,
                        default="./ckpts/", help="Folder to save model")

    # Model Configuration settings
    parser.add_argument("--seed", type=int, nargs="+",
                        default=0, help="Random seed")
    parser.add_argument("--decay_rate", type=float, default=1,
                        help="Decay rate of learning rate")
    parser.add_argument("--decay_step", type=int, default=100,
                        help="Decay step of learning rate")
    parser.add_argument("--lr", type=float, default=0.0001,
                        help="Learning rate of optimizer")
    parser.add_argument("--gradient_accumulation_steps",
                        type=int, default=4, help="gradient accumulation steps")
    parser.add_argument("--k", type=int, default=2, help="k-hop subgraph")
    parser.add_argument("--max_node", type=int, default=100, help="max num of nodes in the k-hop subgraph")
    parser.add_argument("--test_re_split", type=int, default=1)

    # Dataset settings
    parser.add_argument("--test_dataset", type=str, nargs='+',
                        default=["cora"], help="Pre-train datasets for this model")
    
    # Training settings
    parser.add_argument("--epoch", type=int, default=10,
                        help="The max number of epochs")
    parser.add_argument("--if_norm", action='store_true',
                        default=False, help="Indicator of normalization")

    # GPU settings
    parser.add_argument("--no_cuda", action='store_true',
                        default=False, help="Indicator of GPU availability")
    parser.add_argument("--device", type=int, default=1,
                        help='Which gpu to use if any')

    # Text settings
    parser.add_argument("--text_encoder", type=str,
                        default='SentenceBert', help="Text encoder type")
    parser.add_argument("--R", type=int, default=10, help="round")

    args = parser.parse_args()

    return args


def eval(i, idx, model, test_data, args):
    model.eval()
    with torch.no_grad():
        text_features = []
        for text in tqdm.tqdm(test_data.raw_texts, desc="Processing texts"):
            tokens = model.tokenizer(
                text, max_length=256, return_tensors='pt', truncation=True, padding=True).to(args.device)
            text_features.append(model.lora_model(**tokens)[0][:, 0, :].cpu())

        desc = descriptions[args.test_dataset[idx]]
        tokens = model.tokenizer(
            desc, max_length=256, return_tensors='pt', truncation=True, padding=True).to(args.device)
        text_features.append(model.lora_model(**tokens)[0][:, 0, :].cpu())

        node_embeds = torch.cat(text_features, dim=0).to(args.device)
        label_features = []
        for text in tqdm.tqdm(test_data.label_text, desc="Processing label texts"):
            tokens = model.tokenizer(
                text, max_length=256, return_tensors='pt', truncation=True, padding=True).to(args.device)
            label_features.append(model.lora_model(**tokens)[0][:, 0, :].cpu())
        label_embeds = torch.cat(label_features, dim=0).to(args.device)
        args.test_data = args.test_dataset[idx]


        res, score_dict = model.zero_shot_eval(
            node_embeds, label_embeds, test_data.to(args.device))
        
        return res, score_dict


if __name__ == '__main__':

    # Configurations
    args = build_args()
    print("Welcome!")

    set_seed(args.seed[0])

    # GPU initialization
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    args.device = torch.device('cuda:{}'.format(
        args.device) if args.cuda else 'cpu')
    
    # start
    print('= ' * 20)
    print('## Starting Time:', get_cur_time(), flush=True)
    print(args, "\n")

    # Step 1 - Load Test Graphs

    test_datasets = []  
    
    for dataset_name in args.test_dataset:
        test_data = load_graph_dataset(dataset_name = dataset_name, device = args.device, re_split = args.test_re_split)
        test_data.label_text = test_data.label_name
        test_data.dataset_name = dataset_name
        test_datasets.append(test_data)
    
    print(f"[STAGE 1] Loading All Test Graphs {args.test_dataset} ...")

    # Step 2 - Load Wrapped (L)LM Encoder Model
    if args.text_encoder == 'llama':
        model = Text_Lora(args)
    else:
        model = Text_Lora(args).to(args.device)
    total_params = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    
    print(f"[STAGE 2] Preparing Text Encoder {args.text_encoder} with # Trainable Parameters {total_params} ...")

 
    optimizer = torch.optim.Adam(filter(
        lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.decay_step, gamma=args.decay_rate)
    
    # Step 3 - Preparing Training Data
    train_datasets = []
    if 'reddit' in args.train_dataset and 'instagram' in args.train_dataset:
        subpath = "reddit_ins_subgraph.pt"
        if not os.path.exists(subpath):
            for dataset_name in args.train_dataset:
                data =  load_graph_dataset(dataset_name=dataset_name, device=args.device, re_split=True)
                data.label_text = data.label_name
                data.dataset_name = dataset_name
                k_hop_dataset = kHopSubgraphDataset(
                data, num_hops=args.k, max_nodes=args.max_node, dataset_name=dataset_name)
                train_datasets.append(k_hop_dataset)
            concat_dataset = ConcatDataset(train_datasets)
            torch.save(concat_dataset, subpath)
        else:
            concat_dataset = torch.load(subpath)
    else:
        for dataset_name in args.train_dataset:
            data = load_graph_dataset_for_zerog(dataset_name=dataset_name, device=args.device, re_split=True)
            data.label_text = data.label_name
            data.dataset_name = dataset_name
            if dataset_name == "arxiv":
                k_hop_dataset = kHopSubgraphDataset_Arxiv(
                    data, num_hops=1, max_nodes=args.max_node, dataset_name=dataset_name)
            elif dataset_name == "wikics":
                k_hop_dataset = kHopSubgraphDataset(
                    data, num_hops=1, max_nodes=args.max_node, dataset_name=dataset_name)
            else:
                k_hop_dataset = kHopSubgraphDataset(
                    data, num_hops=args.k, max_nodes=args.max_node, dataset_name=dataset_name)
            train_datasets.append(k_hop_dataset)

            concat_dataset = ConcatDataset(train_datasets)
    
    train_dataloader = DataLoader(concat_dataset, batch_size=2, shuffle=True)
    print(f"[STAGE 3] Loading Training Subgraphs from {args.train_dataset}, forming {len(train_dataloader)} train-loaders ...")

    # Step 4 - Model Training
    max_acc = 0     
    for i in range(args.epoch):
        model.train()
   
        res_list=[]
        for idx,test_data in enumerate(test_datasets):
            res = eval(i, idx, model, test_data, args)
            res_list.append(res)
        print(res_list)

        print("batch: {}".format(i))
        start_time = time.time()
        for step, batch in enumerate(train_dataloader):
            data = batch[0].to(args.device)
            loss = model(data,args)
            if torch.isnan(loss).any():
                break
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                loss.backward()
            if (step + 1) % args.gradient_accumulation_steps == 0:
                print("step: {}, loss: {}".format(step, loss.item()))
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
        end_time = time.time()
        print("time: {}".format(end_time - start_time))
        
        res_list = []
        for idx, test_data in enumerate(test_datasets): 
            res, score_dict = eval(i, idx, model, test_data, args)
            print(f"[MODEL EVAL] Epoch {i + 1:03d} {test_data.dataset_name} {score_dict}")
            res_list.append(res)
        print("Epoch: {}, Step: {}, acc: {}".format(i, step, res_list))

