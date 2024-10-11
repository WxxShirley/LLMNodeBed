from model import TextLoraModel, descriptions
import argparse
import torch 
from dataset import KHopSubgraphDataset
import tqdm
from torch_geometric.data import DataLoader
from torch.utils.data import ConcatDataset
import torch.nn as nn 
from torch_geometric.utils import to_undirected
import time
import sys 
sys.path.append("../../")
from common import load_graph_dataset_for_zerog, get_cur_time


def build_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset", type=str, default="cora", help="Pre-training datasets")
    parser.add_argument("--test_datasets", type=str, default="cora")
    parser.add_argument("--model_dir", type=str, default="ckpts", help="Folder to save model")
    
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--if_norm", action="store_true", default=True, help="Indicator of normalization")
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--text_encoder", type=str, default="SentenceBert", help="Type of text encoder")
    parser.add_argument("--R", type=int, default=5, help="round")
    parser.add_argument("--test_R", type=int, default=5)

    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate of optimizer")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4, help="gradient accumulation steps")
    parser.add_argument("--k", type=int, default=2, help="k-hop subgraph")
    parser.add_argument("--batch_size", type=int, default=2)
    
    parser.add_argument("--use_lora", type=int, default=1)
    
    args = parser.parse_args()

    return args 


def eval(test_datasetname, test_gdata):
    model.eval()
    with torch.no_grad():
        text_features = []
        # for text in tqdm.tqdm(test_gdata.raw_texts, desc="[MODEL EVAL] Processing node texts"):
        for text in test_gdata.raw_texts:
            cur_text_feature = model.text_forward(text).cpu()
            text_features.append(cur_text_feature)
        
        desc = descriptions[test_datasetname]
        text_features.append(model.text_forward(desc).cpu())
        node_embeds = torch.cat(text_features, dim=0).to(device)

        label_features = []
        # for text in tqdm.tqdm(test_gdata.label_text, desc="[MODEL EVAL] Processing label texts"):
        for text in test_gdata.label_text:
            cur_label_feature = model.text_forward(text).cpu()
            label_features.append(cur_label_feature)
        label_embeds = torch.cat(label_features, dim=0).to(device)

        res = model.zero_shot_eval(node_embeds, label_embeds, test_gdata)
        return res


if __name__ == "__main__":
    args = build_args()
    device = torch.device(args.device)
    
    print('= ' * 20)
    print('## Starting Time:', get_cur_time(), flush=True)
    print(args, "\n")

    # Step 1 - Load all Test Graphs
    test_names, test_graphs = args.test_datasets.split(","), []
    for dataset_name in test_names:
        graph_data = load_graph_dataset_for_zerog(dataset_name, device)
        if dataset_name in ["citeseer", "arxiv"]:
            graph_data.edge_index = to_undirected(graph_data.edge_index)
        
        test_graphs.append(graph_data)
    print(f"[STAGE 1] Loading {len(test_graphs)} test graphs {test_names} ...")
    
    # Step 2 - Load Wrapped (L)LM Encoder Model
    model = TextLoraModel(args)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[STAGE 2] Preparing Text Encoder {args.text_encoder} with # Parameters {total_params} ...")

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=5e-4)
    
    # Step 3 - Preparing Training Data
    train_sets = []
    for dataset_name in args.dataset.split(","):
        train_graph_data = load_graph_dataset_for_zerog(dataset_name, device)
        num_hops = 1 if dataset_name in ["wikics", "arxiv"] else args.k 
        train_flg = dataset_name in test_names
        max_nodes = 100 if args.text_encoder in ["MiniLM", "SentenceBert"] else 50
        k_hop_dataset = KHopSubgraphDataset(train_graph_data, num_hops=num_hops, max_nodes=max_nodes, dataset_name=dataset_name, train_flg=train_flg)
        
        train_sets.append(k_hop_dataset)
        concat_dataset = ConcatDataset(train_sets)
    train_dataloader = DataLoader(concat_dataset, batch_size=args.batch_size, shuffle=True)
    print(f"[STAGE 3] Loading Training subgraphs from {args.dataset}, forming {len(train_dataloader)} train-loaders ...")
    
    # [Optional] Step 4 - Static (or Zero-shot) Evaluation
    res_list = []
    for idx, test_graph_data in enumerate(test_graphs):
        res = eval(test_names[idx], test_graph_data)
        res_list.append(res)
    print(f"[STAGE 4] Finish Zero-shot Evaluation with Performance {res_list} on {test_names}")
    
    # Step 5 - Model Training 
    best_test_dicts = {name: [0, 0, 0] for name in test_names}
    for i in range(args.epoch):
        model.train()

        start_time = time.time()
        for step, batch in enumerate(train_dataloader):
            data = batch[0].to(device)
            loss = model(data)

            if torch.isnan(loss).any():
                print(loss)
                break 
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                loss.backward()
            
            if (step + 1) % args.gradient_accumulation_steps == 0:
                print(f"[MODEL TRAINING] Step: {step+1:03d} | loss: {loss.item():.5f}")
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
        
        torch.cuda.empty_cache()
        print(f"[MODEL TRAINING] Epoch {i+1:03d} Cost Times {time.time() - start_time:.3f}s")
        res_list = []
        for cur_graphname, test_data in zip(test_names, test_graphs): 
            test_scores, eval_scores = eval(cur_graphname, test_data)
            res_list.append(test_scores)
            
            if eval_scores[0] > best_test_dicts[cur_graphname][0]:
                best_test_dicts[cur_graphname] = [eval_scores[0]] + test_scores
        print(f"[MODEL EVAL] Epoch {i+1:03d} with Performance {res_list} on {test_names}")
    
    print("\n\n")
    for graphname, scores in best_test_dicts.items():
        print(f"[{graphname}] Best Test Acc {scores[1]:.3f} Best Test F1 {scores[2]:.3f}")
    
    print('\n## Finishing Time:', get_cur_time(), flush=True)
    print('= ' * 20)
    print("Done!\n\n")
    