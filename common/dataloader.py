import torch 
import os
import json
from collections import defaultdict
from torch_geometric.utils import to_undirected


def re_split_data(num_node, train_percent=0.6, val_percent=0.2, test_percent=0.2, device="cuda:0"):
    import numpy as np
    node_ids = np.arange(num_node)
    np.random.shuffle(node_ids)
    
    train_ids = np.sort(node_ids[:int(num_node * train_percent)])
    val_ids = np.sort(node_ids[int(num_node * train_percent): int(num_node * (train_percent + val_percent))])
    test_ids = np.sort(node_ids[int(num_node * (train_percent + val_percent)): int(num_node * (train_percent + val_percent + test_percent))])
    
    train_mask = torch.tensor(np.array([idx in train_ids for idx in range(num_node)]))
    val_mask = torch.tensor(np.array([idx in val_ids for idx in range(num_node)]))
    test_mask = torch.tensor(np.array([idx in test_ids for idx in range(num_node)])) 

    return train_mask.to(device), val_mask.to(device), test_mask.to(device)
    

def load_graph_dataset(dataset_name, device, re_split=False, path_prefix="../.."):
    graph_data = torch.load(f"{path_prefix}/datasets/{dataset_name}.pt", weights_only=False).to(device)
    # Alternative 
    # graph_data.edge_index = to_undirected(graph_data.edge_index) if dataset_name in ["citeseer", "arxiv"] else graph_data.edge_index
    graph_data.edge_index = to_undirected(graph_data.edge_index) 
    
    if dataset_name in ["computer", "history", "photo"]:
        if re_split:
            graph_data.train_mask, graph_data.val_mask, graph_data.test_mask = re_split_data(graph_data.num_nodes, device=device)
        else:
            # Semi-supervised setting for Computer, History, and Photo (10%:10%:60% for train:val:test)
            graph_data.train_mask, graph_data.val_mask, graph_data.test_mask = re_split_data(graph_data.num_nodes, train_percent=0.1, val_percent=0.1, test_percent=0.6, device=device)
    else:
        if re_split:
            graph_data.train_mask, graph_data.val_mask, graph_data.test_mask = re_split_data(graph_data.num_nodes, device=device)
    
    return graph_data


def load_graph_dataset_for_gnn(dataset_name, device, re_split=False, path_prefix="../..", emb_model="shallow"):
    graph_data = load_graph_dataset(dataset_name, device, re_split, path_prefix)

    if emb_model != "shallow":
        assert os.path.exists(f"{path_prefix}/datasets/{emb_model}/{dataset_name}.pt")
        node_feat = torch.load(f"{path_prefix}/datasets/{emb_model}/{dataset_name}.pt", map_location=device, weights_only=False).to(device).type(torch.float)
        graph_data.x = node_feat
    
    # TODO: check datasets that need shallow embedding
    # Apply Node2Vec for datasets without shallow embeddings
    if emb_model == "shallow" and dataset_name in ["reddit", "instagram", "computer", "photo", "history"]:
        if os.path.exists(f"{path_prefix}/datasets/Node2Vec/{dataset_name}.pt"):
            node_feat = torch.load(f"{path_prefix}/datasets/Node2Vec/{dataset_name}.pt", map_location=device).to(device)
        else:
            from node2vec import Node2Vec
            from torch_geometric.utils.convert import to_networkx
        
            nx_graph = to_networkx(graph_data)
            node2vec = Node2Vec(nx_graph, dimensions=300, walk_length=30, num_walks=10, workers=4)
            node2vec_model = node2vec.fit(window=10, min_count=1, batch_words=4)
            print(node2vec_model.wv.vectors.shape, type(node2vec_model.wv.vectors))
            node_feat = torch.FloatTensor(node2vec_model.wv.vectors).to(device)
            os.makedirs(f"{path_prefix}/datasets/Node2Vec", exist_ok=True)
            torch.save(node_feat, f"{path_prefix}/datasets/Node2Vec/{dataset_name}.pt")
        graph_data.x = node_feat
    
    return graph_data


def load_graph_dataset_for_tape(dataset_name, device, re_split=False, use_gpt=False, gpt_name="Mistral-7B"):
    graph_data = load_graph_dataset(dataset_name, device, re_split, path_prefix="../..") 

    if use_gpt:
        prediction_file = f"../../results/TAPE/{gpt_name}/{dataset_name}.json" 
        if not os.path.exists(prediction_file):
            print("LLM generated explanations do not exist, use original text instead.")
        else:
            file_reader = open(prediction_file, 'r')
            id2exp = defaultdict(str)
            for line in file_reader:
                content = json.loads(line)
                answer = content["answer"]
                # TODO: check whether it needs to replace prediction part
                # if "Explanation:" in content["answer"]:
                #     answer = content["answer"][content["answer"].index("Explanation:"):]
                # else:
                #     answer = content["answer"]
                id2exp[content["idx"]] = answer
            
            raw_texts = [id2exp.get(idx, graph_data.raw_texts[idx]) for idx in range(graph_data.num_nodes)]
            return graph_data, len(graph_data.label_name), raw_texts

    return graph_data, len(graph_data.label_name), graph_data.raw_texts


def load_graph_dataset_for_llaga(dataset_name, device, re_split=False, encoder="roberta"):
    graph_data = load_graph_dataset_for_gnn(dataset_name, device, re_split, path_prefix="../..", emb_model=encoder)

    return graph_data
