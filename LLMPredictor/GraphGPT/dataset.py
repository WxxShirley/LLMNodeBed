from torch.utils.data import Dataset
import numpy as np
import random
import torch
import sys 
from torch_geometric.utils import k_hop_subgraph
sys.path.append("../..")
from common import prepare_edge_list
from common import CLASSES as classes, MATCHING_TEMPLATES, GraphGPT_DESC as CLASSIFICATION_TEMPLATES


class TextGraphGroundDataset(Dataset):
    def __init__(self, graph_data, num_sampled_neighbors):
        self.graph_data = graph_data 
        self.num_nodes = graph_data.num_nodes 
        
        self.edge_dict = prepare_edge_list(graph_data.edge_index.detach().cpu(), self.num_nodes)
        self.num_sampled_neighbors = num_sampled_neighbors
        
    def __len__(self):
        return self.num_nodes
    
    def __getitem__(self, idx):
        # TODO: fix error in citeseer
        sampled_neighbors = [np.random.choice(self.edge_dict[idx], replace=True) for _ in range(self.num_sampled_neighbors)]
        neighbor_texts = [self.graph_data.raw_texts[neigh_id] for neigh_id in sampled_neighbors]
        return {
            "id": idx, 
            "text": self.graph_data.raw_texts[idx],
            "neighbor_text": neighbor_texts, 
            "neighbor_ids": np.array(sampled_neighbors)
        }


def fetch_title(txt, max_length=512):
    title = None
    if ":" in txt:
        title= txt.split(":")[0]
    title= txt.split(".")[0]
    
    return title[:max_length]


# Example Data: https://huggingface.co/datasets/Jiabin99/graph-matching
class GraphMatchingDataset(Dataset):
    def __init__(self, graph_data, k_hop=1, num_sampled_neighbors=8, graph_type="academic_network", sample_times=1):
        self.graph_data = graph_data 
        self.num_nodes = graph_data.num_nodes 
        self.k_hop = k_hop
        self.num_sampled_neighbors = num_sampled_neighbors
        self.sample_times = sample_times
        self.query_template = MATCHING_TEMPLATES[graph_type]
        self.graph_type = graph_type
        
        self.all_data = self._prepare_matching_data()
    
    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, idx):
        return self.all_data[idx]
        
    def _prepare_matching_data(self):
        data_samples = []
        for node in range(self.num_nodes):
            neighbors, _, _, _ = k_hop_subgraph(node, num_hops=self.k_hop, edge_index=self.graph_data.edge_index)
            neighbors = neighbors.numpy()
            if len(neighbors.tolist()) == 0:
                continue
            
            for _ in range(self.sample_times):
                subset = np.random.choice(neighbors, size=self.num_sampled_neighbors).tolist()
                subset = list(set(subset))
                if node not in subset:
                    subset = [node] + subset[:-1] 
                else:
                    target_idx = subset.index(node)
                    subset[target_idx] = subset[0]
                    subset[0] = node
                
                if len(subset) < self.num_sampled_neighbors:
                    pad_length = self.num_sampled_neighbors - len(subset)
                    subset = subset + [node] * pad_length
                    
                assert subset[0] == node 
                
                texts = []
                for token_id in range(len(subset)): 
                    raw_text = fetch_title(self.graph_data.raw_texts[subset[token_id]])
                    texts.append([token_id, raw_text]) # origin_id in graph-tokens, corresponding text
                
                # Re-order the texts 
                random.shuffle(texts)
                tokenid2text_mapping = {pairs[0]+1: pairs[1] for text_id, pairs in enumerate(texts)}
                query_graph_texts = ". ".join([f"{text_id+1}. {pairs[1]}" for text_id, pairs in enumerate(texts)])
                
                if self.graph_type == "academic_network":
                    cur_query = self.query_template.replace("{{paper_titles}}", query_graph_texts)
                    cur_response = ". ".join([f"Graph token {k} corresponds to paper {tokenid2text_mapping[k]}" for k in sorted(tokenid2text_mapping.keys()) ])
                    cur_response = "Based on the given graph tokens and the list of paper titles, we obtain the matching of graph tokens and papers as follows: " + cur_response
                elif self.graph_type == "social_network":
                    cur_query = self.query_template.replace("{{user_profiles}}", query_graph_texts)
                    cur_response = ". ".join([f"Graph token {k} corresponds to user {tokenid2text_mapping[k]}" for k in sorted(tokenid2text_mapping.keys()) ])
                    cur_response = "Based on the given graph tokens and the descriptions of users, we obtain the matching of graph tokens and users as follows: " + cur_response
                elif self.graph_type == "ecommerce_network":
                    cur_query = self.query_template.replace("{{item_comments}}", query_graph_texts)
                    cur_response = ". ".join([f"Graph token {k} corresponds to item {tokenid2text_mapping[k]}" for k in sorted(tokenid2text_mapping.keys())])
                    cur_response = "Based on the given graph tokens and the comments of items, we obtain the matching of graph tokens and items as follows: " + cur_response

                sample = {
                    "id": node,
                    "nodes": torch.LongTensor(subset),
                    "query": cur_query,
                    "label": cur_response
                }
                data_samples.append(sample)
                
        return data_samples


# Example Data: https://huggingface.co/datasets/Jiabin99/Arxiv-PubMed-mix-NC-LP
class GraphInstructionTuningDataset(Dataset):
    def __init__(self, graph_data, k_hop=1, maximum_neighbors=4, dataset_name="cora", data_type="train"):
        self.graph_data = graph_data
        self.num_nodes = graph_data.num_nodes
        self.k_hop = k_hop 
        self.maximum_neighbors = maximum_neighbors
        self.label_names = classes[dataset_name]
        self.data_type = data_type
        
        label_names = ", ".join(classes[dataset_name])
        self.query_prompt = CLASSIFICATION_TEMPLATES[dataset_name].replace("{{label_names}}", label_names)
        self.data_list = self.format_data()
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        return self.data_list[index]
    
    def format_data(self):
        focus_mask = {"train": self.graph_data.train_mask, "val": self.graph_data.val_mask, "test": self.graph_data.test_mask}[self.data_type]
        focus_nodes = focus_mask.nonzero(as_tuple=False).squeeze().detach().cpu().numpy().tolist()
        
        available_data_list = []
        for cur_node in focus_nodes:
            neighbors, _, _, _ = k_hop_subgraph(cur_node, num_hops=self.k_hop, edge_index=self.graph_data.edge_index)
            neighbors = neighbors.numpy().tolist()
        
            if len(neighbors) > self.maximum_neighbors:
                neighbors = np.random.choice(np.array(neighbors), size=self.maximum_neighbors).tolist()
                neighbors = [cur_node] + neighbors
            else: 
                pad_length = self.maximum_neighbors - len(neighbors) 
                neighbors = [cur_node] + neighbors + [cur_node] * pad_length
            
            assert cur_node == neighbors[0]
        
            cur_query = self.query_prompt.replace("{{raw_text}}", self.graph_data.raw_texts[cur_node])
            cur_response = self.label_names[self.graph_data.y[cur_node].item()]
        
            available_data_list.append({
                "id": cur_node, 
                "nodes": torch.LongTensor(neighbors),
                "query": cur_query,
                "label": cur_response
            })  
        
        return available_data_list   
