from torch.utils.data import Dataset
import numpy as np
import sys 
sys.path.append("../..")
from common import prepare_edge_list


class TextGraphGroundDataset(Dataset):
    def __init__(self, graph_data, num_sampled_neighbors):
        self.graph_data = graph_data 
        self.num_nodes = graph_data.num_nodes 
        
        self.edge_dict = prepare_edge_list(graph_data.edge_index.detach().cpu(), self.num_nodes)
        self.num_sampled_neighbors = num_sampled_neighbors
        
    def __len__(self):
        return self.num_nodes
    
    def __getitem__(self, idx):
        sampled_neighbors = [np.random.choice(self.edge_dict[idx], replace=True) for _ in range(self.num_sampled_neighbors)]
        neighbor_texts = [self.graph_data.raw_texts[neigh_id] for neigh_id in sampled_neighbors]
        return {
            "id": idx, 
            "text": self.graph_data.raw_texts[idx],
            "neighbor_text": neighbor_texts, 
            "neighbor_ids": np.array(sampled_neighbors)
        }
