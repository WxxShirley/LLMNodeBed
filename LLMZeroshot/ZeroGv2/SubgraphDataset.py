from torch_geometric.data import Dataset, Data
from torch_geometric.utils import to_dense_adj, k_hop_subgraph
import numpy as np
from tqdm import tqdm
import math

class kHopSubgraphDataset(Dataset):
    def __init__(self, data, num_hops=1, max_nodes=100, dataset_name="cora", transform=None, pre_transform=None):
        super(kHopSubgraphDataset, self).__init__(None, transform, pre_transform)
        self.data = data
        self.num_hops = num_hops
        self.unique_classes = data.y.unique()
        self.k_over_2 = math.ceil(len(self.unique_classes)/2)
        if dataset_name == "citeseer":
            self.k_over_2 = 2
        elif dataset_name == "arxiv":
            self.k_over_2 = 10
        
        
        self.max_nodes = max_nodes
        self.dataset_name = dataset_name

        self.subgraphs = self._create_subgraphs()

    def _create_subgraphs(self):
        subgraphs = []
        for idx in tqdm(range(self.data.num_nodes)):
            
            subgraph_node_idx, subgraph_edge_index, mapping, edge_mask = k_hop_subgraph(
                node_idx=idx,
                num_hops=self.num_hops,
                edge_index=self.data.edge_index,
                relabel_nodes=True,
                num_nodes=self.data.num_nodes 
            )
            
            unique_classes_in_subgraph = np.unique(self.data.y[subgraph_node_idx].cpu().numpy())
            if len(unique_classes_in_subgraph) >= self.k_over_2 and len(subgraph_node_idx) <= self.max_nodes:
                sub_data = Data(edge_index=subgraph_edge_index)
                sub_data.y = self.data.y[subgraph_node_idx]
                sub_data.raw_text = [self.data.raw_texts[i] for i in subgraph_node_idx.tolist()]
                sub_data.label_text = self.data.label_text
                sub_data.adjacency_matrix = to_dense_adj(subgraph_edge_index, max_num_nodes=mapping.size(0))[0]
                sub_data.dataset_name = self.dataset_name
                subgraphs.append(sub_data)
        return subgraphs

    def len(self):
        return len(self.subgraphs)

    def get(self, idx):
        return self.subgraphs[idx]

    def __getitem__(self, idx):
        return self.get(idx)
