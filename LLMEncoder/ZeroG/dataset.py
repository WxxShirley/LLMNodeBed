from torch_geometric.data import Dataset, Data
from torch_geometric.utils import k_hop_subgraph
import numpy as np
from tqdm import tqdm
import math


class KHopSubgraphDataset(Dataset):
    def __init__(self, data, num_hops=1, max_nodes=100, dataset_name="cora", transform=None, pre_transform=None, train_flg=False):
        super(KHopSubgraphDataset, self).__init__(None, transform, pre_transform)

        self.data = data 
        self.num_hops = num_hops
        self.unique_classes = data.y.unique()
        self.k_over_2 = math.ceil(len(self.unique_classes)/2)
        # if true, we only extract subgraphs from training nodes
        self.train_flg = train_flg

        if dataset_name == "citeseer":
            self.k_over_2 = 2
        elif dataset_name == "arxiv":
            self.k_over_2 = 5 

        self.max_nodes = max_nodes
        self.dataset_name = dataset_name
        
        self.subgraphs = self._create_subgraphs()
    
    def _create_subgraphs(self):
        subgraphs = []

        for idx in tqdm(range(self.data.num_nodes)):
            if self.train_flg and not self.data.train_mask[idx]:
                    continue 

            subg_node_idx, subg_edge_index, mapping, _ = k_hop_subgraph(
                node_idx=idx,
                num_hops=self.num_hops,
                edge_index=self.data.edge_index,
                relabel_nodes=True,
                num_nodes=self.data.num_nodes
            )
            if len(subg_node_idx) > self.max_nodes:
                subg_node_idx, subg_edge_index, mapping, _ = k_hop_subgraph(node_idx=idx, num_hops=max(self.num_hops-1, 1), edge_index=self.data.edge_index, relabel_nodes=True, num_nodes=self.data.num_nodes)

            unique_classes_in_subgraph = np.unique(self.data.y[subg_node_idx].cpu().numpy())

            if len(unique_classes_in_subgraph) >= self.k_over_2 and len(subg_node_idx) <= self.max_nodes:
                subg_data = Data(edge_index=subg_edge_index)
                subg_data.num_nodes = len(subg_node_idx.tolist())
                subg_data.y = self.data.y[subg_node_idx]
                subg_data.raw_text = [self.data.raw_texts[i] for i in subg_node_idx.tolist()]
                subg_data.label_text = self.data.label_text 

                subg_data.dataset_name = self.dataset_name
                subgraphs.append(subg_data)
        
        return subgraphs

    def len(self):
        return len(self.subgraphs)
    
    def get(self, idx):
        return self.subgraphs[idx]
    
    def __getitem__(self, idx):
        return self.get(idx)
