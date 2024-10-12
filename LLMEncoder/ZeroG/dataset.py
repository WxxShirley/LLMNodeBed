from torch_geometric.data import Data
import torch 


class MyTextDataset(torch.utils.data.Dataset):
    def __init__(self, graph_data: Data):
        self.graph = graph_data
        self.train_pairs = self._prepare_training_data()
    
    def _prepare_training_data(self):
        trainsets = []
        for node_idx in range(self.graph.num_nodes):
            if self.graph.train_mask[node_idx]:
                trainsets.append([self.graph.raw_texts[node_idx], self.graph.y[node_idx]])
        return trainsets

    def __getitem__(self, idx):
        train_sample = self.train_pairs[idx]
        item = {
            "text": train_sample[0],
            "label": train_sample[1]
        }
        return item

    def __len__(self):
        return len(self.train_pairs)
