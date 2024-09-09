"""demo for observing graph data"""
import torch 
import prettytable as pt


datanames = ['cora', "pubmed", "citeseer", "wikics", "arxiv", "instagram", "reddit"]
device = 'cuda:0'
table = pt.PrettyTable()
table.field_names = ["Dataset", "# Nodes", "# Links", "# Class", "# Feature"]

for dataname in datanames:
    data = torch.load(f"../datasets/{dataname}.pt").to(device)
    # print(data, data.label_name)
    # each data is an instance of PyG.Data (https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.data.Data.html#torch_geometric.data.Data)
    num_node, num_feature = data.x.shape 
    num_link = data.edge_index.shape[1]
    num_class = len(data.label_name)
    table.add_row([dataname, num_node, num_link, num_class, num_feature])

print(table)
