import torch 
import prettytable as pt
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_networkx
import networkx as nx
from torch_geometric.utils.convert import to_networkx


datanames = ['cora', "citeseer", "instagram","reddit"]
device = 'cpu'
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


for dataname in datanames:
    data = torch.load(f"../datasets/{dataname}.pt").to(device)
    print(data.is_directed())
    index=4
    G = to_networkx(data)
    nei1_li = []
    for FNs in list(nx.neighbors(G, index)):  # find 1_th neighbors
        nei1_li.append(FNs)
    
    if dataname in "instagram":
        print(nei1_li[1:])
    else:
        print(nei1_li)
    print(G.degree()[index])

print("calculate average degree")
data = torch.load(f"../datasets/instagram.pt").to(device)
G=to_networkx(data)
total_degree=0
number_of_nodes=0

for index in range(1, 50):
    if data.label_name[data.y[index]] in "Normal Users":
        number_of_nodes=number_of_nodes+1
        total_degree=total_degree+G.degree()[index]
        print(data.raw_texts[index],"\n")
d= dict(nx.degree(G))
print("平均度为：", sum(d.values())/len(G.nodes))
