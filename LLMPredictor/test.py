import torch 
import prettytable as pt
from torch_geometric.loader import NeighborLoader
from torch_geometric.utils import to_networkx
import networkx as nx
from torch_geometric.utils.convert import to_networkx
import sys

sys.path.append("../")
from common import RAW_NEIGHBOR_PROMPTS as PROMPT_DICT
from common import load_graph_dataset, compute_acc_and_f1

datanames = ['cora', "citeseer", "instagram"]
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
    index=999
    G = to_networkx(data)
    neighbor_list = list(nx.neighbors(G, index))
    neighbor_list = neighbor_list[:10] if len(neighbor_list) > 10 else neighbor_list
    print(neighbor_list)
    # for FNs in list(nx.neighbors(G, index)):  # find 1_th neighbors
    #     nei1_li.append(FNs)
    neighbor_info_list=[]
    for iter in neighbor_list:
        neighbor_info_list.append(data.raw_texts[iter])
    neighbor_info = "\none of its neighbors' feature:".join(neighbor_info_list)
    print(neighbor_info)

# print("calculate average degree")
# data = torch.load(f"../datasets/instagram.pt").to(device)
# G=to_networkx(data)
# total_degree=0
# number_of_nodes=0

# for index in range(1, 50):
#     if data.label_name[data.y[index]] in "Commercial Users":
#         number_of_nodes=number_of_nodes+1
#         total_degree=total_degree+G.degree()[index]
#         print(data.raw_texts[index],"\n")
# d= dict(nx.degree(G))
# print("平均度为：", sum(d.values())/len(G.nodes))
