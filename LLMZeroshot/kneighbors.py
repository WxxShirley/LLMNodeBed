import torch
import sys
import os
import argparse
from sklearn.neighbors import kneighbors_graph
import networkx as nx
from torch_geometric.utils.convert import to_networkx
import csv
sys.path.append("../")
from common import load_graph_dataset, load_graph_dataset_for_gnn, normalize_adj_matrix

def kneighbor_index(node_embeds):
    X = node_embeds
    A = kneighbors_graph(X, 7, mode='connectivity', include_self=False).toarray()

    num_node = node_embeds.shape[0]
    k_neighbors = {index: [] for index in range(0, num_node)}

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j]:
                k_neighbors[i].append(j)
    return k_neighbors

def oneneighbor_index(dataset, device):
    
    data = load_graph_dataset (dataset_name = dataset, device=device, path_prefix="..")

    G = to_networkx(data)
    num_node = data.x.shape[0]
    one_neighbors = {index: [] for index in range(0, num_node)}

    for index in range(0, num_node):
        neighbor_list = list(nx.neighbors(G, index))
        one_neighbors[index] = neighbor_list
        one_neighbors[index] = [item for item in one_neighbors[index] if str(item) != str(index)]

    return one_neighbors

def extract_common_values(dict1, dict2):
    common_values = {}
    
    for key in dict1.keys() & dict2.keys():

        common_list = list(set(dict1[key]) & set(dict2[key]))
        if common_list: 
            common_values[key] = common_list
        else:
            common_values[key] = []

    return common_values


def k_1_neighbor_intersection(emb, dataset,device):
    k_neighbors = kneighbor_index(emb)
    one_neighbors = oneneighbor_index(dataset,device)
    common_neighbors = extract_common_values(k_neighbors, one_neighbors)
    return common_neighbors

def save_index(dataset, dataset_emb, device, type):
    if type == "gnn":
        os.makedirs("../datasets/gnn-neighbors", exist_ok=True)
        file_path = f"../datasets/gnn-neighbors/{dataset}.csv"
        neighbors = k_1_neighbor_intersection(dataset_emb, dataset, device)
    elif type == "lm":
        os.makedirs("../datasets/k-1-neighbors", exist_ok=True)
        file_path = f"../datasets/k-1-neighbors/{dataset}.csv"
        neighbors = k_1_neighbor_intersection(dataset_emb, dataset, device)
    else:
        os.makedirs("../datasets/1-neighbors", exist_ok=True)
        file_path = f"../datasets/1-neighbors/{dataset}.csv"
        neighbors = oneneighbor_index(dataset, device)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        for key, value in neighbors.items():
            writer.writerow([key, value])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="cora",
                        choices=['cora', "pubmed", "citeseer", "wikics", "arxiv", "instagram", "reddit", "photo"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--type", type=str, default='lm',
                        choices=['gnn', 'lm', 'none'])

    args = parser.parse_args()
    device = torch.device(args.device)

    # load initial embedding
    if args.type == "none":
        data = load_graph_dataset (dataset_name = args.dataset, device=device,path_prefix="..")
    else:
        data = load_graph_dataset_for_gnn (dataset_name = args.dataset, device=args.device, path_prefix="..", emb_model="e5-large")

    node_embeds = data.x
    # load adjacent matrix
    if args.type == "gnn":
        adj_normed = normalize_adj_matrix(data.edge_index, data.num_nodes, args.device)
        for _ in range(args.layer):
            node_embeds = torch.mm(adj_normed, node_embeds)

    save_index(args.dataset, node_embeds, device, args.type)
