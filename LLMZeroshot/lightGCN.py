import torch
import sys
import os
import argparse
from sklearn.neighbors import kneighbors_graph
import csv
sys.path.append("../")
from common import load_graph_dataset, normalize_adj_matrix

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

def save_index(dataset):
    os.makedirs("../datasets/gnn-neighbors", exist_ok=True)
    file_path = f"../datasets/gnn-neighbors/{dataset}.csv"
    common_neighbors = kneighbor_index(dataset)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        for key, value in common_neighbors.items():
            writer.writerow([key, value])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="cora",
                        choices=['cora', "pubmed", "citeseer", "wikics", "arxiv", "instagram", "reddit"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--layer", type=int, default=3)

    args = parser.parse_args()

    # load initial embedding
    data = load_graph_dataset (dataset_name = args.dataset, device=args.device,emb_model="e5-large")
    node_embeds = data.x
    # load adjacent matrix
    adj_normed = normalize_adj_matrix(data.edge_index, data.num_nodes, args.device)

    for _ in range(args.layer):
        node_embeds = torch.mm(adj_normed, node_embeds)

    k_neighbors = kneighbor_index(node_embeds)
    print(k_neighbors)


