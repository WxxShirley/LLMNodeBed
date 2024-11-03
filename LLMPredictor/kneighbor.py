import os
import torch
from sklearn.neighbors import kneighbors_graph
import sys
import networkx as nx
from torch_geometric.utils.convert import to_networkx
import csv
from torch_geometric.utils import to_undirected

sys.path.append("../")
from common import RAW_NEIGHBOR_PROMPTS as PROMPT_DICT
from common import load_graph_dataset, compute_acc_and_f1

def kneighbor_index(dataset):

    data = load_graph_dataset (dataset_name = dataset, device="cpu",emb_model="e5-large")
    X = data.x
    A = kneighbors_graph(X, 7 , mode='connectivity', include_self=False).toarray()

    num_node = data.x.shape[0]
    k_neighbors = {index: [] for index in range(0, num_node)}

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j]:
                k_neighbors[i].append(j)
    
    return k_neighbors




def oneneighbor_index(dataset):
    
    data = load_graph_dataset (dataset_name = dataset, device="cpu")

    if dataset in ["citeseer", "arxiv"]:
        data.edge_index = to_undirected(data.edge_index)

    G = to_networkx(data)
    num_node = data.x.shape[0]
    one_neighbors = {index: [] for index in range(0, num_node)}

    #max_length = [0,0]

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

def k_1_neighbor_intersection(dataset):
    k_neighbors = kneighbor_index(dataset)
   
    one_neighbors = oneneighbor_index(dataset)

    common_neighbors = extract_common_values(k_neighbors, one_neighbors)


    return common_neighbors


def save_index(dataset):
    os.makedirs("../datasets/1-neighbors", exist_ok=True)
    file_path = f"../datasets/1-neighbors/{dataset}.csv"
    common_neighbors = oneneighbor_index(dataset)

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        for key, value in common_neighbors.items():
            writer.writerow([key, value])
            
        

if __name__ == '__main__':
    dataset = "pubmed"
    # common_neighbors = k_1_neighbor_intersection (dataset)
    # print("\n1")
    # k_neighbors = kneighbor_index(dataset)
    # print("\n2")
    # one_neighbors = oneneighbor_index(dataset)
    # print("\n3")
    # print(common_neighbors)
    save_index(dataset)