import os
import torch
from sklearn.neighbors import kneighbors_graph
import sys
import networkx as nx
from torch_geometric.utils.convert import to_networkx

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


def extract_common_values(dict1, dict2):
    common_values = {}
    
    for key in dict1.keys() & dict2.keys():

        common_list = list(set(dict1[key]) & set(dict2[key]))
        if common_list: 
            common_values[key] = common_list

    return common_values


def k_1_neighbor_intersection(dataset):
    k_neighbors = kneighbor_index(dataset)
    data = load_graph_dataset (dataset_name = dataset, device="cpu")

    G = to_networkx(data)
    num_node = data.x.shape[0]
    one_neighbors = {index: [] for index in range(0, num_node)}

    #max_length = [0,0]

    for index in range(0, num_node):
        neighbor_list = list(nx.neighbors(G, index))
        one_neighbors[index] = neighbor_list
    #     max_length[0] = index if len(one_neighbors[index])>max_length[1] else max_length[0]
    #     max_length[1] = len(one_neighbors[index]) if len(one_neighbors[index])>max_length[1] else max_length[1]
    
    # print (max_length)
    common_neighbors = extract_common_values(k_neighbors, one_neighbors)
    
    return common_neighbors


if __name__ == '__main__':
    common_neighbors = k_1_neighbor_intersection ("cora")
    print(common_neighbors)