import os
import torch
from sklearn.neighbors import kneighbors_graph
import sys

sys.path.append("../")
from common import RAW_NEIGHBOR_PROMPTS as PROMPT_DICT
from common import load_graph_dataset, compute_acc_and_f1

def kneighbor_index (dataset):

    data = load_graph_dataset (dataset,device="cpu",emb_model="e5-large")
    X = data.x
    A = kneighbors_graph(X, 5 , mode='connectivity', include_self=False).toarray()

    num_node = data.x.shape[0]
    k_neighbors = {index: [] for index in range(1, num_node)}

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            if A[i, j]:
                k_neighbors[i+1].append(j+1)