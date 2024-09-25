from openai import OpenAI
import os
import argparse
import torch
import csv
import sys
import time
from http import HTTPStatus
import dashscope
from dashscope import Generation
import networkx as nx
from torch_geometric.utils.convert import to_networkx
from sklearn.neighbors import kneighbors_graph
from kneighbor import kneighbor_index, extract_common_values, k_1_neighbor_intersection


sys.path.append("../")
from common import LLM_NEIGHBOR_PROMPTS as PROMPT_DICT
from common import load_graph_dataset, compute_acc_and_f1


def evaluate(file_path, dataset):
    true_labels, predict_labels = [], []

    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if "Accuracy" in str(row[0]):
                continue
            if dataset == "citeseer":
                if (row[2][:2] in ["ML", "IR", "DB", "HC", "AI"]):
                    true_labels.append(row[2][:2])
                else:
                    true_labels.append(row[2])
                if (row[1][:2] in ["ML", "IR", "DB", "HC", "AI"]):
                    predict_labels.append(row[1][:2])
                else:
                    predict_labels.append(row[1])
            else:
                true_labels.append(row[2])
                predict_labels.append(row[1])

    accuracy, f1 = compute_acc_and_f1(predict_labels, true_labels)
    print(f'Accuracy: {accuracy:.3f}, F1 Score: {f1:.3f}')

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([f'Accuracy: {accuracy:.3f}', f'F1 Score: {f1:.3f}'])


