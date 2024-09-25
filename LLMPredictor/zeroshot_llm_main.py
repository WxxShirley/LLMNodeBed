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
from get_prediction import prediction
from evaluation import evaluate

sys.path.append("../")
from common import LLM_NEIGHBOR_PROMPTS as PROMPT_DICT
from common import load_graph_dataset, compute_acc_and_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="instagram")
    # chatglm3-6b   deepseek-chat   qwen-turbo
    parser.add_argument("--model_name", type=str, default="chatglm3-6b")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--prediction_type", type=str, default="lm")

    args = parser.parse_args()

    # Get the index set of the training set on given dataset
    device = torch.device(args.device)
    graph_data = load_graph_dataset(args.dataset, device)
    test_indexes = torch.where(graph_data.test_mask == True)[0].cpu().numpy().tolist()
    
    test_indexes = [1,2,3,4]

    # Create csv file

    if args.prediction_type == "none":
        zero_shot_predfolder = "../results/LLMPredictor/llm_zero_shot"
    elif args.prediction_type == "raw":
        zero_shot_predfolder = "../results/LLMPredictor/llm_raw_neighbors"
    elif args.prediction_type == "lm":
        zero_shot_predfolder = "../results/LLMPredictor/llm_lm_neighbors"
    else:
        zero_shot_predfolder = "../results/LLMPredictor/llm_llm_neighbors"

    file_path = f"{zero_shot_predfolder}/{args.model_name}/{args.dataset}.csv"
    os.makedirs(zero_shot_predfolder, exist_ok=True)
    os.makedirs(f"{zero_shot_predfolder}/{args.model_name}", exist_ok=True)

    has_inferenced_index = []
    if os.path.exists(file_path):
        for line in csv.reader(open(file_path, 'r')):
            if len(line) == 3 and (line[0][0] >= '0' and line[0][0] <= '9'):
                has_inferenced_index.append(eval(line[0]))
        print(f"{file_path} already exists with {len(has_inferenced_index)} cases have been inferenced!")

    write_file = open(file_path, 'a', newline='')


    #instance
 
    # Make zero-shot predictions
    for index in test_indexes:
        if index in has_inferenced_index:
            continue

        try:
            make_prediction = prediction(args.prediction_type, args.dataset, args.model_name, index, file_path, graph_data)
            make_prediction.write_in_file()
        except Exception as e:
            # time.sleep(1)
            print(f"[ERROR] {index} encounter error {e}")

    # Calculate acc and f1
    evaluate(file_path, args.dataset)
