import os
import argparse
import torch
import csv
import sys
from model import LLMDirectInference
from evaluation import simiple_evaluate
import time
sys.path.append("../")
from common import load_graph_dataset, set_seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--model_name", type=str, default="deepseek-chat")
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--prompt_type", type=str, default="direct", choices=['direct', 'cot', 'tot', 'react', 'neighbor', 'summary'])
    parser.add_argument("--maximum_neighbor", type=int, default=10)
    parser.add_argument("--set_seed", type=int, default=0)

    args = parser.parse_args()
    seed = set_seed(args.set_seed)

    device = torch.device(args.device)
    graph_data = load_graph_dataset(args.dataset, device, re_split=True)
    test_indexes = torch.where(graph_data.test_mask == True)[0].cpu().numpy().tolist()
    labels = graph_data.y.cpu().numpy()

    # Create csv file
    zero_shot_predfolder = f"../../results/zero_shot/{args.prompt_type}"
    file_path = f"{zero_shot_predfolder}/{args.model_name}/{args.dataset}.csv"
    os.makedirs(f"{zero_shot_predfolder}/{args.model_name}", exist_ok=True)

    has_inferenced_index = []
    if os.path.exists(file_path):
        for line in csv.reader(open(file_path, 'r')):
            if len(line) >= 2 and (line[0][0] >= '0' and line[0][0] <= '9'):
                has_inferenced_index.append(eval(line[0]))
        print(f"{file_path} already exists with {len(has_inferenced_index)} cases have been inferenced!")

    # Create Predictor and do the inference job 
    pred_obj = LLMDirectInference(dataset_name=args.dataset, 
                                  graph_data=graph_data, 
                                  prompt_type=args.prompt_type, 
                                  llm_name=args.model_name, 
                                  write_file_path=file_path,
                                  maximum_neighbor=args.maximum_neighbor)
    pred_obj.do_prediction(test_indexes, has_inferenced_index)

    # Calculate acc and f1
    acc, f1, hall = simiple_evaluate(file_path, pred_obj.label_space)
    print(f"{args.model_name} on {args.dataset} with prompt template {args.prompt_type}")
    print(f"Accuracy {acc}  Macro-F1 {f1}  Hallucination {hall}")
