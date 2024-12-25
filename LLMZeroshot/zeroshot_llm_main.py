import os
import argparse
import torch
import csv
import sys
from get_prediction import prediction
from evaluation import evaluate
import numpy as np

sys.path.append("../")
from common import load_graph_dataset, compute_acc_and_f1, set_seed

def sample_test_indexes(test_indexes, labels, sample_ratio):
    # Count the number of categories
    label_counts = {}
    for idx in test_indexes:
        label = labels[idx].item()
        if label not in label_counts:
            label_counts[label] = []
        label_counts[label].append(idx)

    # 进行按比例采样
    sampled_indexes = []
    for label, indexes in label_counts.items():
        # 计算采样数量
        sample_count = max(1, int(len(indexes) * sample_ratio))
        print(sample_count,"\n")
        # 随机采样指定数量的索引
        sampled_indexes.extend(np.random.choice(indexes, sample_count, replace=False))

    print(len(sampled_indexes),"\n",sampled_indexes)
    return sampled_indexes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="cora")
    # chatglm3-6b   deepseek-chat   qwen-turbo  gpt-4  gpt-4o
    parser.add_argument("--model_name", type=str, default="gpt-4o")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--prediction_type", type=str, default="gnn")

    args = parser.parse_args()

    # Get the index set of the training set on given dataset
    device = torch.device(args.device)
    graph_data = load_graph_dataset(args.dataset, device, True, "..")
    test_indexes = torch.where(graph_data.test_mask == True)[0].cpu().numpy().tolist()
    labels = graph_data.y.cpu().numpy()

    # Create csv file
    if args.prediction_type == "none":
        zero_shot_predfolder = "../results/zero_shot/none"
    elif args.prediction_type == "cot":
        zero_shot_predfolder = "../results/zero_shot/cot"
    elif args.prediction_type == "tot":
        zero_shot_predfolder = "../results/zero_shot/tot"
    elif args.prediction_type == "react":
        zero_shot_predfolder = "../results/zero_shot/react"
    elif args.prediction_type == "lm":
        zero_shot_predfolder = "../results/zero_shot/lm"
    elif args.prediction_type == "gnn":
        zero_shot_predfolder = "../results/zero_shot/gnn"
    else:
        zero_shot_predfolder = "../results/zero_shot/llm"

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

    # test_indexes = [112,220,2230]
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
    evaluate(file_path, args.model_name, args.dataset)
