from openai import OpenAI
import os
import argparse
import torch
import csv
import sys
import time
from http import HTTPStatus
from dashscope import Generation
import dashscope

sys.path.append("../")
from common import ZEROSHOT_PROMPTS as PROMPT_DICT
from common import load_graph_dataset, compute_acc_and_f1


def get_response(dataset, model_name, index, write_file_path):
    discription = graph_data.raw_texts[index]
    question = PROMPT_DICT[dataset]
    
    if model_name == "chatglm3-6b":
        dashscope.api_key = "sk-6ed3b105aaac459097168fd8cca58513"
        messages=[
            {
                'role': 'user',
                'content': f"{discription}\n{question}"
            }]
    
        gen = Generation()
        response = gen.call(
            'chatglm3-6b',
            messages=messages,
            result_format='message',  # set the result is message format.
        )
        #prediction = response["output"]["choices"][0]["message"]["content"].replace('\n', '').strip()
        prediction = response.split(", ")[-1].strip("'")

    elif model_name == "deepseek-chat":
        client = OpenAI(
            api_key="sk-a8a2461880014e87bc10ca6c9c62bfdc", 
            base_url="https://api.deepseek.com")

        response = client.chat.completions.create(
            model = model_name,
            messages=[
                {
                'role': 'user',
                'content': f"{discription}\n{question}"
                }
            ],
            stream=False
        )
        prediction = response.choices[0].message.content

    else:
        client = OpenAI(
            # Available keys
            # sk-0b8693b926d24c299458c8b607f8c46f (xixi)
            # sk-b8659491d3fc429b982b9a04f9fd55c0 (xixi)
            # sk-946342daa7234baeb39287866be76505 (xixi)
            # sk-6ed3b105aaac459097168fd8cca58513 (fangzhou)
            api_key="sk-946342daa7234baeb39287866be76505",
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )


        completion = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    'role': 'user',
                    'content': f"{discription}\n{question}"
                }
            ]
        )

        prediction = completion.choices[0].message.content


    true_label = graph_data.label_name[graph_data.y[index]]

    with open(write_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        if dataset == "arxiv":
            writer.writerow([index, prediction[:11], true_label])
        else:
            writer.writerow([index, prediction, true_label])
        print([index, prediction, true_label])



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
            
            if dataset == "cora":
                true_labels.append(row[2])
                if "Rule_Learning" in row[1]:
                    predict_labels.append("Rule_Learning")
                elif "Neural_Networks" in row[1]:
                    predict_labels.append("Neural_Networks")
                elif "Case_Based" in row[1]:
                    predict_labels.append("Case_Based")
                elif "Genetic_Algorithms" in row[1]:
                    predict_labels.append("Genetic_Algorithms")
                elif "Theory" in row[1]:
                    predict_labels.append("Theory")
                elif "Reinforcement_Learning" in row[1]:
                    predict_labels.append("Reinforcement_Learning")
                elif "Probabilistic_Methods" in row[1]:
                    predict_labels.append("Probabilistic_Methods")
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="instagram")
    parser.add_argument("--model_name", type=str, default="deepseek-chat")
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    # Get the index set of the training set on given dataset
    device = torch.device(args.device)
    graph_data = load_graph_dataset(args.dataset, device)
    test_indexes = torch.where(graph_data.test_mask == True)[0].cpu().numpy().tolist()

    test_indexes = [1,2,3]

    # Create csv file
    zero_shot_predfolder = "../results/LLMPredictor/llm_zero_shot"
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

    # Make zero-shot predictions
    for index in test_indexes:
        if index in has_inferenced_index:
            continue

        try:
            get_response(args.dataset, args.model_name, index, file_path)
        except Exception as e:
            # time.sleep(1)
            print(f"[ERROR] {index} encounter error {e}")

    # Calculate acc and f1
    evaluate(file_path, args.dataset)
