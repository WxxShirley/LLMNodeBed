import csv
from sklearn.metrics import accuracy_score, f1_score
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import ast
import sys
sys.path.append("../")
from common import LLM_NEIGHBOR_PROMPTS as PROMPT_DICT
from common import load_graph_dataset, compute_acc_and_f1

def compute_acc_and_f1(pred, ground_truth):
    accuracy = accuracy_score(ground_truth, pred) * 100.0
    weighted_f1 = f1_score(ground_truth, pred, average="weighted") * 100.0
    macro_f1 = f1_score(ground_truth, pred, average="macro") * 100.0
    
    return round(accuracy, 3), round(weighted_f1, 3), round(macro_f1, 3)


def acc_two_kinds_f1(file_path, dataset):
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
            elif dataset == "cora" and model_name == "chatglm3-6b":
                true_labels.append(row[2])
                if "Rule" in row[1]:
                    predict_labels.append("Rule_Learning")
                elif "Neural" in row[1]:
                    predict_labels.append("Neural_Networks")
                elif "Case" in row[1]:
                    predict_labels.append("Case_Based")
                elif "Genetic" in row[1]:
                    predict_labels.append("Genetic_Algorithms")
                elif "Theory" in row[1]:
                    predict_labels.append("Theory")
                elif "Reinforcement" in row[1]:
                    predict_labels.append("Reinforcement_Learning")
                elif "Probabilistic" in row[1]:
                    predict_labels.append("Probabilistic_Methods")
                else:
                    predict_labels.append(row[1])
            else:
                true_labels.append(row[2])
                predict_labels.append(row[1])

    accuracy, weighted_f1, macro_f1 = compute_acc_and_f1(predict_labels, true_labels)
    print(f'Accuracy: {accuracy:.3f}\nWeighted F1 Score: {weighted_f1:.3f}\nMacro F1 Score:{macro_f1:.3f}')


    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([f'Accuracy: {accuracy:.3f}, Weighted F1 Score: {weighted_f1:.3f}, Macro F1 Score:{macro_f1:.3f}'])
    
    return true_labels, predict_labels



def bias_of_category(dataset, prediction_type, model_name, file_path):
    true_labels, predict_labels = acc_two_kinds_f1(file_path, dataset)

    if dataset == "cora":
        labels = ["Rule_Learning", "Neural_Networks", "Case_Based", "Genetic_Algorithms", "Theory", "Reinforcement_Learning", "Probabilistic_Methods", "Others"]
    elif dataset == "citeseer":
        labels = ["Agents", "ML (Machine Learning)", "IR (Information Retrieval)", "DB (Databases)", "HCI (Human-Computer Interaction)", "AI (Artificial Intelligence)", "Others"]
    else:
        labels = ["Normal Users", "Commercial Users", "Others"]

    true_labels_counting = [0] * len(labels)
    predict_labels_counting = [0] * len(labels)


    for index in range(len(predict_labels)):
        add_true = -1
        add_predict = -1
        for i in range(len(labels)):
            if predict_labels[index][:2] == labels[i][:2]:
                add_predict = i
            if true_labels[index][:2] == labels[i][:2]:
                add_true = i
        true_labels_counting[add_true] += 1
        predict_labels_counting[add_predict] += 1
                

    y = np.arange(len(labels))
    height = 0.35
    fig, ax = plt.subplots(figsize=(20, 6))
    
    # Create horizontal bars
    rects1 = ax.barh(y - height/2, true_labels_counting, height, label='True-Label')
    rects2 = ax.barh(y + height/2, predict_labels_counting, height, label='Predict-Label')

    ax.set_xlabel('Number of labels')
    ax.set_title(f'{prediction_type} + {model_name} + {dataset}')
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.legend()

    # Adding labels on the bars
    for rect in rects1:
        width = rect.get_width()
        ax.text(width, rect.get_y() + rect.get_height() / 2, str(width), ha='left', va='center')

    for rect in rects2:
        width = rect.get_width()
        ax.text(width, rect.get_y() + rect.get_height() / 2, str(width), ha='left', va='center')

    plt.show()
    plt.savefig('./2.jpg')


def homophily_ratio(dataset):
    graph_data = load_graph_dataset(dataset,"cpu","shallow")
    file_path = f'../datasets/1-neighbors/{dataset}.csv'

    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        one_neighbor_list = []
        row_num = 0
        all_average = 0
        for row in reader:
            central_node = int(row[0])
            one_neighbor_list = ast.literal_eval(row[1])
            list_lenth = len(one_neighbor_list)
            if list_lenth == 0:
                continue
            
            row_num = row_num + 1
            same_category = 0
            for a_neighbor in one_neighbor_list:
                if graph_data.y[central_node] == graph_data.y[a_neighbor]:
                    same_category = same_category + 1
            if dataset == "instagram":
                same_category = same_category - 1
            row_average = float(same_category)/list_lenth
            all_average += row_average
    
    print(f'homophily: {float(all_average)/row_num}')
            
                

                






if __name__ == "__main__":
    dataset = "citeseer"
    prediction_type = "llm"
    model_name = "deepseek-chat"


    if prediction_type == "none":
        zero_shot_predfolder = "../results/LLMPredictor/llm_zero_shot"
    elif prediction_type == "raw":
        zero_shot_predfolder = "../results/LLMPredictor/llm_raw_neighbors"
    elif prediction_type == "lm":
        zero_shot_predfolder = "../results/LLMPredictor/llm_lm_neighbors"
    else:
        zero_shot_predfolder = "../results/LLMPredictor/llm_llm_neighbors"

    file_path = f"{zero_shot_predfolder}/{model_name}/{dataset}.csv"

    acc_two_kinds_f1(file_path,dataset)
    bias_of_category(dataset,prediction_type,model_name,file_path)
    homophily_ratio(dataset)