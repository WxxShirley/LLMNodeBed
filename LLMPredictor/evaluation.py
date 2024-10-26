from openai import OpenAI
import os
import argparse
import torch
import csv
import sys
from http import HTTPStatus
from dashscope import Generation
import re


sys.path.append("../")
from common import LLM_NEIGHBOR_PROMPTS as PROMPT_DICT
from common import load_graph_dataset, compute_acc_and_f1

labels = {
    "cora": [
        'Rule_Learning', 'Neural_Networks', 'Case_Based', 'Genetic_Algorithms', 'Theory', 'Reinforcement_Learning', 'Probabilistic_Methods'
    ],
    "pubmed": [
        'Experimental', 'Diabetes Mellitus Type 1', 'Diabetes Mellitus Type 2' 
    ],
    "citeseer": [
        'Agents', 'ML (Machine Learning)', 'IR (Information Retrieval)', 'DB (Databases)', 'HCI (Human-Computer Interaction)', 'AI (Artificial Intelligence)'
    ],
    "wikics": [
       'Computational Linguistics', 'Databases', 'Operating Systems', 'Computer Architecture', 'Computer Security', 'Internet Protocols', 'Computer File Systems', 'Distributed Computing Architecture', 'Web Technology', 'Programming Language Topics'
    ],
    "instagram": [
       'Normal Users', 'Commercial Users'
    ],
    "reddit": [
        'Normal Users', 'Popular Users'
    ]
}

def check_correct(dataset, row):
    true_labels= []
    true_labels.append(row[2])

    false_labels = labels[dataset].copy()
    false_labels.remove(row[2])

    if dataset == "cora":
        # '_' ->' '
        true_labels.append(row[2].replace('_', ' '))
        new_labels = [item.replace('_', ' ') for item in false_labels]
        false_labels.extend(new_labels)

        # lower case
        true_labels.append(row[2].lower())
        true_labels.append(true_labels[1].lower())
        new_labels = [item.lower() for item in false_labels]
        false_labels.extend(new_labels)

    elif dataset == "citeseer":
        # content inside ()
        if row[2] != "Agents":
            true_labels.append(row[2].split('(')[1].replace(')',''))
        
        new_labels = false_labels.copy()
        for item in new_labels:
            if item != "Agents":
                false_labels.append(item.split('(')[1].replace(')',''))     

        # acronym
        true_labels.append(row[2].split(' (')[0])
        new_labels = [item.split(' (')[0] for item in false_labels]
        false_labels.extend(new_labels)
    
    elif dataset == "reddit" or dataset == "instagram":
        # delete "s"
        true_labels.append(row[2][:-1])
        new_labels = [item[:-1] for item in false_labels]
        false_labels.extend(new_labels)

    true_labels_in_completion = [x in row[1] for x in true_labels]
    false_label_in_completion = [x in row[1] for x in false_labels]

    if any(true_labels_in_completion) and not any(false_label_in_completion):
        return "correct"
    elif any(true_labels_in_completion) and any(false_label_in_completion):
        return "uncertain"
    elif any(false_label_in_completion):
        return "wrong"
    else:
        return "hallucination"






def write_correctness(file_path):

    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        rows = list(reader)
    
    for row in rows:
        if row[0][0]< '0' or row[0][0] > '9':
            continue
        check_result = check_correct(args.dataset, row)
        row.append(check_result)
        #row[3] = check_result

    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)







def evaluate(file_path, model_name, dataset):
    #write_correctness(file_path)
    true_labels, predict_labels = [], []
    total_num = 0
    hallucination = 0

    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0][0]< '0' or row[0][0] > '9':
                continue
            
            total_num += 1

            if row[3] == "correct":
                true_labels.append(row[2])
                predict_labels.append(row[2])

            elif row[3] == "wrong":
                i=0
                found = 0
                true_labels.append(row[2])
                for item in labels[dataset]:

                    if dataset == "cora":
                        if item.replace('_', ' ') in row[1] or item.lower() in row[1] or item in row[1]:
                            predict_labels.append(item)

                    elif dataset == "citeseer":
                        i += 1
                        
                        if item == "Agents":
                            if item in row[1]:
                                predict_labels.append(item)
                                found = 1
                        else:
                            if item.split('(')[1].replace(')','') in row[1] or item.split(' (')[0] in row[1] or item in row[1]:
                                predict_labels.append(item)
                                found = 1
                    
                    elif dataset == "reddit" or dataset == "instagram":
                        if item[:-1] in row[1] or item in row[1]:
                            predict_labels.append(item)
                    
                    else:
                        if item in row[1]:
                            predict_labels.append(item)
            
            elif row[3] == "uncertain":
                predict_labels.append(row[1])
                true_labels.append(row[2])

            else:
                hallucination += 1
                predict_labels.append(row[1])
                true_labels.append(row[2])

    hullucination_rate = hallucination/total_num
    accuracy, f1 = compute_acc_and_f1(predict_labels, true_labels)
    print(f'Accuracy: {accuracy:.3f}, F1 Score: {f1:.3f}, Hullucination Rate: {hullucination_rate:.3f}')

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([f'Accuracy: {accuracy:.3f}', f'F1 Score: {f1:.3f}', f'Hullucination Rate: {hullucination_rate:.3f}'])


