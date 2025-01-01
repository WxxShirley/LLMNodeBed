import csv
import sys

sys.path.append("../")
from common import load_graph_dataset, compute_acc_and_f1

labels = {
    "cora": [
        'Rule_Learning', 'Neural_Networks', 'Case_Based', 'Genetic_Algorithms', 'Theory', 'Reinforcement_Learning',
        'Probabilistic_Methods'
    ],
    "pubmed": [
        'Experimental', 'Diabetes Mellitus Type 1', 'Diabetes Mellitus Type 2'
    ],
    "citeseer": [
        'Agents', 'ML (Machine Learning)', 'IR (Information Retrieval)', 'DB (Databases)',
        'HCI (Human-Computer Interaction)', 'AI (Artificial Intelligence)'
    ],
    "wikics": [
        'Computational Linguistics', 'Databases', 'Operating Systems', 'Computer Architecture', 'Computer Security',
        'Internet Protocols', 'Computer File Systems', 'Distributed Computing Architecture', 'Web Technology',
        'Programming Language Topics'
    ],
    "instagram": [
        'Normal Users', 'Commercial Users'
    ],
    "reddit": [
        'Normal Users', 'Popular Users'
    ],
    "photo": [
        "Video Surveillance", "Accessories", "Binoculars & Scopes", "Video", "Lighting & Studio", "Bags & Cases",
        "Tripods & Monopods", "Flashes", "Digital Cameras", "Film Photography", "Lenses", "Underwater Photography"
    ]
}


def check_correct(dataset, row):
    true_labels = []
    true_labels.append(row[2])

    false_labels = labels[dataset].copy()
    false_labels.remove(row[2])

    if dataset == "cora":
        # '_' ->' '
        true_labels.append(row[2].replace('_', ' '))
        new_labels = [item.replace('_', ' ') for item in false_labels]
        false_labels.extend(new_labels)

        # '_' ->'-'
        true_labels.append(row[2].replace('_', '-'))
        new_labels = [item.replace('_', '-') for item in false_labels]
        false_labels.extend(new_labels)

        # lower case
        true_labels.append(row[2].lower())
        true_labels.append(true_labels[1].lower())
        new_labels = [item.lower() for item in false_labels]
        false_labels.extend(new_labels)

    elif dataset == "photo":
        # lower case
        true_labels.append(row[2].lower())
        true_labels.append(true_labels[1].lower())
        new_labels = [item.lower() for item in false_labels]
        false_labels.extend(new_labels)

    elif dataset == "citeseer":
        # content inside ()
        if row[2] != "Agents":
            true_labels.append(row[2].split('(')[1].replace(')', ''))

        new_labels = false_labels.copy()
        for item in new_labels:
            if item != "Agents":
                false_labels.append(item.split('(')[1].replace(')', ''))

                # acronym
        true_labels.append(row[2].split(' (')[0])
        new_labels = [item.split(' (')[0] for item in false_labels]
        false_labels.extend(new_labels)

    elif dataset == "reddit" or dataset == "instagram":
        # delete "s"
        true_labels.append(row[2][:-1])
        new_labels = [item[:-1] for item in false_labels]
        false_labels.extend(new_labels)

        # no "Users"
        true_labels.append(row[2][:-6])
        new_labels = [item[:-6] for item in false_labels]
        false_labels.extend(new_labels)

        # lower case
        true_labels.append(row[2].lower())
        true_labels.append(true_labels[1].lower())
        new_labels = [item.lower() for item in false_labels]
        false_labels.extend(new_labels)

    true_labels_in_completion = [x in row[1] for x in true_labels]
    false_label_in_completion = [x in row[1] for x in false_labels]

    if "none" in row[1] or "None" in row[1]:
        return "uncertain"

    if any(true_labels_in_completion) and not any(false_label_in_completion):
        return "correct"
    elif any(true_labels_in_completion) and any(false_label_in_completion):
        if dataset == "photo" and "Video Surveillance" in row[1] and row[2] == "Video Surveillance":
            return "correct"
        elif dataset == "photo" and row[2] == "Video" and "Video Surveillance" in row[1]:
            return "wrong"
        return "uncertain"
    elif any(false_label_in_completion):
        return "wrong"
    else:
        return "hallucination"


def write_correctness(file_path, dataset):
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        rows = list(reader)

    for row in rows:
        if row[0][0] < '0' or row[0][0] > '9':
            continue
        check_result = check_correct(dataset, row)
        try:
            row[4] = check_result
        except IndexError:
            row.append(check_result)

    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)


def num_tokens(file_path, dataset):
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        num_token = 0
        num_item = 0
        for row in reader:
            if row[0][0] < '0' or row[0][0] > '9':
                continue
            num_token = num_token + int(row[3])
            num_item = num_item + 1

    return num_token / num_item


def rewrite_output(file_path):
    # Read the existing rows
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)  # Store rows in memory for modification

    # Modify the rows as needed
    for row in rows:
        if row and (row[0][0] < '0' or row[0][0] > '9'):
            continue
        if len(row) > 1:  # Ensure there is a second column
            row[1] = row[1].split('.')[0]  # Modify row[1]

    # Write the modified rows back to the file
    with open(file_path, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)


def evaluate(file_path, model_name, dataset, prediction_type):
    test = 0

    # only extract output before the first "."
    if model_name == "mistral-7b" and prediction_type in ['none', 'lm', 'gnn', 'llm']:
        rewrite_output(file_path)

    write_correctness(file_path, dataset)
    true_labels, predict_labels = [], []
    total_num = 0
    hallucination = 0

    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if row[0][0] < '0' or row[0][0] > '9':
                continue

            total_num += 1
            if row[4] == "correct":
                true_labels.append(row[2])
                predict_labels.append(row[2])

            elif row[4] == "wrong":
                found = 0
                true_labels.append(row[2])
                for item in labels[dataset]:

                    if dataset == "cora":
                        if item.replace('_', ' ') in row[1] or item.lower() in row[1] or item in row[1] or item.replace(
                                '_', '-') in row[1] or item.replace('_', ' ').lower() in row[1] or item.replace('_',
                                                                                                                '-').lower() in \
                                row[1]:
                            found += 1
                            if found <= 1:
                                predict_labels.append(item)

                    elif dataset == "citeseer":

                        if item == "Agents":
                            if item in row[1]:
                                found += 1
                                if found <= 1:
                                    predict_labels.append(item)
                        else:
                            if item.split('(')[1].replace(')', '') in row[1] or item.split(' (')[0] in row[1] or item in \
                                    row[1]:
                                found += 1
                                if found <= 1:
                                    predict_labels.append(item)

                    elif dataset == "reddit" or dataset == "instagram":
                        if item[:-1] in row[1] or item in row[1] or item[:-6] in row[1] or item[:-1].lower() in row[
                            1] or item.lower() in row[1] or item[:-6].lower() in row[1]:
                            found += 1
                            if found <= 1:
                                predict_labels.append(item)

                    elif dataset == "photo":
                        if item == "Video":
                            if (item in row[1] or item.lower() in row[1]) and "Video Surveillance" not in row[1]:
                                found += 1
                                if found <= 1:
                                    predict_labels.append(item)
                        else:
                            if item in row[1] or item.lower() in row[1]:
                                found += 1
                                if found <= 1:
                                    predict_labels.append(item)

                    else:
                        if item in row[1]:
                            found += 1
                            if found <= 1:
                                predict_labels.append(item)

            else:
                hallucination += 1

            if len(predict_labels) != len(true_labels):
                test = test + 1
                if test <= 1:
                    print(row[0], row[1], row[2], len(predict_labels), len(true_labels), true_labels[-1])

    hallucination_rate = hallucination / total_num * 100
    accuracy, macro_f1, weighted_f1 = compute_acc_and_f1(predict_labels, true_labels)
    num_token = num_tokens(file_path, dataset)
    print(
        f'Accuracy: {accuracy:.3f}, F1 Score: {macro_f1:.3f}, Hallucination Rate: {hallucination_rate:.3f}, avg tokens:{num_token:.3f}')

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(
            [f'Accuracy: {accuracy:.3f}', f'F1 Score: {macro_f1:.3f}', f'Hallucination Rate: {hallucination_rate:.3f}',
             f'avg tokens:{num_token:.3f}'])
