from openai import OpenAI
import os
import argparse
import torch
import prettytable as pt
import csv
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

CORA_Q = """Question: Which of the following 
        sub-categories of AI does this paper belong to? Here are the 7 categories: 
        Rule_Learning, Neural_Networks, Case_Based, 
        Genetic_Algorithms, Theory, Reinforcement_Learning, Probabilistic_Methods.
        Reply only one category that you think this paper might belong to. 
        Only reply the category name without any other words.
        Answer:"""

PUBMED_Q = """Question: Which of the following 
        topic does this scientific publication talk about? Here are the 3 categories: 
        Experimental, Diabetes Mellitus Type 1, Diabetes Mellitus Type 2.
        Experimental category usually refers to Experimentally induced diabetes, 
        Diabetes Mellitus Type 1 usually means the content of the paper is about Diabetes Mellitus Type 1,
        Diabetes Mellitus Type 2 usually means the content of the paper is about Diabetes Mellitus Type 2.
        Reply only one category that you think this paper might belong to. Only reply the category name without any other words.
        Answer:
    """

CITESEER_Q = """Question: Which of the following 
        theme does this paper belong to? Here are the 6 categories: 
        Agents, ML (Machine Learning), IR (Information Retrieval), DB (Databases), 
        HCI (Human-Computer Interaction), AI (Artificial Intelligence).
        Reply only one category that you think this paper might belong to. 
        Only reply the category full name I give you without any other words.
        Answer:
    """

WIKICS_Q = """Question: Which of the following 
        branch of Computer science does this Wikipedia-based dataset belong to? Here are the 10 categories: 
        Computational Linguistics, Databases, Operating Systems, Computer Architecture, Computer Security, 
        Internet Protocols, Computer File Systems, 
        Distributed Computing Architecture, Web Technology, Programming Language Topics.
        Reply only one category that you think this paper might belong to. Only reply the category full name without any other words.
        Answer:
    """

ARXIV_Q = """Question: Which of the following 
        arXiv CS sub-categories does this dataset belong to? Here are the 40 categories: 
        'arxiv cs na', 'arxiv cs mm', 'arxiv cs lo', 'arxiv cs cy', 'arxiv cs cr', 
        'arxiv cs dc', 'arxiv cs hc', 'arxiv cs ce', 'arxiv cs ni', 'arxiv cs cc',
        'arxiv cs ai', 'arxiv cs ma', 'arxiv cs gl', 'arxiv cs ne', 'arxiv cs sc', 
        'arxiv cs ar', 'arxiv cs cv', 'arxiv cs gr', 'arxiv cs et', 'arxiv cs sy', 
        'arxiv cs cg', 'arxiv cs oh', 'arxiv cs pl', 'arxiv cs se', 'arxiv cs lg', 
        'arxiv cs sd', 'arxiv cs si', 'arxiv cs ro', 'arxiv cs it', 'arxiv cs pf', 
        'arxiv cs cl', 'arxiv cs ir', 'arxiv cs ms', 'arxiv cs fl', 'arxiv cs ds', 
        'arxiv cs os', 'arxiv cs gt', 'arxiv cs db', 'arxiv cs dl', 'arxiv cs dm'. 
        Use the words in this part to answer me, not the explanation part bellow.
        
        Here are the explanation of each category:
        'cs.AI (Artificial Intelligence)',

        'cs.AR (Hardware Architecture)',
        
        'cs.CC (Computational Complexity)',
        
        'cs.CE (Computational Engineering, Finance, and Science)',
        'cs.CG (Computational Geometry)',
        
        'cs.CL (Computation and Language)',
        
        'cs.CR (Cryptography and Security)',
        'cs.CV (Computer Vision and Pattern Recognition)',
        'cs.CY (Computers and Society)',
        'cs.DB (Databases)',
        'cs.DC (Distributed, Parallel, and Cluster Computing)',
        'cs.DL (Digital Libraries)',
        'cs.DM (Discrete Mathematics)',
        'cs.DS (Data Structures and Algorithms)',
        'cs.ET (Emerging Technologies)',
        'cs.FL (Formal Languages and Automata Theory)',
        'cs.GL (General Literature)',
        'cs.GR (Graphics)',
        'cs.GT (Computer Science and Game Theory)',
        'cs.HC (Human-Computer Interaction)',
        
        'cs.IR (Information Retrieval)',
        'cs.IT (Information Theory)',
        'cs.LG (Machine Learning)',
        'cs.LO (Logic in Computer Science)',
        'cs.MA (Multiagent Systems)',
        'cs.MM (Multimedia)',
        'cs.MS (Mathematical Software)',
        'cs.NA (Numerical Analysis)',
        'cs.NE (Neural and Evolutionary Computing)',
        'cs.NI (Networking and Internet Architecture)',
        'cs.OH (Other Computer Science)',
        'cs.OS (Operating Systems)',
        'cs.PF (Performance)',
        'cs.PL (Programming Languages)',
        'cs.RO (Robotics)',
        'cs.SC (Symbolic Computation)',
        'cs.SD (Sound)',
        'cs.SE (Software Engineering)',
        'cs.SI (Social and Information Networks)',
        'cs.SY (Systems and Control)'
        Reply only one category that you think this paper might belong to. 
        Only reply the category name (not the explanation) I given without any other words, please don't use your own words.
        Be careful, only use the name of the category I give you, not the explanation part or any other words.
        Answer:
    """

INSTAGRAM_Q = """Question: Which of the following 
        categories does this instagram user belong to? Here are the 2 categories: 
        Normal Users, Commercial Users. 
        Reply only one category that you think this paper might belong to. 
        Only reply the category name I give of the category: Normal Users, Commercial Users, without any other words.
        Answer:
    """

REDDIT_Q = """Question: Which of the following 
        categories does this reddit user belong to? Here are the 2 categories: 
        Normal Users, Popular Users. 
        Popular Users' posted content are often more attractive.
        Reply only one category that you think this paper might belong to. 
        Only reply the category name I give of the category: Normal Users, Popular Users, without any other words.
        Answer:
    """

DEFAULT_QUESTION = {
    "cora_q": CORA_Q,
    "pubmed_q": PUBMED_Q,
    "citeseer_q": CITESEER_Q,
    "wikics_q": WIKICS_Q,
    "arxiv_q": ARXIV_Q,
    "instagram_q": INSTAGRAM_Q,
    "reddit_q": REDDIT_Q,
}

def get_response(dataset, model_name, index, write_file_path):
    device = 'cpu'
    data = torch.load(f"../datasets/{dataset}.pt").to(device)

    discription = data.raw_texts[index]
    question = DEFAULT_QUESTION[f"{dataset}_q"]

    client = OpenAI(
        #api_key=os.getenv("DASHSCOPE_API_KEY"), # 如果您没有配置环境变量，请在此处用您的API Key进行替换
        api_key="sk-6ed3b105aaac459097168fd8cca58513",
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 填写DashScope服务的base_url
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

    prediction=completion.choices[0].message.content
    true_label=data.label_name[data.y[index]]

    with open(write_file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([model_name, index, prediction, true_label])
        print([model_name, index, prediction, true_label])

def acc_f1(dataset, file_path, model_name):
    true_labels=[]
    predict_labels=[]
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for row in reader:
            if str(row[0]) == model_name:
                if str(str(row[1])[0]) == "A":
                    continue
                true_labels.append(row[3])
                predict_labels.append(row[2])


    accuracy = accuracy_score(true_labels, predict_labels)
    print(f'Accuracy: {accuracy:.3f}')
    f1 = f1_score(true_labels, predict_labels, average='weighted')
    print(f'F1 Score: {f1:.3f}')

    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([model_name, f'Accuracy: {accuracy:.3f}',f'F1 Score: {f1:.3f}'])



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="arxiv")
    parser.add_argument("--model_name", type=str, default="qwen-turbo")


    args = parser.parse_args()

    device = 'cpu'

    # get the index set of the training set on given dataset
    data = torch.load(f"../datasets/{args.dataset}.pt").to(device)
    test_mask = data.test_mask[0] if len(data.test_mask) == 10 else data.test_mask
    test_indexes = torch.where(test_mask == True)[0].cpu().numpy().tolist()

    test_indexes=[1,2,3,4,5]

    # create csv file
    os.makedirs("../results/LLMPredictor/llm_zero_shot", exist_ok=True)
    file_path = f"../results/LLMPredictor/llm_zero_shot/{args.dataset}.csv"
    write_file = open(file_path, 'a', newline='')

    #make predictions
    for index in test_indexes:

        #test if already have
        whether_to_predict=1

        with open (file_path,'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if str(row[0]) == str(args.model_name) and str(row[1]) == str(index):
                    whether_to_predict=0
                    break

        if whether_to_predict == 0:
            continue

        #prediction
        get_response(args.dataset, args.model_name, index, file_path)

    # calculate acc and f1
    acc_f1(args.dataset, file_path, args.model_name)

