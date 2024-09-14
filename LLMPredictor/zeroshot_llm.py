from openai import OpenAI
import os
import argparse
import torch
import prettytable as pt

def cora_response(model_name, index):
    device = 'cpu'
    data = torch.load(f"../datasets/cora.pt").to(device)
    discription = data.raw_texts[index]
    question = (
        "Question: Which of the following "
        "sub-categories of AI does this paper belong to? Here are the 7 categories: "
        "Rule_Learning, Neural_Networks, Case_Based, "
        "Genetic_Algorithms, Theory, Reinforcement_Learning, Probabilistic_Methods."
        "Reply only one category that you think this paper might belong to. Only reply the category name without any other words."
        "Answer:"
    )

    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"), # 如果您没有配置环境变量，请在此处用您的API Key进行替换
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
    print(completion.choices[0].message.content)
    if completion.choices[0].message.content == data.label_name[data.y[index]]:
        print ("right")
    else:
        print ("false")

def pubmed_response(model_name, index):
    device = 'cpu'
    data = torch.load(f"../datasets/pubmed.pt").to(device)
    discription = data.raw_texts[index]
    question = (
        "Question: Which of the following "
        "topic does this scientific publication talk about? Here are the 3 categories: "
        "Experimental, Diabetes Mellitus Type 1, Diabetes Mellitus Type 2."
        "Experimental category usually refers to Experimentally induced diabetes, "
        "Diabetes Mellitus Type 1 usually means the content of the paper is about Diabetes Mellitus Type 1,"
        "Diabetes Mellitus Type 2 usually means the content of the paper is about Diabetes Mellitus Type 2."
        "Reply only one category that you think this paper might belong to. Only reply the category name without any other words."
        "Answer:"
    )

    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"), # 如果您没有配置环境变量，请在此处用您的API Key进行替换
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
    print(completion.choices[0].message.content)
    if completion.choices[0].message.content == data.label_name[data.y[index]]:
        print ("right")
    else:
        print ("false")

def citeseer_response(model_name, index):
    device = 'cpu'
    data = torch.load(f"../datasets/citeseer.pt").to(device)
    discription = data.raw_texts[index]
    question = (
        "Question: Which of the following "
        "theme does this paper belong to? Here are the 6 categories: "
        "Agents, ML (Machine Learning), IR (Information Retrieval), DB (Databases), "
        "HCI (Human-Computer Interaction), AI (Artificial Intelligence."
        "Reply only one category that you think this paper might belong to. Only reply the category name without any other words."
        "Answer:"
    )

    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"), # 如果您没有配置环境变量，请在此处用您的API Key进行替换
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
    print(completion.choices[0].message.content)
    if completion.choices[0].message.content == data.label_name[data.y[index]]:
        print ("right")
    else:
        print ("false")

def wikics_response(model_name, index):
    device = 'cpu'
    data = torch.load(f"../datasets/wikics.pt").to(device)
    discription = data.raw_texts[index]
    question = (
        "Question: Which of the following "
        "branch of Computer science does this Wikipedia-based dataset belong to? Here are the 10 categories: "
        "Computational Linguistics, Databases, Operating Systems, Computer Architecture, Computer Security, "
        "Internet Protocols, Computer File Systems, "
        "Distributed Computing Architecture, Web Technology, Programming Language Topics."
        "Reply only one category that you think this paper might belong to. Only reply the category full name without any other words."
        "Answer:"
    )

    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"), # 如果您没有配置环境变量，请在此处用您的API Key进行替换
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
    print(completion.choices[0].message.content)
    if completion.choices[0].message.content == data.label_name[data.y[index]]:
        print ("right")
    else:
        print ("false")


def arxiv_response(model_name, index):
    device = 'cpu'
    data = torch.load(f"../datasets/arxiv.pt").to(device)
    discription = data.raw_texts[index]
    question = (
        "Question: Which of the following "
        "arXiv CS sub-categories does this dataset belong to? Here are the 40 categories: "
        "'arxiv cs na', 'arxiv cs mm', 'arxiv cs lo', 'arxiv cs cy', 'arxiv cs cr', "
        "'arxiv cs dc', 'arxiv cs hc', 'arxiv cs ce', 'arxiv cs ni', 'arxiv cs cc', "
        "'arxiv cs ai', 'arxiv cs ma', 'arxiv cs gl', 'arxiv cs ne', 'arxiv cs sc', "
        "'arxiv cs ar', 'arxiv cs cv', 'arxiv cs gr', 'arxiv cs et', 'arxiv cs sy', "
        "'arxiv cs cg', 'arxiv cs oh', 'arxiv cs pl', 'arxiv cs se', 'arxiv cs lg', "
        "'arxiv cs sd', 'arxiv cs si', 'arxiv cs ro', 'arxiv cs it', 'arxiv cs pf', "
        "'arxiv cs cl', 'arxiv cs ir', 'arxiv cs ms', 'arxiv cs fl', 'arxiv cs ds', "
        "'arxiv cs os', 'arxiv cs gt', 'arxiv cs db', 'arxiv cs dl', 'arxiv cs dm'. "
        "Reply only one category that you think this paper might belong to. Only reply the category name I given without any other words."
        "Answer:"
    )

    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"), # 如果您没有配置环境变量，请在此处用您的API Key进行替换
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
    print(completion.choices[0].message.content)

    if completion.choices[0].message.content == data.label_name[data.y[index]]:
        print ("right")
    else:
        print ("false")

def instagram_response(model_name, index):
    device = 'cpu'
    data = torch.load(f"../datasets/instagram.pt").to(device)
    discription = data.raw_texts[index]
    question = (
        "Question: Which of the following "
        "categories does this instagram user belong to? Here are the 2 categories: "
        "Normal Users, Commercial Users. "
        "Reply only one category that you think this paper might belong to. "
        "Only reply the category name I give of the category: Normal Users, Commercial Users, without any other words."
        "Answer:"
    )

    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"), # 如果您没有配置环境变量，请在此处用您的API Key进行替换
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
    print(completion.choices[0].message.content)

    if completion.choices[0].message.content == data.label_name[data.y[index]]:
        print ("right")
    else:
        print ("false")


def reddit_response(model_name, index):
    device = 'cpu'
    data = torch.load(f"../datasets/reddit.pt").to(device)
    discription = data.raw_texts[index]
    question = (
        "Question: Which of the following "
        "categories does this reddit user belong to? Here are the 2 categories: "
        "Normal Users, Popular Users. "
        "Popular Users' posted content are often more attractive."
        "Reply only one category that you think this paper might belong to. "
        "Only reply the category name I give of the category: Normal Users, Popular Users, without any other words."
        "Answer:"
    )

    client = OpenAI(
        api_key=os.getenv("DASHSCOPE_API_KEY"), # 如果您没有配置环境变量，请在此处用您的API Key进行替换
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
    print(completion.choices[0].message.content)

    if completion.choices[0].message.content == data.label_name[data.y[index]]:
        print ("right")
    else:
        print ("false")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="reddit")
    parser.add_argument("--model_name", type=str, default="qwen-turbo")

    index=190

    args = parser.parse_args()
    if args.dataset == "cora":
        cora_response(args.model_name, index)
    elif args.dataset == "pubmed":
        pubmed_response(args.model_name, index)
    elif args.dataset == "citeseer":
        citeseer_response(args.model_name, index)
    elif args.dataset == "wikics":
        wikics_response(args.model_name, index)
    elif args.dataset == "arxiv":
        arxiv_response(args.model_name, index)
    elif args.dataset == "instagram":
        instagram_response(args.model_name, index)
    elif args.dataset == "reddit":
        reddit_response(args.model_name, index)
    else:
        print ("Error, no such dataset.")
        exit(1)
