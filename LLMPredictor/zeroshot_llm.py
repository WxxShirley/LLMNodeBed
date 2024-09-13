from openai import OpenAI
import os
import argparse
import torch
import prettytable as pt
def cora_response(model_name):
    device = 'cpu'
    data = torch.load(f"../datasets/cora.pt").to(device)
    discription = data.raw_texts[4]
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--model_name", type=str, default="qwen-turbo")

    args = parser.parse_args()
    if args.dataset == "cora":
        cora_response(args.model_name)
    else:
        print ("Error, no such dataset.")
        exit(1)
