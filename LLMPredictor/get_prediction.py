from openai import OpenAI
import ast
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
from kneighbor import kneighbor_index, extract_common_values, k_1_neighbor_intersection, oneneighbor_index


sys.path.append("../")
from common import ZEROSHOT_PROMPTS, RAW_NEIGHBOR_PROMPTS, LM_NEIGHBOR_PROMPTS, LLM_NEIGHBOR_PROMPTS
from common import load_graph_dataset, compute_acc_and_f1



class prediction:

    def __init__(self, prediction_type, dataset, model_name, index, write_file_path, graph_data):
        allowed_types = {'none', 'raw', 'lm', 'llm', 'summary'}
        
        if prediction_type not in allowed_types:
            raise ValueError(f"Invalid prediction_type: {prediction_type}. ")
        
        self.prediction_type = prediction_type
        self.dataset = dataset
        self.model_name = model_name
        self.index = index
        self.write_file_path = write_file_path
        self.graph_data = graph_data
    

    def prompt(self):
        
        node_discription = self.graph_data.raw_texts[self.index]

        G=to_networkx(self.graph_data)
        neighbor_list = list(nx.neighbors(G, self.index))
        if self.dataset == "instagram":
            neighbor_list = neighbor_list[1:]
        neighbor_list = neighbor_list[:5] if len(neighbor_list) > 5 else neighbor_list
        neighbor_info_list = []
        for iter in neighbor_list:
            neighbor_info_list.append(self.graph_data.raw_texts[iter])
        neighbor_info = "\none of its neighbors' feature:".join(neighbor_info_list)


        if self.prediction_type == "none":
            question = ZEROSHOT_PROMPTS[self.dataset]
            prompt_content = f"{node_discription}\n{question}"
        
        elif self.prediction_type == "raw":
            question = RAW_NEIGHBOR_PROMPTS[self.dataset]
            prompt_content = f"{node_discription}\n{neighbor_info}\n{question}"

        elif self.prediction_type == "lm":
            # k_1_neighbor_list = k_1_neighbor_intersection(self.dataset)[self.index]
            # if len(k_1_neighbor_list) == 0:
            #     k_1_neighbor_list = list(nx.neighbors(G, self.index))
            # k_1_neighbor_info_list = []
            k_1_neighbor_list = []
            k_1_neighbor_file_path = f"../datasets/k-1-neighbors/{self.dataset}.csv"
            
            with open(k_1_neighbor_file_path, 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    
                    if int(row[0]) == self.index:
                        k_1_neighbor_list = ast.literal_eval(row[1])
            if len(k_1_neighbor_list) == 0:
                k_1_neighbor_list = list(nx.neighbors(G, self.index))[:5]
            
            k_1_neighbor_info_list = []
            for iter in k_1_neighbor_list:
                k_1_neighbor_info_list.append(self.graph_data.raw_texts[iter])
            k_1_neighbor_info = "\none of its neighbors' feature:".join(k_1_neighbor_info_list)

            question = LM_NEIGHBOR_PROMPTS[self.dataset]
            prompt_content = f"{node_discription}\n{k_1_neighbor_info}\n{question}"
        
        elif self.prediction_type == "summary":
            data_type = "articles" if self.dataset in ["cora", "citeseer"] else "users"
            discription = f"The following list records some {data_type} related to the current one"
            question = "Please summarize the information above with a short paragraph, find some common points which can reflect the category of this paper."
            prompt_content = f"{discription}\n{neighbor_info}\n{question}"

        else:
            summary_prediction = prediction("summary", self.dataset, self.model_name, self.index, self.write_file_path, self.graph_data)
            summary = summary_prediction.get_response()
            question = LLM_NEIGHBOR_PROMPTS[self.dataset]
            prompt_content = f"{node_discription}\n{summary}\n{question}"
        
        return prompt_content
            
    


    def get_response(self):
        prompt_content = self.prompt()

        if self.model_name == "chatglm3-6b":
            dashscope.api_key = "sk-946342daa7234baeb39287866be76505"
            messages=[
                {
                    'role': 'user',
                    'content': f"{prompt_content}"
                }]
    
            gen = Generation()
            response = gen.call(
                'chatglm3-6b',
                messages=messages,
                result_format='message',  # set the result is message format.
            )
            prediction = response["output"]["choices"][0]["message"]["content"].replace('\n', '').strip()


        elif self.model_name == "deepseek-chat":
            client = OpenAI(
                api_key="sk-a8a2461880014e87bc10ca6c9c62bfdc", 
                base_url="https://api.deepseek.com")
            response = client.chat.completions.create(
                model = self.model_name,
                messages=[
                    {
                    'role': 'user',
                    'content': f"{prompt_content}"
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
                api_key="sk-6ed3b105aaac459097168fd8cca58513",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            completion = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': f"{prompt_content}"
                    }
                ]
            )
            prediction = completion.choices[0].message.content

        return prediction
    

    def write_in_file(self):

        prediction_content = self.get_response()

        if self.prediction_type == "summary":
            print(prediction_content)
        
        else:
            true_label = self.graph_data.label_name[self.graph_data.y[self.index]]

            with open(self.write_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                if self.dataset == "arxiv":
                    writer.writerow([self.index, prediction_content[:11], true_label])
                else:
                    writer.writerow([self.index, prediction_content, true_label])
                print([self.index, prediction_content, true_label])








