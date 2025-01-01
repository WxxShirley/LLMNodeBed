from openai import OpenAI
from openai import AzureOpenAI
import ast
import csv
import sys
from http import HTTPStatus
import dashscope
from dashscope import Generation
import tiktoken
import re
import requests
import json


sys.path.append("../")
from common import DIRECT_PROMPTS, LM_NEIGHBOR_PROMPTS, LLM_NEIGHBOR_PROMPTS, GNN_NEIGHBOR_PROMPTS, COT_PROMPTS, \
    TOT_PROMPTS, REACT_PROMPTS
from common import load_graph_dataset, compute_acc_and_f1
from common import API_KEYS, GPT4_RESOURCE, GPT4o_RESOURCES


class prediction:

    def __init__(self, prediction_type, dataset, model_name, index, write_file_path, graph_data):
        allowed_types = {'none', 'cot', 'tot', 'react', 'lm', 'gnn', 'llm', 'summary'}

        if prediction_type not in allowed_types:
            raise ValueError(f"Invalid prediction_type: {prediction_type}. ")

        self.prediction_type = prediction_type
        self.dataset = dataset
        self.model_name = model_name
        self.index = index
        self.write_file_path = write_file_path
        self.graph_data = graph_data

    def prompt(self):
        # calculate tokens
        enc = tiktoken.get_encoding("o200k_base")

        node_discription = "Given the information of the node: " + self.graph_data.raw_texts[self.index]
        # get 1-ego neighbor info
        one_neighbor_list = []
        one_neighbor_file_path = f"../datasets/1-neighbors/{self.dataset}.csv"

        with open(one_neighbor_file_path, 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if int(row[0]) == self.index:
                    one_neighbor_list = ast.literal_eval(row[1])

        one_neighbor_list_restrict = one_neighbor_list[:5]

        one_neighbor_info_list = []
        for iter in one_neighbor_list_restrict:
            one_neighbor_info_list.append(self.graph_data.raw_texts[iter])
        neighbor_info = "\none of its neighbors' feature:" + "\none of its neighbors' feature:".join(one_neighbor_info_list)

        if self.prediction_type == "none":
            question = DIRECT_PROMPTS[self.dataset]
            prompt_content = f"{node_discription}\n{question}"

        elif self.prediction_type == "cot":
            question = COT_PROMPTS[self.dataset]
            prompt_content = f"{node_discription}\n{question}"

        elif self.prediction_type == "tot":
            question = TOT_PROMPTS[self.dataset]
            prompt_content = f"{node_discription}\n{question}"

        elif self.prediction_type == "react":
            question = REACT_PROMPTS[self.dataset]
            prompt_content = f"{node_discription}\n{question}"

        elif self.prediction_type == "lm":
            k_1_neighbor_list = []
            k_1_neighbor_file_path = f"../datasets/k-1-neighbors/{self.dataset}.csv"

            with open(k_1_neighbor_file_path, 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    if int(row[0]) == self.index:
                        k_1_neighbor_list = ast.literal_eval(row[1])
            if len(k_1_neighbor_list) == 0:
                k_1_neighbor_list = one_neighbor_list[:5]

            print(k_1_neighbor_list)
            k_1_neighbor_info_list = []
            for iter in k_1_neighbor_list:
                k_1_neighbor_info_list.append(self.graph_data.raw_texts[iter])
            k_1_neighbor_info = "\none of its neighbors' feature:" + "\none of its neighbors' feature:".join(k_1_neighbor_info_list)

            question = LM_NEIGHBOR_PROMPTS[self.dataset]
            prompt_content = f"{node_discription}\n{k_1_neighbor_info}\n{question}"

        elif self.prediction_type == "gnn":
            k_1_neighbor_list = []
            k_1_neighbor_file_path = f"../datasets/gnn-neighbors/{self.dataset}.csv"

            with open(k_1_neighbor_file_path, 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                    if int(row[0]) == self.index:
                        k_1_neighbor_list = ast.literal_eval(row[1])
            if len(k_1_neighbor_list) == 0:
                k_1_neighbor_list = one_neighbor_list[:5]

            print(k_1_neighbor_list)
            k_1_neighbor_info_list = []
            for iter in k_1_neighbor_list:
                k_1_neighbor_info_list.append(self.graph_data.raw_texts[iter])
            k_1_neighbor_info = "\none of its neighbors' feature:" + "\none of its neighbors' feature:".join(k_1_neighbor_info_list)

            question = GNN_NEIGHBOR_PROMPTS[self.dataset]
            prompt_content = f"{node_discription}\n{k_1_neighbor_info}\n{question}"

        elif self.prediction_type == "summary":
            if self.dataset in ["cora", "citeseer", "arxiv", "wikics"]:
                data_type = "paper"
                link_type = "citation"
            elif self.dataset in ["instagram", "reddit"]:
                data_type = "user"
                link_type = "following"
            else:
                data_type = "item"
                link_type = "co-purchase"
            discription = f"The following list records some {data_type} related to the current one, with relationship being {link_type}."
            question = f"Please summarize the information above with a short paragraph, find some common points which can reflect the category of this {data_type}."
            prompt_content = f"{discription}\n{neighbor_info}\n{question}"

        else:
            summary_prediction = prediction("summary", self.dataset, self.model_name, self.index, self.write_file_path,
                                            self.graph_data)
            summary, summary_token_len = summary_prediction.get_response()
            question = LLM_NEIGHBOR_PROMPTS[self.dataset]
            prompt_content = f"{node_discription}\n{summary}\n{question}"

        if self.prediction_type != "llm":
            input_tokens = enc.encode(prompt_content)
            input_tokens_len = len(input_tokens)
        else:
            input_tokens = enc.encode(prompt_content)
            input_tokens_len = summary_token_len + len(input_tokens)

        return prompt_content, input_tokens_len

    def get_response(self):

        enc = tiktoken.get_encoding("o200k_base")
        prompt_content, input_tokens_len = self.prompt()

        if self.model_name == "chatglm3-6b":
            dashscope.api_key = API_KEYS[self.model_name]
            messages = [
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
                api_key=API_KEYS[self.model_name],
                base_url="https://api.deepseek.com")
            response = client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': f"{prompt_content}"
                    }
                ],
                stream=False
            )
            prediction = response.choices[0].message.content


        elif self.model_name == "gpt-4" or self.model_name == "gpt-4o":
            if self.model_name == "gpt-4o":
                current_llm = GPT4o_RESOURCES
            else:
                current_llm = GPT4_RESOURCE

            kvs = current_llm["pairs"][3]
            client = AzureOpenAI(
                azure_endpoint=kvs["endpoint"],
                api_key=kvs["key"],
                api_version=current_llm["version"],
            )

            response = client.chat.completions.create(
                model=current_llm["model"],
                messages=[
                    {"role": "system", "content": "You are a helpful assistant"},
                    {"role": "user", "content": prompt_content},
                ]
            )
            prediction = response.choices[0].message.content


        elif self.model_name == "mistral-7b":
            headers = {
                "Content-Type": 'application/json',
            }
            url = f"http://127.0.0.1:8008/v1/chat/completions"

            prompt = prompt_content
            payload = json.dumps({
                "model": "Mistral-7B-Instruct-v0.2",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.2,
                "top_p": 0.1,
                "frequency_penalty": 0,
                "presence_penalty": 1.05,
                "max_tokens": 4096,
                "stream": False,
                "stop": None
            })
            response = requests.post(url, headers=headers, data=payload, timeout=300)
            try:
                resp = response.json()
                msg = resp["choices"][0]["message"]["content"]
                print(msg)
                prediction = msg
            except Exception as e:
                print("Decode Error")


        else:
            client = OpenAI(
                api_key=API_KEYS[self.model_name],
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

        if self.prediction_type == "summary":
            print(prediction)

        output_tokens = enc.encode(prediction)
        all_length = len(output_tokens) + input_tokens_len
        return prediction, all_length

    def write_in_file(self):

        prediction_content, all_token_len = self.get_response()

        if self.prediction_type == "summary":
            print(prediction_content)

        else:
            true_label = self.graph_data.label_name[self.graph_data.y[self.index]]

            if self.prediction_type in ['cot', 'tot', 'react']:
                match = re.search(r'<classification:\s*(.*?)>', prediction_content)
                try:
                    if match is not None and match.group(1) != "":
                        prediction_content = match.group(1)
                    else:
                        # Attempt to split using "classification"
                        try:
                            prediction_content = prediction_content.split("classification")[1]
                        except IndexError:
                            # If the split fails, try splitting with "Classification"
                            prediction_content = prediction_content.split("Classification")[1]
                except AttributeError:
                    # Handle the case where match is None
                    try:
                        prediction_content = prediction_content.split("classification")[1]
                    except IndexError:
                        prediction_content = prediction_content.split("Classification")[1]

            with open(self.write_file_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                if self.dataset == "arxiv":
                    writer.writerow([self.index, prediction_content[:11], true_label, all_token_len])
                else:
                    writer.writerow([self.index, prediction_content, true_label, all_token_len])
                print([self.index, prediction_content, true_label, all_token_len])








