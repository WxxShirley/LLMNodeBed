import csv
import sys
import re
import copy
sys.path.append("../..")
from common import DIRECT_PROMPTS, LLM_NEIGHBOR_PROMPTS, COT_PROMPTS, \
    TOT_PROMPTS, REACT_PROMPTS, ALL_NEIGHBOR_PROMPTS
from common import prepare_edge_list
from apis import invoke_llm_api


class LLMDirectInference:
    def __init__(self, dataset_name, graph_data, prompt_type, llm_name, write_file_path, maximum_neighbor=8):
        print(f"Initialize the LLM Direct Inference Predictor based on {prompt_type.upper()} prompt template ... ")
        self.dataset_name = dataset_name
        self.graph_data = graph_data
        
        prompt_mapping = {
            "direct": DIRECT_PROMPTS,
            "cot": COT_PROMPTS,
            "tot": TOT_PROMPTS,
            "react": REACT_PROMPTS,
            "neighbor": ALL_NEIGHBOR_PROMPTS,
            "summary": LLM_NEIGHBOR_PROMPTS
        }
        self.prompt_type = prompt_type
        self.prompt = prompt_mapping[prompt_type].get(dataset_name, "")

        self.llm_name = llm_name 
        self.write_path = write_file_path
        self.maximum_neighbor = maximum_neighbor
        self.label_space = graph_data.label_name
        
        if prompt_type in ["neighbor", "summary"]:
            self.edge_list = prepare_edge_list(graph_data.edge_index.cpu(), graph_data.num_nodes)

    def do_prediction(self, test_idxes, has_inferenced):
        print(f"Do the LLM Direct Inference Job ... ")
        for idx in test_idxes:
            if idx in has_inferenced:
                continue
            
            origin_txt = self.graph_data.raw_texts[idx]
            
            if self.prompt_type == "summary":
                data_type, link_type = "paper", "citation"
                if self.dataset_name in ["instagram", "reddit"]:
                    data_type, link_type = "user", "following"
                elif self.dataset_name in ["wikics"]:
                    data_type, link_type = "entity", "web link"
                else:
                    data_type, link_type = "item", "co-purchase"
                
                neighbors = self.edge_list[idx][:self.maximum_neighbor]
                neighbor_str = "Neighbor's text information:" + "\nNeighbor's text information:".join([self.graph_data.raw_texts[neigh] for neigh in neighbors])
          
                discription = f"The following list records some {data_type} related to the current one, with relationship being {link_type}."
                question = f"Please summarize the information above with a short paragraph, find some common points which can reflect the category of this {data_type}."
                query_content = f"{discription}\n{neighbor_str}\n{question}"
            elif self.prompt_type == "neigbor":
                neighbors = self.edge_list[idx][:self.maximum_neighbor]
                neighbor_str = "Neighbor's text information:" + "\nNeighbor's text information:".join([self.graph_data.raw_texts[neigh] for neigh in neighbors])
                
                query_content = f"{origin_txt}\n{neighbor_str}\n{self.prompt}"
            else:
                reformat_str = "The answer should be in the strict format as: <reason: >, <classification: Your Classified Label>." if self.prompt_type in ["tot", "react"] else ""
                query_content = f"{origin_txt}\n{self.prompt}{reformat_str}"

            try: 
                prediction = invoke_llm_api(self.llm_name, query_content)
                if self.prompt_type == "summary":
                    generated_summary = copy.deepcopy(prediction)
                    new_query_content = f"{origin_txt}\n{generated_summary}\n{self.prompt}"
                    prediction = invoke_llm_api(self.llm_name, new_query_content)
            except Exception as e:
                print(f"[ERROR] {idx} encounter error {e}")
                prediction = None 

            if prediction:
                true_label = self.label_space[self.graph_data.y[idx].cpu().item()]

                # Open-source LLM
                if self.llm_name in ['Llama-3.1-8B-Instruct', 'Mistral-7B-Instruct-v0.2']:
                    origin_pred = copy.deepcopy(prediction)
                    prediction = post_parse_for_opensource_llm(prediction, self.label_space, self.dataset_name)

                    if prediction == " " and self.prompt_type in ["cot", "tot", "react"]:
                        reformat_query = f"{origin_pred}\nGiven the content, what is the final classification answer? Here are the categories: {self.label_space}. Only answer ONE category name."
                        prediction = invoke_llm_api(self.llm_name, reformat_query)
                        prediction = post_parse_for_opensource_llm(prediction, self.label_space, self.dataset_name)
                        
                # Close-source LLM
                else:
                    if self.prompt_type in ['cot', 'tot', 'react']:
                        match = re.search(r'<classification:\s*(.*?)>', prediction)
                        if match is not None and match.group(1) != "":
                            prediction = match.group(1)
                        else:
                            match = re.search(r'<classification:\s*>?\s*([\w_]+)', prediction)
                            if match is not None and match.group(1) != "":
                                prediction = match.group(1)

                prediction = prediction[:11] if self.dataset_name == "arxiv" else prediction
                
                with open(self.write_path, 'a', newline='') as file:
                    writer = csv.writer(file) 
                    writer.writerow([idx, prediction, true_label])
                    file.flush()
                
                print(idx, prediction, true_label)


def post_parse_for_opensource_llm(pred_str, label_space, dataset_name):
    # print("RAW: ", pred_str)
    lines = [line.strip() for line in pred_str.splitlines() if line.strip()]

    for line in reversed(lines):
        match = re.fullmatch(r'[\w\s]+', line)
        if match:
            prediction = match.group(0)
            return prediction 
    
    for gt_label in label_space:
        candidates = {
            "cora": [gt_label, gt_label.replace('_', ' '), gt_label.replace('_', '-'), gt_label.lower()], 
            "instagram": [gt_label, gt_label.lower(), gt_label[:-1], gt_label[:-1].lower(), gt_label.split(" ")[0]],
            "photo": [gt_label, gt_label.lower()],
            "wikics": [gt_label, gt_label.lower()]
         }[dataset_name]
        
        for candidate in candidates:
            if candidate in pred_str:
                 return gt_label
            
    return " "
