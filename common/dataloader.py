import torch 
import os
import json
from collections import defaultdict


def load_graph_dataset(dataset_name, device, emb_model="shallow"):
    graph_data = torch.load(f"../datasets/{dataset_name}.pt").to(device)
    
    if emb_model != "shallow":
        assert os.path.exists(f"../datasets/{emb_model}/{dataset_name}.pt")
        node_feat = torch.load(f"../datasets/{emb_model}/{dataset_name}.pt", map_location=device).to(device).type(torch.float)
        graph_data.x = node_feat
    
    if len(graph_data.train_mask) == 10:
        graph_data.train_mask, graph_data.val_mask, graph_data.test_mask = graph_data.train_mask[0], graph_data.val_mask[0], graph_data.test_mask[0]
    
    return graph_data


def load_graph_dataset_for_tape(dataset_name, device, use_gpt=False, gpt_name="GPT-3.5-turbo"):
    graph_data = torch.load(f"../../datasets/{dataset_name}.pt").to(device)
    
    if len(graph_data.train_mask) == 10:
        graph_data.train_mask, graph_data.val_mask, graph_data.test_mask = graph_data.train_mask[0], graph_data.val_mask[0], graph_data.test_mask[0]
    
    if use_gpt:
        prediction_file = f"../../results/LLMEncoder/TAPE/{gpt_name}/{dataset_name}.json" 
        if not os.path.exists(prediction_file):
            print("LLM generated explanations do not exist, use original text instead.")
        else:
            file_reader = open(prediction_file, 'r')
            id2exp = defaultdict(str)
            for line in file_reader:
                content = json.loads(line)
                answer = content["answer"]
                # TODO: check whether it needs to replace prediction part
                # if "Explanation:" in content["answer"]:
                #     answer = content["answer"][content["answer"].index("Explanation:"):]
                # else:
                #     answer = content["answer"]
                id2exp[content["idx"]] = answer
            
            raw_texts = [id2exp.get(idx, graph_data.raw_texts[idx]) for idx in range(graph_data.num_nodes)]
            return graph_data, len(graph_data.label_name), raw_texts

    return graph_data, len(graph_data.label_name), graph_data.raw_texts
