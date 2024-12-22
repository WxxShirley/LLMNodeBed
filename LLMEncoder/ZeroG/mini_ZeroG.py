import torch 
import argparse
import sys
import tqdm
import os
import torch.nn as nn 
import json
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
sys.path.append("../")
from common import compute_acc_and_f1, mean_pooling, mean_pooling_llm, normalize_adj_matrix
from common import load_graph_dataset, MODEL_PATHs as encoder_dict, ZeroG_DESC as descriptions


label_descriptions = {
    "cora": {'Rule_Learning': 'A method of learning rules from data, often used in machine learning to derive logical statements that predict outcomes.', 'Neural_Networks': 'Computational models inspired by the human brain, used for recognizing patterns and making predictions.', 'Case_Based': 'An approach that solves new problems based on solutions of similar past cases, often used in expert systems.', 'Genetic_Algorithms': 'Optimization algorithms inspired by natural selection, used to find approximate solutions to complex problems.', 'Theory': 'The foundational principles and ideas that underlie the study and application of a particular domain or discipline.', 'Reinforcement_Learning': 'A type of machine learning where agents learn to make decisions by receiving rewards or punishments.', 'Probabilistic_Methods': 'Techniques that use probability theory to model uncertainty and make predictions based on data.'}, 
    "citeseer": {'Agents': 'Software entities that perform tasks autonomously, often interacting with other agents and systems.', 'ML (Machine Learning)': 'A field of AI focused on algorithms that improve automatically through experience and data.', 'IR (Information Retrieval)': 'The process of obtaining relevant information from large collections of data, often with search engines.', 'DB (Databases)': 'Organized collections of structured information stored electronically, accessible in various ways.', 'HCI (Human-Computer Interaction)': 'The study of how people interact with computers and designing interfaces that facilitate this.', 'AI (Artificial Intelligence)': 'The simulation of human intelligence processes by machines, especially computer systems.'},
    "pubmed": {'Experimental': 'Pertaining to a trial or test phase involving innovative methods or studies.', 'Diabetes Mellitus Type 1': '', 'Diabetes Mellitus Type 2': ''}, 
    # {'Experimental': 'Pertaining to a trial or test phase involving innovative methods or studies.', 'Diabetes Mellitus Type 1': 'A chronic condition where the pancreas produces little or no insulin due to autoimmune destruction of insulin-producing cells.', 'Diabetes Mellitus Type 2': 'A metabolic disorder characterized by high blood sugar and insulin resistance, commonly linked to genetics and lifestyle factors.'},
    "wikics": {'Computational Linguistics': 'The study of using computational methods to process and analyze human language data.', 'Databases': 'Structured collections of data, typically managed by database management systems to enable retrieval and manipulation.', 'Operating Systems': 'Software that manages computer hardware and software resources, providing common services for application programs.', 'Computer Architecture': "The design and organization of a computer's core components, including the CPU, memory, and input/output systems.", 'Computer Security': 'Measures and protocols for protecting computer systems and data from unauthorized access, damage, or theft.', 'Internet Protocols': 'Rules and conventions for data exchange across networks, forming the foundational protocols for the Internet.', 'Computer File Systems': 'Methods and data structures used by operating systems to store, retrieve, and organize files on storage devices.', 'Distributed Computing Architecture': 'Systems that distribute computation across multiple networked computers to achieve parallel processing and resilience.', 'Web Technology': 'Tools and techniques for creating, managing, and interacting with websites and web applications.', 'Programming Language Topics': 'Concepts, syntax, and semantics of programming languages, including paradigms, structures, and implementations.'}, 
    "instagram": {'Normal Users': 'Individuals who use the platform for personal, non-commercial purposes. They typically have access to basic features and services.', 'Commercial Users': 'Businesses or professionals utilizing the platform for commercial activities. They may have access to advanced features that support business operations.'}, 
    "reddit": {'Normal Users': 'Regular users who engage with the platform without significant influence or large followings.', 'Popular Users': 'Users with a substantial online presence and influence, attracting a larger audience or following.'} 
}


class MiniZeroG(nn.Module):
    def __init__(self, args):
        super(MiniZeroG, self).__init__()
        self.args = args 
        self.device = device
        self.add_virtual_node = args.virtual_node
        
        assert args.text_encoder in encoder_dict.keys()
        encoder_fullname = encoder_dict[args.text_encoder]

        if args.text_encoder in ["SentenceBert", "roberta", "e5-large", "MiniLM"]:
            self.tokenizer = AutoTokenizer.from_pretrained(encoder_fullname)
            self.text_model = AutoModel.from_pretrained(encoder_fullname).to(self.device)
        else:
            # Load local LLM
            self.tokenizer = AutoTokenizer.from_pretrained(encoder_fullname)
            self.text_model = AutoModelForCausalLM.from_pretrained(encoder_fullname, torch_dtype=torch.float16).to(self.device)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def text_forward(self, text):
        text = "Empty text" if len(text) == 0 else text
        tokens = self.tokenizer(text, max_length=256, return_tensors='pt', truncation=True, padding=True).to(self.device)
        if self.args.text_encoder in ["SentenceBert", "roberta", "e5-large"]:
            # Except for MiniLM, we use the CLS embedding
            text_embeds = self.text_model(**tokens)[0][:, 0, :]
        elif self.args.text_encoder == "MiniLM":
            text_embeds = mean_pooling(self.text_model(**tokens), tokens["attention_mask"])
        else:
            # For LLMs, we use the sentence-level mean pooling
            outputs = self.text_model(**tokens, output_hidden_states=True)
            text_embeds = mean_pooling_llm(outputs.hidden_states[-1], tokens["attention_mask"])
        del tokens
        
        return text_embeds
    
    def zero_shot_eval(self, node_embeds, label_embeds, data, R):
        if self.args.if_norm:
            # print(f"hit feature normalization")
            node_embeds = (node_embeds - node_embeds.mean(0)) / node_embeds.std(0)
            label_embeds = (label_embeds - label_embeds.mean(0)) / label_embeds.std(0)
        
        if self.add_virtual_node:
            num_exist_nodes = data.y.shape[0] + 1 
            virtual_node_idx = data.y.shape[0]
            new_edges_to_virtual = []
            for node_idx in range(num_exist_nodes-1):
                new_edges_to_virtual.extend([[node_idx, virtual_node_idx], [virtual_node_idx, node_idx]])
        
            new_edge_index = torch.cat([data.edge_index.t(), torch.tensor(new_edges_to_virtual, dtype=torch.long).to(self.device)], dim=0).t()
            adj_normed = normalize_adj_matrix(new_edge_index, num_exist_nodes, self.device)
        else:
            adj_normed = normalize_adj_matrix(data.edge_index, data.num_nodes, self.device)

        for _ in range(R):
            node_embeds = torch.mm(adj_normed, node_embeds)
        
        node_embeds = node_embeds[:-1, :] if args.virtual_node else node_embeds
        node_embeds /= node_embeds.norm(dim=-1, keepdim=True).to(self.device)
        label_embeds /= label_embeds.norm(dim=-1, keepdim=True)

        dists = torch.einsum('bn,cn->bc', node_embeds, label_embeds)
        preds = torch.argmax(dists, dim=1)

        test_acc, test_f1 = compute_acc_and_f1(preds[data.test_mask].cpu(), data.y[data.test_mask].cpu())
        eval_acc, eval_f1 = compute_acc_and_f1(preds[data.val_mask].cpu(), data.y[data.val_mask].cpu())
        del node_embeds, label_embeds, dists, preds
        
        return [test_acc, test_f1], [eval_acc, eval_f1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_datasets", type=str, default="cora,citeseer,pubmed,wikics,instagram,reddit")
    parser.add_argument("--if_norm", action="store_true", default=True, help="Indicator of normalization")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--text_encoder", type=str, default="SentenceBert", help="Type of text encoder")
    parser.add_argument("--R_list", type=str, default="0,1,2,3,4,5,6,7,8,9,10")
    parser.add_argument("--virtual_node", type=int, default=1)
    parser.add_argument("--enrich_label", type=int, default=1)

    args = parser.parse_args()
    device = torch.device(args.device)
    print("=" * 20)
    print(args)

    test_names, test_graphs = args.test_datasets.split(","), []

    for dataset_name in test_names:
        test_graph = load_graph_dataset(dataset_name, device)
        test_graphs.append(test_graph)

    model = MiniZeroG(args)
    model.eval()

    res_list, score_dict = [], {}
    for idx, test_graph_data in enumerate(test_graphs):
        # Zero-shot Prediction 
        test_graph_name = test_names[idx]
        with torch.no_grad():
            r_list = [int(r) for r in args.R_list.split(",")]
            best_eval_acc, best_test_acc, best_test_f1, best_r = 0, 0, 0, r_list[0]
            
            text_features = []
            for text in tqdm.tqdm(test_graph_data.raw_texts, desc=f"Processing {test_graph_name} node texts"):
            # for text in test_graph_data.raw_texts:
                cur_text_feature = model.text_forward(text).cpu()
                text_features.append(cur_text_feature)
            
            if args.virtual_node:
                desc = descriptions[test_graph_name]
                text_features.append(model.text_forward(desc).cpu())
                
            node_embeds = torch.cat(text_features, dim=0).to(device)

            label_features = []
            label_text = test_graph_data.label_name if not args.enrich_label else [one_label + label_descriptions[test_names[idx]][one_label] for one_label in test_graph_data.label_name]
            for text in tqdm.tqdm(label_text, desc=f"Processing {test_graph_name} label texts"):
            # for text in label_text:
                cur_label_feature = model.text_forward(text).cpu()
                label_features.append(cur_label_feature)
            label_embeds = torch.cat(label_features, dim=0).to(device)
            
            for cur_r in r_list:
                test_scores, eval_scores = model.zero_shot_eval(node_embeds, label_embeds, test_graph_data, cur_r)
                if eval_scores[0] > best_eval_acc:
                    best_eval_acc = eval_scores[0]
                    best_test_acc, best_test_f1 = test_scores
                    best_r = cur_r 
                if cur_r == 0:
                    print(f"{test_graph_name} without GNN Test Scores {test_scores}")
            print(f"{test_graph_name} GridSearch Best Test Scores {best_test_acc:.2f} {best_test_f1:.2f} with R={best_r}")
        
        score_dict[test_graph_name] = [best_test_acc, best_test_f1]
        res_list.append([best_test_acc, best_test_f1])
    print(f"Encoder {args.text_encoder}")
    print(f"Test Datasets {test_names}")
    print(f"Test Acc & F1 {res_list}")
    
    os.makedirs("../../results/ZeroG", exist_ok=True)
    with open("../../results/ZeroG/minizerog.txt", "a+") as file:
        score_dict["a-encoder"] = args.text_encoder
        score_dict["search_R"] = args.R_list
        score_dict["a-config"] = f"virtual node {args.virtual_node}; enrich_label {args.enrich_label}"
        # TODO: for ablation study, you have to specify more arguments
        file.write(json.dumps(score_dict) + "\n")
   
    print("=" * 20, "\n\n")
