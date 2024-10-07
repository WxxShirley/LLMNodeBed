import torch 
import argparse
import sys
import tqdm
import torch.nn as nn 
import json
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
sys.path.append("../")
from common import compute_acc_and_f1, mean_pooling, mean_pooling_llm, normalize_adj_matrix
from common import load_graph_dataset


encoder_dict = {
    "MiniLM": "sentence-transformers/all-MiniLM-L6-v2", # 22M
    "SentenceBert": "sentence-transformers/multi-qa-distilbert-cos-v1", # 66M
    "roberta": "sentence-transformers/all-roberta-large-v1", # 355M
    "e5-large": "intfloat/e5-large-v2", # 355M

    # TODO: add Qwen2.5-3B, Llama3-8B, etc
}


# descriptions for introduced virtual nodes
descriptions = {
    "cora": "The Cora dataset is a fundamental resource in the field of graph learning, particularly within the realm of machine learning research. It represents a network of scientific publications. There are 7 categories in Cora: Theory: This category covers theoretical aspects of machine learning and AI. Reinforcement Learning: This category includes research on reinforcement learning, a type of machine learning where an agent learns to make decisions to achieve a goal, focusing on algorithms, methodologies, and applications in decision-making areas. Genetic Algorithms: This category deals with genetic algorithms, a type of optimization algorithm inspired by natural evolution. Neural Networks: This category focuses on artificial neural networks, a subset of machine learning mimicking the human brain, covering various architectures, training techniques, and applications. Probabilistic Methods: This category pertains to research on probabilistic methods in machine learning, using probability mathematics to handle uncertainty and make predictions. Case Based: This category focuses on case-based reasoning in AI, a method that solves new problems by referring to similar past cases. Rule Learning: This category is about rule-based learning in machine learning, involving the generation of rules for decision-making systems, focusing on algorithms, transparency, and applications in fields requiring interpretability. The average degree of Cora is 4.",
    "citeseer": "The Citeseer dataset is a prominent academic resource in the field of computer science, categorizing publications into six distinct areas. These are Agents, focusing on intelligent agents; Machine Learning (ML), covering all aspects of learning techniques and applications; Information Retrieval (IR), dealing with data and text indexing and retrieval; Databases (DB), related to database management and data mining; Human-Computer Interaction (HCI), emphasizing computer technology interfaces for humans; and Artificial Intelligence (AI), a broad category encompassing general AI theory and applications, excluding certain subfields. The average degree of this graph is 2.",
    "pubmed": "The PubMed dataset comprises three categories: Experimental studies on diabetes mechanisms and therapies, Type 1 Diabetes research focusing on autoimmune processes and treatments, and Type 2 Diabetes studies emphasizing insulin resistance and management strategies. Each category addresses specific aspects of diabetes research, aiding in understanding and treating this complex disease. The average degree of this graph is 4.5.",
    "arxiv": "The arXiv dataset is a notable resource in the field of graph learning, particularly in the area of computer science research. This dataset forms a directed graph representing the citation network among all Computer Science papers on arXiv, as indexed by the Microsoft Academic Graph (MAG). Each node in this network corresponds to a paper, and directed edges indicate citations. The dataset's primary challenge is predicting the 40 subject areas of arXiv CS papers, such as cs.AI, cs.LG, and cs.OS. The task is structured as a 40-class classification problem.",
    "wikics": "The Wiki CS dataset is a comprehensive collection of Wikipedia entries, systematically categorized into ten distinct areas of computer science. These categories include Computational Linguistics, focusing on the intersection of computer science and linguistics; Databases, covering database technologies and theories; Operating Systems, detailing the software that manages computer hardware; Computer Architecture, exploring the design and structure of computer systems; Computer Security, addressing the protection of information systems; Internet Protocols, discussing the rules governing internet data exchange; Computer File Systems, about methods for storing and organizing computer files; Distributed Computing Architecture, concerning computations spread across multiple machines; Web Technology, focusing on the technologies underpinning the web; and Programming Language Topics, which includes various aspects of programming languages. This dataset serves as a valuable resource for understanding diverse computer science topics as represented in Wikipedia, reflecting the breadth and depth of the field.",
    "reddit":"Reddit is also a social network where each node denotes a user, the node features are the content of users’ historically published subreddits, and edges denote whether two users have replied to each other. The prediction task is to classify whether a user is in the top 50% popular (average score of all subreddits).",
    "instagram":"Instagram is a social network where edges represent following relationships, nodes represent users, and the prediction task is to classify commercial and normal users in this network.",
}


class MiniZeroG(nn.Module):
    def __init__(self, args):
        super(MiniZeroG, self).__init__()
        self.args = args 
        self.device = device
        
        assert args.text_encoder in encoder_dict.keys()
        encoder_fullname = encoder_dict[args.text_encoder]

        if args.text_encoder in ["SentenceBert", "roberta", "e5-large", "MiniLM"]:
            self.tokenizer = AutoTokenizer.from_pretrained(encoder_fullname)
            self.text_model = AutoModel.from_pretrained(encoder_fullname).to(self.device)
            if args.text_encoder in ['SentenceBert']:
                self.target_modules = ["q_lin", "v_lin"]
            elif args.text_encoder in ["roberta", "e5-large", "MiniLM"]:
                self.target_modules = ["query", "value"]
        else:
            # Load local LLM
            self.tokenizer = AutoTokenizer.from_pretrained(encoder_fullname)
            self.text_model = AutoModelForCausalLM.from_pretrained(encoder_fullname, torch_dtype=torch.float16).to(self.device)
            self.target_modules = ["q_proj", "v_proj"]
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.criteria = nn.CrossEntropyLoss()

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
    
    def zero_shot_eval(self, node_embeds, label_embeds, data, dataset_name):
        if self.args.if_norm:
            # print(f"hit feature normalization")
            node_embeds = (node_embeds - node_embeds.mean(0)) / node_embeds.std(0)
            label_embeds = (label_embeds - label_embeds.mean(0)) / label_embeds.std(0)
        
        # TODO: add ablation study to check whether the introduced virtual node is useful
        num_exist_nodes = data.y.shape[0] + 1 
        virtual_node_idx = data.y.shape[0]
        if dataset_name in ["citeseer"]:
            new_edges_to_virtual = [[node_idx, virtual_node_idx] for node_idx in range(num_exist_nodes-1)]
        elif dataset_name in ["cora", "pubmed", "wikics", "reddit", "instagram"]:
            new_edges_to_virtual = []
            for node_idx in range(num_exist_nodes-1):
                new_edges_to_virtual.extend([[node_idx, virtual_node_idx], [virtual_node_idx, node_idx]])
        
        new_edge_index = torch.cat([data.edge_index.t(), torch.tensor(new_edges_to_virtual, dtype=torch.long).to(self.device)], dim=0).t()

        adj_normed = normalize_adj_matrix(new_edge_index, num_exist_nodes, self.device)

        for _ in range(self.args.test_R):
            node_embeds = torch.mm(adj_normed, node_embeds)
        node_embeds = node_embeds[:-1, :]
        node_embeds /= node_embeds.norm(dim=-1, keepdim=True).to(self.device)
        label_embeds /= label_embeds.norm(dim=-1, keepdim=True)

        dists = torch.einsum('bn,cn->bc', node_embeds, label_embeds)
        preds = torch.argmax(dists, dim=1)

        acc, f1 = compute_acc_and_f1(preds[data.test_mask].cpu(), data.y[data.test_mask].cpu())
        del node_embeds, label_embeds, dists, preds
        
        return [acc, f1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_datasets", type=str, default="cora,citeseer,pubmed,wikics,instagram,reddit")
    parser.add_argument("--if_norm", action="store_true", default=True, help="Indicator of normalization")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--text_encoder", type=str, default="SentenceBert", help="Type of text encoder")
    parser.add_argument("--test_R", type=int, default=5)

    args = parser.parse_args()
    device = torch.device(args.device)
    print("=" * 20)

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
            text_features = []
            for text in tqdm.tqdm(test_graph_data.raw_texts, desc=f"Processing {test_graph_name} node texts"):
                cur_text_feature = model.text_forward(text).cpu()
                text_features.append(cur_text_feature)
            
            desc = descriptions[test_graph_name]
            text_features.append(model.text_forward(desc).cpu())
            node_embeds = torch.cat(text_features, dim=0).to(device)

            label_features = []
            for text in tqdm.tqdm(test_graph_data.label_name, desc=f"Processing {test_graph_name} label texts"):
                cur_label_feature = model.text_forward(text).cpu()
                label_features.append(cur_label_feature)
            label_embeds = torch.cat(label_features, dim=0).to(device)
            
            res = model.zero_shot_eval(node_embeds, label_embeds, test_graph_data, test_graph_name)
        
        score_dict[test_graph_name] = res 
        res_list.append(res)
    print(f"Encoder {args.text_encoder}")
    print(f"Test Datasets {test_names}")
    print(f"Test Acc & F1 {res_list}")

    with open("../results/minizerog.txt", "a+") as file:
        score_dict["encoder"] = args.text_encoder
        score_dict["test_R"] = args.test_R
        # TODO: for ablation study, you have to specify more arguments
        file.write(json.dumps(score_dict) + "\n")
   
    print("=" * 20)
