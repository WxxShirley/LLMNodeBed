import torch.nn as nn
import torch
import copy
import sys
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sklearn.metrics import accuracy_score, f1_score
import torch.nn.functional as F
from peft import LoraModel, LoraConfig
from transformers import BertModel, BertTokenizer, RobertaTokenizer, RobertaModel, AutoTokenizer, AutoModel, AutoModelForMaskedLM,RobertaTokenizer, RobertaModel, T5Tokenizer, T5Model
sys.path.append("../../../../")
from common import MODEL_PATHs as encoder_dict
from common import normalize_adj_matrix, compute_acc_and_f1, mean_pooling_llm


class Text_Lora(nn.Module):
    def __init__(self, args):
        super(Text_Lora, self).__init__()
        self.args = args

        encoder_fullname = encoder_dict[args.text_encoder]

        if args.text_encoder == 'SentenceBert':
            self.tokenizer = AutoTokenizer.from_pretrained(encoder_fullname)
            self.textmodel = AutoModel.from_pretrained(encoder_fullname)
            self.target_modules=["q_lin", "v_lin"]
        #Roberta
        if args.text_encoder == 'roberta':
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base',)
            self.textmodel = RobertaModel.from_pretrained('roberta-base')
            self.target_modules = ["query", "value"]
        elif args.text_encoder == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.textmodel = BertModel.from_pretrained('bert-base-uncased',device_map="auto")
            self.target_modules = ["query", "value"]
        elif args.text_encoder == 'bert-large':
            self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
            self.textmodel = BertModel.from_pretrained('bert-large-uncased',device_map="auto")
            self.target_modules = ["query", "value"]

        elif args.text_encoder == 't5':
            self.tokenizer = T5Tokenizer.from_pretrained("t5-large")
            self.textmodel = T5Model.from_pretrained("t5-large",device_map="auto")
            self.target_modules = ["q", "v"]
        elif args.text_encoder == 'llama':
            self.tokenizer = AutoTokenizer.from_pretrained("/data/yuhanli/Llama-2-7b-hf")
            self.textmodel = AutoModel.from_pretrained("/data/yuhanli/Llama-2-7b-hf", device_map="auto")
            # self.textmodel = AutoModel.from_pretrained("/data/yuhanli/Llama-2-7b-hf")
            self.target_modules = ["q_proj", "v_proj"]
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.config = LoraConfig(
            task_type="SEQ_CLS",
            r=4,
            lora_alpha=16,
            # target_modules=["q_lin", "v_lin"],
            target_modules=self.target_modules,
            lora_dropout=0.1,
        )
        self.lora_model = LoraModel(self.textmodel, self.config, "default")
        self.descriptions = {
            "cora": "The Cora dataset is a fundamental resource in the field of graph learning, particularly within the realm of machine learning research. It represents a network of scientific publications. There are 7 categories in Cora: Theory: This category covers theoretical aspects of machine learning and AI. Reinforcement Learning: This category includes research on reinforcement learning, a type of machine learning where an agent learns to make decisions to achieve a goal, focusing on algorithms, methodologies, and applications in decision-making areas. Genetic Algorithms: This category deals with genetic algorithms, a type of optimization algorithm inspired by natural evolution. Neural Networks: This category focuses on artificial neural networks, a subset of machine learning mimicking the human brain, covering various architectures, training techniques, and applications. Probabilistic Methods: This category pertains to research on probabilistic methods in machine learning, using probability mathematics to handle uncertainty and make predictions. Case Based: This category focuses on case-based reasoning in AI, a method that solves new problems by referring to similar past cases. Rule Learning: This category is about rule-based learning in machine learning, involving the generation of rules for decision-making systems, focusing on algorithms, transparency, and applications in fields requiring interpretability. The average degree of Cora is 4.",
            "citeseer": "The Citeseer dataset is a prominent academic resource in the field of computer science, categorizing publications into six distinct areas. These are Agents, focusing on intelligent agents; Machine Learning (ML), covering all aspects of learning techniques and applications; Information Retrieval (IR), dealing with data and text indexing and retrieval; Databases (DB), related to database management and data mining; Human-Computer Interaction (HCI), emphasizing computer technology interfaces for humans; and Artificial Intelligence (AI), a broad category encompassing general AI theory and applications, excluding certain subfields. The average degree of this graph is 2.",
            "pubmed": "The PubMed dataset comprises three categories: Experimental studies on diabetes mechanisms and therapies, Type 1 Diabetes research focusing on autoimmune processes and treatments, and Type 2 Diabetes studies emphasizing insulin resistance and management strategies. Each category addresses specific aspects of diabetes research, aiding in understanding and treating this complex disease. The average degree of this graph is 4.5.",
            "arxiv": "The arXiv dataset is a notable resource in the field of graph learning, particularly in the area of computer science research. This dataset forms a directed graph representing the citation network among all Computer Science papers on arXiv, as indexed by the Microsoft Academic Graph (MAG). Each node in this network corresponds to a paper, and directed edges indicate citations. The dataset's primary challenge is predicting the 40 subject areas of arXiv CS papers, such as cs.AI, cs.LG, and cs.OS. The task is structured as a 40-class classification problem.",
            "wikics": "The Wiki CS dataset is a comprehensive collection of Wikipedia entries, systematically categorized into ten distinct areas of computer science. These categories include Computational Linguistics, focusing on the intersection of computer science and linguistics; Databases, covering database technologies and theories; Operating Systems, detailing the software that manages computer hardware; Computer Architecture, exploring the design and structure of computer systems; Computer Security, addressing the protection of information systems; Internet Protocols, discussing the rules governing internet data exchange; Computer File Systems, about methods for storing and organizing computer files; Distributed Computing Architecture, concerning computations spread across multiple machines; Web Technology, focusing on the technologies underpinning the web; and Programming Language Topics, which includes various aspects of programming languages. This dataset serves as a valuable resource for understanding diverse computer science topics as represented in Wikipedia, reflecting the breadth and depth of the field.",
            "home": "This graph is of amazon products about home using. There are six categories: Baby Products, Appliances, All Beauty, Luxury Beauty, Office & School Supplies, and Home Improvement. The average degree of this graph is 30.39.",
            "tech": "This graph is of amazon products about technologies. There are three categories: Software, Video Games, and Industrial & Scientific. The average degree of this graph is 87.60.",
            "reddit":"Reddit is also a social network where each node denotes a user, the node features are the content of users’ historically published subreddits, and edges denote whether two users have replied to each other. The prediction task is to classify whether a user is in the top 50% popular (average score of all subreddits).",
            "instagram":"Instagram is a social network where edges represent following relationships, nodes represent users, and the prediction task is to classify commercial and normal users in this network.",
        }

        self.criteria = nn.CrossEntropyLoss()

    def forward(self, data,args):
        # insert a description node
        virtual_node_description = self.descriptions[data.dataset_name]
        all_node_texts = data.raw_text + [virtual_node_description]

        tokens = self.tokenizer(all_node_texts, max_length=256, return_tensors='pt',
                                truncation=True, padding=True).to(self.args.device)
        if args.text_encoder == 'llama':
            outputs = self.lora_model(**tokens, output_hidden_states=True)
            node_embeds = outputs.hidden_states[-1][:, 0, :]
        else:
            node_embeds = self.lora_model(**tokens)[0][:, 0, :]

        tokens = self.tokenizer(data.label_text, max_length=256, return_tensors='pt',
                                truncation=True, padding=True).to(self.args.device)
        if args.text_encoder == 'llama':
            outputs_label = self.lora_model(**tokens, output_hidden_states=True)
            label_embeds = outputs_label.hidden_states[-1][:, 0, :]
        else:
            label_embeds = self.lora_model(**tokens)[0][:, 0, :]

        if self.args.if_norm:
            node_embeds = (node_embeds - node_embeds.mean(0)) / \
                node_embeds.std(0)
            label_embeds = (label_embeds - label_embeds.mean(0)
                            ) / label_embeds.std(0)

        # change the adj matrix
        num_existing_nodes = data.y.shape[0] + 1
        virtual_node_index = data.y.shape[0]
        if data.dataset_name in ["citeseer", "arxiv"]:
            new_edges_to_virtual = [[node_idx, virtual_node_index]
                                    for node_idx in range(num_existing_nodes-1)]
        elif data.dataset_name in ["cora", "pubmed", "wikics", "home", "tech","reddit","instagram"]:
            new_edges_to_virtual = []
            for node_idx in range(num_existing_nodes-1):
                new_edges_to_virtual.append([node_idx, virtual_node_index])
                new_edges_to_virtual.append([virtual_node_index, node_idx])
        new_edge_index = torch.cat([data.edge_index.t(), torch.tensor(
            new_edges_to_virtual, dtype=torch.long).to(self.args.device)], dim=0).t()

        adj_normed = self.normalize_adjacency_matrix(
            new_edge_index, num_existing_nodes)
        for _ in range(self.args.R):
            node_embeds = torch.mm(adj_normed, node_embeds)
        new_node_embeds = node_embeds[:-1, :]
        logits = torch.mm(new_node_embeds, label_embeds.transpose(1, 0))
        logits = torch.div(logits, 1)
        # 11*7 -> 10*7
        # 10*1 forever
        labels = data.y.long().to(self.args.device) if data.y.dim(
        ) == 1 else data.y.squeeze(1).long().to(self.args.device)
        CL_loss = self.criteria(logits, labels)

        return CL_loss

    def zero_shot_eval(self, node_embeds, label_embeds, data):
        if self.args.if_norm:
            node_embeds = (node_embeds - node_embeds.mean(0)) / \
                node_embeds.std(0)
            label_embeds = (label_embeds - label_embeds.mean(0)
                            ) / label_embeds.std(0)

        # change the adj matrix
        num_existing_nodes = data.y.shape[0] + 1
        virtual_node_index = data.y.shape[0]
        if self.args.test_data in ["citesee"]:
            new_edges_to_virtual = [[node_idx, virtual_node_index]
                                    for node_idx in range(num_existing_nodes-1)]
        elif self.args.test_data in ["cora", "pubmed", "citeseer", "arxiv", "wikics", "facebook", 'home', 'tech','reddit','instagram']:
            new_edges_to_virtual = []
            for node_idx in range(num_existing_nodes-1):
                new_edges_to_virtual.append([node_idx, virtual_node_index])
                new_edges_to_virtual.append([virtual_node_index, node_idx])
        new_edge_index = torch.cat([data.edge_index.t(), torch.tensor(
            new_edges_to_virtual, dtype=torch.long).to(self.args.device)], dim=0).t()
        adj_normed = self.normalize_adjacency_matrix(
            new_edge_index, num_existing_nodes)

        labels = data.y.long().to(self.args.device)
        score_dict = {}
        test_node_embeds = copy.deepcopy(node_embeds)
        
        for cur_R in range(self.args.R):
            node_embeds = torch.mm(adj_normed, node_embeds)
            test_node_embeds = torch.mm(adj_normed,test_node_embeds)
            test_label_embeds = copy.deepcopy(label_embeds)
            test_label_embeds /= test_label_embeds.norm(dim=-1, keepdim=True)
            comp_node_embeds = copy.deepcopy(test_node_embeds[:-1, :])
            comp_node_embeds /= comp_node_embeds.norm(dim=-1, keepdim = True)
            cur_dists = torch.einsum('bn,cn->bc', comp_node_embeds, test_label_embeds)
            preds = torch.argmax(cur_dists, dim=1)
            test_acc, test_f1, _ = compute_acc_and_f1(labels[data.test_mask].cpu(), preds[data.test_mask].cpu())
            score_dict[cur_R] = [test_acc, test_f1]

        node_embeds = node_embeds[:-1, :]
        node_embeds /= node_embeds.norm(dim=-1,
                                        keepdim=True).to(self.args.device)
        label_embeds /= label_embeds.norm(dim=-1, keepdim=True)
        dists = torch.einsum('bn,cn->bc', node_embeds, label_embeds)
        preds = torch.argmax(dists, dim=1)
        labels = data.y.long().to(self.args.device)
        if len(data.test_mask) == 10 :
            data.test_mask = data.test_mask[0]
        test_mask = data.test_mask
        test_acc, test_f1, test_weight_f1 = compute_acc_and_f1(labels[test_mask].cpu(), preds[test_mask].cpu())
        return [test_acc,test_f1],score_dict


    def normalize_adjacency_matrix(self, edge_index, num_nodes):
        # edge_index = data.edge_index
        # num_nodes = data.y.shape[0]

        edge_index_self_loops = torch.stack(
            [torch.arange(num_nodes), torch.arange(num_nodes)], dim=0).to(self.args.device)
        edge_index = torch.cat([edge_index, edge_index_self_loops], dim=1)

        adj = torch.sparse_coo_tensor(edge_index, torch.ones(
            edge_index.shape[1]).to(self.args.device), (num_nodes, num_nodes))

        deg = torch.sparse.sum(adj, dim=1).to_dense()
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        adj_normalized = adj
        # adj_normalized = adj.coalesce()
        deg_inv_sqrt_mat = torch.sparse_coo_tensor(torch.arange(num_nodes).unsqueeze(
            0).repeat(2, 1).to(self.args.device), deg_inv_sqrt, (num_nodes, num_nodes))
        adj_normalized = torch.sparse.mm(
            deg_inv_sqrt_mat, torch.sparse.mm(adj_normalized, deg_inv_sqrt_mat))

        return adj_normalized
