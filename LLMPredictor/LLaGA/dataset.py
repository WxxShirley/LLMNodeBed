from torch.utils.data import Dataset
import copy 
from constant import *
import random
import torch 
import sys 
sys.path.append("../..")
from common import normalize_adj_matrix
import numpy as np
import scipy.sparse as sp


descriptions = {
    "cora": "Given a node-centered graph: <graph>, each node represents a paper, we need to classify the center node into 7 classes: Case_Based, Genetic_Algorithms, Neural_Networks, Probabilistic_Methods, Reinforcement_Learning, Rule_Learning, Theory, please tell me which class the center node belongs to?",
    "pubmed": "Given a node-centered graph: <graph>, each node represents a paper about Diabetes, we need to classify the center node into 3 classes: Experimentally induced diabetes, Type 1 diabetes, Type 2 diabetes, please tell me which class the center node belongs to?",
    "arxiv": "Given a node-centered graph: <graph>, we need to classify the center node into 40 classes: cs.NA(Numerical Analysis), cs.MM(Multimedia), cs.LO(Logic in Computer Science), cs.CY(Computers and Society), cs.CR(Cryptography and Security), cs.DC(Distributed, Parallel, and Cluster Computing), cs.HC(Human-Computer Interaction), cs.CE(Computational Engineering, Finance, and Science), cs.NI(Networking and Internet Architecture), cs.CC(Computational Complexity), cs.AI(Artificial Intelligence), cs.MA(Multiagent Systems), cs.GL(General Literature), cs.NE(Neural and Evolutionary Computing), cs.SC(Symbolic Computation), cs.AR(Hardware Architecture), cs.CV(Computer Vision and Pattern Recognition), cs.GR(Graphics), cs.ET(Emerging Technologies), cs.SY(Systems and Control), cs.CG(Computational Geometry), cs.OH(Other Computer Science), cs.PL(Programming Languages), cs.SE(Software Engineering), cs.LG(Machine Learning), cs.SD(Sound), cs.SI(Social and Information Networks), cs.RO(Robotics), cs.IT(Information Theory), cs.PF(Performance), cs.CL(Computational Complexity), cs.IR(Information Retrieval), cs.MS(Mathematical Software), cs.FL(Formal Languages and Automata Theory), cs.DS(Data Structures and Algorithms), cs.OS(Operating Systems), cs.GT(Computer Science and Game Theory), cs.DB(Databases), cs.DL(Digital Libraries), cs.DM(Discrete Mathematics), please tell me which class the center node belongs to?",
    "citeseer": "Given a node-centered graph: <graph>, each node represents a paper, we need to classify the center node into 6 classes: Agents, ML (Machine Learning), IR (Information Retrieval), DB (Databases), HCI (Human-Computer Interaction), AI (Artificial Intelligence), please tell me which class the center node belongs to?",
    "wikics": "Given a node-centered graph: <graph>, each node represents an entity, we need to classify the center node into 10 classes: Computational Linguistics, Databases, Operating Systems, Computer Architecture, Computer Security, Internet Protocols, Computer File Systems, Distributed Computing Architecture, Web Technology, Programming Language Topics, please tell me which class the center node belongs to?",
    "reddit": "Given a node-centered graph: <graph>, each node represents an user, we need to classify the center node into 2 classes: Normal Users and Popular Users, please tell me which class the center node belongs to?",
    "instagram": "Given a node-centered graph: <graph>, each node represents an user, we need to classify the center node into 2 classes: Normal Users and Commercial Users, please tell me which class the center node belongs to?",
}

classes = {
    "arxiv": ["cs.NA(Numerical Analysis)", "cs.MM(Multimedia)", "cs.LO(Logic in Computer Science)", "cs.CY(Computers and Society)", "cs.CR(Cryptography and Security)", "cs.DC(Distributed, Parallel, and Cluster Computing)", "cs.HC(Human-Computer Interaction)", "cs.CE(Computational Engineering, Finance, and Science)", "cs.NI(Networking and Internet Architecture)", "cs.CC(Computational Complexity)", "cs.AI(Artificial Intelligence)", "cs.MA(Multiagent Systems)", "cs.GL(General Literature)", "cs.NE(Neural and Evolutionary Computing)", "cs.SC(Symbolic Computation)", "cs.AR(Hardware Architecture)", "cs.CV(Computer Vision and Pattern Recognition)", "cs.GR(Graphics)", "cs.ET(Emerging Technologies)", "cs.SY(Systems and Control)", "cs.CG(Computational Geometry)", "cs.OH(Other Computer Science)", "cs.PL(Programming Languages)", "cs.SE(Software Engineering)", "cs.LG(Machine Learning)", "cs.SD(Sound)", "cs.SI(Social and Information Networks)", "cs.RO(Robotics)", "cs.IT(Information Theory)", "cs.PF(Performance)", "cs.CL(Computational Complexity)", "cs.IR(Information Retrieval)", "cs.MS(Mathematical Software), cs.FL(Formal Languages and Automata Theory)", "cs.DS(Data Structures and Algorithms)", "cs.OS(Operating Systems)", "cs.GT(Computer Science and Game Theory)", "cs.DB(Databases)", "cs.DL(Digital Libraries)", "cs.DM(Discrete Mathematics)"],
    "cora": ['Rule_Learning', 'Neural_Networks', 'Case_Based', 'Genetic_Algorithms', 'Theory', 'Reinforcement_Learning', 'Probabilistic_Methods'],
    "pubmed": ['Experimentally induced diabetes', 'Type 1 diabetes', 'Type 2 diabetes'],
    "citeseer": ['Agents', 'ML (Machine Learning)', 'IR (Information Retrieval)', 'DB (Databases)', 'HCI (Human-Computer Interaction)', 'AI (Artificial Intelligence)'],
    "wikics": ['Computational Linguistics', 'Databases', 'Operating Systems', 'Computer Architecture', 'Computer Security', 'Internet Protocols', 'Computer File Systems', 'Distributed Computing Architecture', 'Web Technology', 'Programming Language Topics'],
    "reddit": ['Normal Users', 'Popular Users'],
    "instagram": ['Normal Users', 'Commercial Users']
}


SYSTEM_PROMPT = "You are a helpful language and graph assistant. You can understand the graph content provided by the user and assist with the node classification task by outputting the label that is most likely to apply to the node."


class LLaGADataset(Dataset):
    def __init__(self, args, graph_data, data_type="train", repeats=1):
        super(LLaGADataset, self).__init__() 
        
        self.dataset_name = args.dataset 
        self.graph = graph_data 
        self.data_type = data_type 
        self.repeats = repeats 
        
        self.neighbor_template = args.neighbor_template 
        self.k_hop, self.sample_size = args.k_hop, args.sample_size
        self.data_list = self.format_data()
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        return self.data_list[index]
       
    def format_data(self):
        focus_mask = {"train": self.graph.train_mask, "val": self.graph.val_mask, "test": self.graph.test_mask}[self.data_type]
        focus_nodes = focus_mask.nonzero(as_tuple=False).squeeze().detach().cpu().numpy().tolist()
        
        # print(focus_nodes[:10])
        if self.neighbor_template == "ND":
            edge_list = prepare_edge_list(self.graph.edge_index.to("cpu"), self.graph.num_nodes)
        
        available_data_list = []
        for node in focus_nodes:
            sample = {
                "id": node, 
                "query": f"{SYSTEM_PROMPT}\n{descriptions[self.dataset_name]}",
                "origin_txt": self.graph.raw_texts[node],
                "label": classes[self.dataset_name][self.graph.y[node].item()], 
            }
            if self.neighbor_template == "ND":
                neigh_seq = get_node_neighbor_detail(edge_list, node, k_hop=self.k_hop, sample_size=self.sample_size) 
                sample["graph"] = neigh_seq
            
            available_data_list.extend([sample] * self.repeats)
        
        return available_data_list


def prepare_edge_list(edge_index, num_nodes):
    """Convert [torch.LongTensor] edge_index into [List] edge_list"""
    row, col = edge_index
    edge_list = [[] for _ in range(num_nodes)] 
    
    row, col = row.numpy(), col.numpy()
    for i in range(row.shape[0]):
        edge_list[row[i]].append(int(col[i]))
    return edge_list 


def get_node_neighbor_detail(edge_list, node_idx, k_hop, sample_size, avoid_idx=None):
    assert k_hop > 0 and sample_size > 0 
    neighbors = [[node_idx]]
    for t in range(k_hop):
        last_hop = neighbors[-1]
        current_hop = [] 
        for i in last_hop:
            if i == DEFAULT_GRAPH_PAD_ID: 
                current_hop.extend([DEFAULT_GRAPH_PAD_ID]*sample_size)
                continue 
            
            node_neighbor = copy.copy(edge_list[i])
            if t == 0 and avoid_idx is not None and avoid_idx in node_neighbor:
                node_neighbor.remove(avoid_idx)
            if len(node_neighbor) > sample_size:
                sampled_neighbor = random.sample(node_neighbor, sample_size)
            else:
                sampled_neighbor = node_neighbor + [DEFAULT_GRAPH_PAD_ID] * (sample_size - len(node_neighbor))
            
            current_hop.extend(sampled_neighbor)
        neighbors.append(current_hop)
        
    node_sequence = [n for hop in neighbors for n in hop]
    return node_sequence


def build_laplacian_emb(k_hop=2, sample_size=10):
    n = int(((sample_size ** (k_hop+1)) - 1) / (sample_size - 1))
    edge_row, edge_col = [], []
    last_hop_start = last_hop_end = 0 
    
    for i in range(k_hop):
        edge_row.extend([x for x in range(last_hop_start, last_hop_end + 1) for _ in range(sample_size)])
        edge_col.extend(list(range(last_hop_start * sample_size + 1, last_hop_end * sample_size + sample_size +1)))
        last_hop_start = last_hop_start * sample_size + 1 
        last_hop_end = last_hop_end * sample_size + sample_size
    
    edge_row = np.array(edge_row)
    edge_col = np.array(edge_col)
    A = sp.coo_matrix((np.array([1]*len(edge_row)),(edge_col, edge_row)), shape=(n,n))
    L = sp.eye(n) - A

    _, EigVec = np.linalg.eig(L.toarray())

    PE = torch.FloatTensor(EigVec)
    print(f"[DATA] Build LapLacian Embedding Matrix {PE.shape}")
    return PE


def build_hopfield_emb(x: torch.FloatTensor, edge_index: torch.LongTensor, n_layers:int):
    num_nodes = x.shape[0]
    norm_adj = normalize_adj_matrix(edge_index, num_nodes, edge_index.device)
    
    all_embeds = [x]
    for _ in range(n_layers):
        x = torch.sparse.mm(norm_adj, x)
        all_embeds.append(x)
        # print(all_embeds[-1])
    return all_embeds


# Test Codes
# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser()
    
#     parser.add_argument("--dataset", type=str, default="cora")
#     parser.add_argument("--neighbor_template", default="HO", choices=["ND", "HO"])
#     parser.add_argument("--k_hop", type=int, default=2)
#     parser.add_argument("--sample_size", type=int, default=10)
#     args = parser.parse_args() 
    
#     device = torch.device("cuda:0")
#     graph_data = torch.load(f"../../datasets/{args.dataset}.pt").to(device)
#     graph_data.x = torch.load(f"../../datasets/roberta/{args.dataset}.pt").to(device)
#     graph_data.train_mask, graph_data.val_mask, graph_data.test_mask = graph_data.train_mask[0], graph_data.val_mask[0], graph_data.test_mask[0]
#     train_dataset = LLaGADataset(args, graph_data=graph_data, data_type="train")
    
#     print(train_dataset[0], "\n")
#     print(train_dataset[1])
#     build_hopfield_emb(graph_data.x, graph_data.edge_index, 2)
    