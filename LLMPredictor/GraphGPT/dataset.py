from torch.utils.data import Dataset
import numpy as np
import random
import torch
import sys 
from torch_geometric.utils import k_hop_subgraph
sys.path.append("../..")
from common import prepare_edge_list, load_graph_dataset_for_zerog


# TODO: this part is duplicate with LLaGA, consider reformulate both in `common`
classes = {
    "arxiv": ["cs.NA(Numerical Analysis)", "cs.MM(Multimedia)", "cs.LO(Logic in Computer Science)", "cs.CY(Computers and Society)", "cs.CR(Cryptography and Security)", "cs.DC(Distributed, Parallel, and Cluster Computing)", "cs.HC(Human-Computer Interaction)", "cs.CE(Computational Engineering, Finance, and Science)", "cs.NI(Networking and Internet Architecture)", "cs.CC(Computational Complexity)", "cs.AI(Artificial Intelligence)", "cs.MA(Multiagent Systems)", "cs.GL(General Literature)", "cs.NE(Neural and Evolutionary Computing)", "cs.SC(Symbolic Computation)", "cs.AR(Hardware Architecture)", "cs.CV(Computer Vision and Pattern Recognition)", "cs.GR(Graphics)", "cs.ET(Emerging Technologies)", "cs.SY(Systems and Control)", "cs.CG(Computational Geometry)", "cs.OH(Other Computer Science)", "cs.PL(Programming Languages)", "cs.SE(Software Engineering)", "cs.LG(Machine Learning)", "cs.SD(Sound)", "cs.SI(Social and Information Networks)", "cs.RO(Robotics)", "cs.IT(Information Theory)", "cs.PF(Performance)", "cs.CL(Computational Complexity)", "cs.IR(Information Retrieval)", "cs.MS(Mathematical Software)", "cs.FL(Formal Languages and Automata Theory)", "cs.DS(Data Structures and Algorithms)", "cs.OS(Operating Systems)", "cs.GT(Computer Science and Game Theory)", "cs.DB(Databases)", "cs.DL(Digital Libraries)", "cs.DM(Discrete Mathematics)"],
    "cora": ['Rule_Learning', 'Neural_Networks', 'Case_Based', 'Genetic_Algorithms', 'Theory', 'Reinforcement_Learning', 'Probabilistic_Methods'],
    "pubmed": ['Experimentally induced diabetes', 'Type 1 diabetes', 'Type 2 diabetes'],
    "citeseer": ['Agents', 'ML (Machine Learning)', 'IR (Information Retrieval)', 'DB (Databases)', 'HCI (Human-Computer Interaction)', 'AI (Artificial Intelligence)'],
    "wikics": ['Computational Linguistics', 'Databases', 'Operating Systems', 'Computer Architecture', 'Computer Security', 'Internet Protocols', 'Computer File Systems', 'Distributed Computing Architecture', 'Web Technology', 'Programming Language Topics'],
    "reddit": ['Normal Users', 'Popular Users'],
    "instagram": ['Normal Users', 'Commercial Users']
}


class TextGraphGroundDataset(Dataset):
    def __init__(self, graph_data, num_sampled_neighbors):
        self.graph_data = graph_data 
        self.num_nodes = graph_data.num_nodes 
        
        self.edge_dict = prepare_edge_list(graph_data.edge_index.detach().cpu(), self.num_nodes)
        self.num_sampled_neighbors = num_sampled_neighbors
        
    def __len__(self):
        return self.num_nodes
    
    def __getitem__(self, idx):
        sampled_neighbors = [np.random.choice(self.edge_dict[idx], replace=True) for _ in range(self.num_sampled_neighbors)]
        neighbor_texts = [self.graph_data.raw_texts[neigh_id] for neigh_id in sampled_neighbors]
        return {
            "id": idx, 
            "text": self.graph_data.raw_texts[idx],
            "neighbor_text": neighbor_texts, 
            "neighbor_ids": np.array(sampled_neighbors)
        }


MATCHING_TEMPLATES = {
    "academic_network": "Given a sequence of graph tokens <graph> that constitue a subgraph of a citation graph, where the first token represents the central node of the subgraph, and the remaining nodes represent the first or second order neighbors of the central node. Each graph token contains the title and abstract information of the paper at this node. Here is a list of paper titles: {{paper_titles}}. Please reorder the list of papers according to the order of graph tokens (i.e., complete the matching of graph tokens and papers).",
    "social_network": "Given a sequence of graph tokens <graph> that constitue a subgraph of a social network, where the first token represents the central node (user) of the subgraph, and the remaining nodes represent the first or second order neighbors of the central node. Each graph token contains the profile description of the user represented by this node. Here is a list of user profile descriptions: {{user_profiles}}. Please reorder the list of users according to the order of the graph tokens (i.e., complete the matching of graph tokens and users).",
}


def fetch_title(txt, max_length=512):
    title = None
    if ":" in txt:
        title= txt.split(":")[0]
    title= txt.split(".")[0]
    
    return title[:max_length]


# Example Data: https://huggingface.co/datasets/Jiabin99/graph-matching
class GraphMatchingDataset(Dataset):
    def __init__(self, graph_data, k_hop=1, num_sampled_neighbors=8, graph_type="academic_network", sample_times=1):
        self.graph_data = graph_data 
        self.num_nodes = graph_data.num_nodes 
        self.k_hop = k_hop
        self.num_sampled_neighbors = num_sampled_neighbors
        self.sample_times = sample_times
        self.query_template = MATCHING_TEMPLATES[graph_type]
        self.graph_type = graph_type
        
        self.all_data = self._prepare_matching_data()
    
    def __len__(self):
        return len(self.all_data)
    
    def __getitem__(self, idx):
        return self.all_data[idx]
        
    def _prepare_matching_data(self):
        data_samples = []
        for node in range(self.num_nodes):
            neighbors, _, _, _ = k_hop_subgraph(node, num_hops=self.k_hop, edge_index=self.graph_data.edge_index)
            neighbors = neighbors.numpy()
            
            for _ in range(self.sample_times):
                subset = np.random.choice(neighbors, size=self.num_sampled_neighbors).tolist()
                subset = list(set(subset))
                if node not in subset:
                    subset = [node] + subset[:-1] 
                else:
                    target_idx = subset.index(node)
                    subset[target_idx] = subset[0]
                    subset[0] = node
                
                if len(subset) < self.num_sampled_neighbors:
                    pad_length = self.num_sampled_neighbors - len(subset)
                    subset = subset + [node] * pad_length
                    
                assert subset[0] == node 
                
                texts = []
                for token_id in range(len(subset)): 
                    raw_text = fetch_title(self.graph_data.raw_texts[subset[token_id]])
                    texts.append([token_id, raw_text]) # origin_id in graph-tokens, corresponding text
                
                # Re-order the texts 
                random.shuffle(texts)
                tokenid2text_mapping = {pairs[0]+1: pairs[1] for text_id, pairs in enumerate(texts)}
                query_graph_texts = ". ".join([f"{text_id+1}. {pairs[1]}" for text_id, pairs in enumerate(texts)])
                
                if self.graph_type == "academic_network":
                    cur_query = self.query_template.replace("{{paper_titles}}", query_graph_texts)
                    cur_response = ". ".join([f"Graph token {k} corresponds to paper {tokenid2text_mapping[k]}" for k in sorted(tokenid2text_mapping.keys()) ])
                    cur_response = "Based on the given graph tokens and the list of paper titles, we obtain the matching of graph tokens and papers as follows: " + cur_response
                elif self.graph_type == "social_network":
                    cur_query = self.query_template.replace("{{user_profiles}}", query_graph_texts)
                    cur_response = ". ".join([f"Graph token {k} corresponds to user {tokenid2text_mapping[k]}" for k in sorted(tokenid2text_mapping.keys()) ])
                    cur_response = "Based on the given graph tokens and the descriptions of users, we obtain the matching of graph tokens and users as follows: " + cur_response
                
                sample = {
                    "id": node,
                    "nodes": torch.LongTensor(subset),
                    "query": cur_query,
                    "label": cur_response
                }
                data_samples.append(sample)
                
        return data_samples


CLASSIFICATION_TEMPLATES = {
    "cora": 'Given a citation graph: \n<graph>\nwhere the 0th node is the target paper, with the following information: \n{{raw_text}}\n Question: Which of the following specific research does this paper belong to: {{label_names}}. Directly give the full name of the most likely category of this paper.',
    "citeseer": 'Given a citation graph: \n<graph>\nwhere the 0th node is the target paper, with the following information: \n{{raw_text}}\n Question: Which of the following specific research does this paper belong to: {{label_names}}. Directly give the full name of the most likely category of this paper.',
    "pubmed" : 'Given a citation graph: \n<graph>\nwhere the 0th node is the target paper, with the following information: \n{{raw_text}}\n Question: Which of the following specific diabetes research does this paper belong to: {{label_names}}. Directly give the full name of the most likely category of this paper.', 
    "wikics": 'Given a citation graph: \n<graph>\nwhere the 0th node is the target paper, with the following information: \n{{raw_text}}\n Question: Which of the following specific research does this paper belong to: {{label_names}}. Directly give the full name of the most likely category of this paper.', 
    "arxiv": 'Given a citation graph: \n<graph>\nwhere the 0th node is the target paper, with the following information: \n{{raw_text}}\n Question: Which of the following arXiv CS sub-category does this paper belong to: {{label_names}}. Directly give the most likely arXiv CS sub-categories of this paper.', 
    "reddit": "Given a social network: \n<graph>\nwhere the 0th node is the target user, with the following information: \n{{raw_text}}\n Question: We need to classify the center user into 2 classes: {{label_names}}. Directly tell me which class the center user belongs to. ",
    "instagram": "Given a social network: \n<graph>\nwhere the 0th node is the target user, with the following information: \n{{raw_text}}\n Question: We need to classify the center user into 2 classes: {{label_names}}. Directly tell me which class the center user belongs to. "
}


# Example Data: https://huggingface.co/datasets/Jiabin99/Arxiv-PubMed-mix-NC-LP
class GraphInstructionTuningDataset(Dataset):
    def __init__(self, graph_data, k_hop=1, maximum_neighbors=4, dataset_name="cora", data_type="train"):
        self.graph_data = graph_data
        self.num_nodes = graph_data.num_nodes
        self.k_hop = k_hop 
        self.maximum_neighbors = maximum_neighbors
        self.label_names = classes[dataset_name]
        self.data_type = data_type
        
        label_names = ", ".join(classes[dataset_name])
        self.query_prompt = CLASSIFICATION_TEMPLATES[dataset_name].replace("{{label_names}}", label_names)
        self.data_list = self.format_data()
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        return self.data_list[index]
    
    def format_data(self):
        focus_mask = {"train": self.graph_data.train_mask, "val": self.graph_data.val_mask, "test": self.graph_data.test_mask}[self.data_type]
        focus_nodes = focus_mask.nonzero(as_tuple=False).squeeze().detach().cpu().numpy().tolist()
        
        available_data_list = []
        for cur_node in focus_nodes:
            neighbors, _, _, _ = k_hop_subgraph(cur_node, num_hops=self.k_hop, edge_index=self.graph_data.edge_index)
            neighbors = neighbors.numpy().tolist()
        
            if len(neighbors) > self.maximum_neighbors:
                neighbors = np.random.choice(np.array(neighbors), size=self.maximum_neighbors).tolist()
                neighbors = [cur_node] + neighbors
            else: 
                pad_length = self.maximum_neighbors - len(neighbors) 
                neighbors = [cur_node] + neighbors + [cur_node] * pad_length
            
            assert cur_node == neighbors[0]
        
            cur_query = self.query_prompt.replace("{{raw_text}}", self.graph_data.raw_texts[cur_node])
            cur_response = self.label_names[self.graph_data.y[cur_node].item()]
        
            available_data_list.append({
                "id": cur_node, 
                "nodes": torch.LongTensor(neighbors),
                "query": cur_query,
                "label": cur_response
            })  
        
        return available_data_list   


# if __name__ == "__main__":
#     g_data = load_graph_dataset_for_zerog(dataset_name="cora", device=torch.device("cpu"))
#     gset = GraphInstructionTuningDataset(g_data)
    
#     random_ids = np.random.choice(np.array(len(gset)), size=2).tolist()
    
#     print(len(gset))
#     for node in random_ids:
#         print(gset[node], "\n")
    