"""data analysis"""
import torch 
from collections import defaultdict
import random


def load_target_graph(dataset_name):
    graph_data = torch.load(f"./datasets/{dataset_name}.pt").to(device)

    if len(graph_data.train_mask) == 10:
        graph_data.train_mask, graph_data.val_mask, graph_data.test_mask = graph_data.train_mask[0], graph_data.val_mask[0], graph_data.test_mask[0]

    neighbor_dict = defaultdict(list)
    edges = graph_data.edge_index.t().cpu().numpy().tolist()

    for (u, v) in edges:
        neighbor_dict[v].append(u)
        neighbor_dict[u].append(v)
     
    for centric_node, neighbors in neighbor_dict.items():
        # remove self-loop and possible duplicate neighbors
        neighbors = list(set(neighbors) - {centric_node})
        neighbor_dict[centric_node] = neighbors
        
    return graph_data, neighbor_dict


def compute_homo_ratio(graph, neighbor_dict):
    homo_ratios = []
    for node_idx in range(graph.num_nodes):
        centric_label = graph.y[node_idx].cpu().item()
        # print(node_idx, centric_label)
        neighbors_label = graph.y[neighbor_dict[node_idx]].cpu().numpy().tolist()

        if len(neighbors_label) == 0:
            homo_ratios.append(1)
        else:
            counter = [1 if n_label == centric_label else 0 for n_label in neighbors_label]
            homo_ratios.append(sum(counter) / len(neighbors_label))
    
    avg_homo_ratios = round(sum(homo_ratios) / len(homo_ratios) * 100, 3)
    return avg_homo_ratios


def label_distribution(graph):
    counter = {label_idx: 0 for label_idx, label_name in enumerate(graph.label_name)}
    for node_idx in range(graph.num_nodes):
        node_label = graph.y[node_idx].cpu().item()
        counter[node_label] += 1
    counter = {k: round(v / graph.num_nodes * 100, 3) for k, v in counter.items()}
    return counter


def degree_distribution(graph, neighbor_dict): 
    counter = {label_name: [] for label_idx, label_name in enumerate(graph.label_name)}
    for node_idx in range(graph.num_nodes):
        node_degree = len(neighbor_dict[node_idx])
        node_label = graph.label_name[graph.y[node_idx].cpu().item()]
        counter[node_label].append(node_degree)  
        
    counter = {k: round(sum(v) / len(v), 3) for k, v in counter.items()}
    type_degrees = list(counter.values())
    max_d, min_d = max(type_degrees), min(type_degrees)
    type_degrees = [ deg / max_d for deg in type_degrees]
    # print(type_degrees)
    
    mses = []
    for idx, val in enumerate(type_degrees):
        for val2 in type_degrees[idx+1:]:
            mse = (val - val2) * (val - val2)
            mses.append(mse) 
    return counter, round(sum(mses) / len(mses), 3)


if __name__ == "__main__":
    device = torch.device("cuda:0")

    for dataname in ["cora", "citeseer", "pubmed", "wikics", "instagram", "reddit", "arxiv"]:
        g_data, g_dict = load_target_graph(dataname)
        
        # Part I Homophily Analysis
        # cora Homophily Ratio 82.516
        # citeseer Homophily Ratio 72.931
        # pubmed Homophily Ratio 79.241
        # wikics Homophily Ratio 68.668
        # instagram Homophily Ratio 65.353
        # reddit Homophily Ratio 55.524
        # arxiv Homophily Ratio 63.53
        cur_homophily = compute_homo_ratio(g_data, g_dict)
        print(f"{dataname} Homophily Ratio {cur_homophily}")
        
        # Part II Label Analysis 
        # cora 7 Labels ['Rule_Learning', 'Neural_Networks', 'Case_Based', 'Genetic_Algorithms', 'Theory', 'Reinforcement_Learning', 'Probabilistic_Methods']
        # citeseer 6 Labels ['Agents', 'ML (Machine Learning)', 'IR (Information Retrieval)', 'DB (Databases)', 'HCI (Human-Computer Interaction)', 'AI (Artificial Intelligence)']
        # pubmed 3 Labels ['Experimental', 'Diabetes Mellitus Type 1', 'Diabetes Mellitus Type 2']
        # wikics 10 Labels ['Computational Linguistics', 'Databases', 'Operating Systems', 'Computer Architecture', 'Computer Security', 'Internet Protocols', 'Computer File Systems', 'Distributed Computing Architecture', 'Web Technology', 'Programming Language Topics']
        # instagram 2 Labels ['Normal Users', 'Commercial Users']
        # reddit 2 Labels ['Normal Users', 'Popular Users']
        label_names = g_data.label_name 
        print(f"{dataname} {len(label_names)} Labels {label_names}")
        
        # Part III Label Distribution Analysis  
        # cora {0: 6.647, 1: 30.207, 2: 11.004, 3: 15.436, 4: 12.962, 5: 8.013, 6: 15.731} 
        # citeseer {0: 17.64, 1: 17.859, 2: 20.119, 3: 21.5, 4: 15.38, 5: 7.502} 
        # pubmed {0: 20.809, 1: 39.94, 2: 39.25} 
        # wikics {0: 2.521, 1: 5.7, 2: 18.4, 3: 16.52, 4: 22.895, 5: 6.666, 6: 3.53, 7: 7.393, 8: 4.205, 9: 12.17} 
        # instagram {0: 63.709, 1: 36.291} 
        # reddit {0: 50.0, 1: 50.0} 
        # arxiv {0: 0.334, 1: 0.406, 2: 2.858, 3: 1.228, 4: 3.462, 5: 2.928, 6: 0.955, 7: 0.348, 8: 3.68, 9: 1.665, 10: 4.647, 11: 0.443, 12: 0.017, 13: 1.392, 14: 0.353, 15: 0.238, 16: 16.134, 17: 0.304, 18: 0.442, 19: 1.699, 20: 1.226, 21: 0.232, 22: 1.124, 23: 1.674, 24: 13.102, 25: 0.742, 26: 2.719, 27: 2.835, 28: 12.641, 29: 0.246, 30: 6.976, 31: 1.67, 32: 0.243, 33: 0.751, 34: 4.646, 35: 0.075, 36: 2.081, 37: 1.399, 38: 0.89, 39: 1.198} 
        label_dist = label_distribution(g_data)
        print(f"{dataname} {label_dist} \n")  
        
        # Part IV Label Inference with Structure 
        # this is only obvious for reddit 
        # reddit {'Normal Users': 12.402, 'Popular Users': 23.833} 
        
        # other datasets 
        # instagram {'Normal Users': 28.284, 'Commercial Users': 25.85} 
        # cora {'Rule_Learning': 7.367, 'Neural_Networks': 7.169, 'Case_Based': 7.43, 'Genetic_Algorithms': 8.943, 'Theory': 9.003, 'Reinforcement_Learning': 9.585, 'Probabilistic_Methods': 7.826} 
        # citeseer {'Agents': 2.783, 'ML (Machine Learning)': 2.355, 'IR (Information Retrieval)': 3.978, 'DB (Databases)': 2.295, 'HCI (Human-Computer Interaction)': 2.157, 'AI (Artificial Intelligence)': 1.971} 
        # pubmed {'Experimental': 7.712, 'Diabetes Mellitus Type 1': 9.823, 'Diabetes Mellitus Type 2': 8.83}
        
        # degree mse
        # cora 1.868; citeseer 1.071; pubmed 2.231; instagram 5.924 
        # reddit 130.668; reddit arxiv 46.288 
        # normalized degree mse
        # cora 0.02; citeseer 0.068; pubmed 0.023; instagram 0.007; arxiv 0.064 
        # reddit 0.23; wikics 0.153
        degree_cnt, degree_mse = degree_distribution(g_data, g_dict)
        print(dataname, degree_mse)
        
        # Part V Label Inference with Text 
        if dataname in ["instagram"]:
            random_ids = random.choices(list(range(g_data.num_nodes)), k=10)
            for node_idx in random_ids:
                print(node_idx, g_data.label_name[g_data.y[node_idx].cpu().item()], g_data.raw_texts[node_idx], "\n")
