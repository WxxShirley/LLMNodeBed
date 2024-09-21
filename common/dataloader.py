import torch 
import os


def load_graph_dataset(dataset_name, device, emb_model="shallow"):
    graph_data = torch.load(f"../datasets/{dataset_name}.pt").to(device)
    
    if emb_model != "shallow":
        assert os.path.exists(f"../datasets/{emb_model}/{dataset_name}.pt")
        node_feat = torch.load(f"../datasets/{emb_model}/{dataset_name}.pt", map_location=device).to(device).type(torch.float)
        graph_data.x = node_feat
    
    if len(graph_data.train_mask) == 10:
        graph_data.train_mask, graph_data.val_mask, graph_data.test_mask = graph_data.train_mask[0], graph_data.val_mask[0], graph_data.test_mask[0]
    
    return graph_data



def load_graph_dataset_for_tape(dataset_name, device, use_gpt=False):
    graph_data = torch.load(f"../../datasets/{dataset_name}.pt").to(device)
    
    if len(graph_data.train_mask) == 10:
        graph_data.train_mask, graph_data.val_mask, graph_data.test_mask = graph_data.train_mask[0], graph_data.val_mask[0], graph_data.test_mask[0]
    
    # TODO: fill use_gpt

    return graph_data, len(graph_data.label_name), graph_data.raw_texts
