from transformers import AutoModel, AutoTokenizer
import torch 
from utils import compute_contrastive_loss
import sys 
sys.path.append("../..")
from common import GNNEncoder, MODEL_PATHs as lm_name_dict


class TextGraphCLIP(torch.nn.Module):
    def __init__(self, gnn_input_dim, gnn_hidden_dim, gnn_layers, projector_dim, text_model="MiniLM", device=torch.device("cuda:0"), lm_freeze=True, num_samples=5):
        super().__init__()
        
        self.device = device
        self.graph_model = GNNEncoder(input_dim=gnn_input_dim,
                                      hidden_dim=gnn_hidden_dim,
                                      output_dim=gnn_hidden_dim,
                                      n_layers=gnn_layers,
                                      gnn_type="GCN",
                                      dropout=0.5)
        
        assert text_model in lm_name_dict.keys(), "Please provide valid LM encoder name"
        lm_fullname = lm_name_dict[text_model]
        self.text_tokenizer = AutoTokenizer.from_pretrained(lm_fullname)
        self.text_model = AutoModel.from_pretrained(lm_fullname)
        self.graph_projector = torch.nn.Linear(in_features=gnn_hidden_dim, out_features=projector_dim, bias=True)
        
        if lm_freeze:
            for k, v in self.text_model.named_parameters():
                v.requires_grad = False
        self.num_neigh_samples = num_samples
    
    def featch_graph_embedding(self, graph_data):
        self.graph_model.eval()
        node_embed = self.graph_model(graph_data.x, graph_data.edge_index)
        node_embed = self.graph_projector(node_embed).detach().cpu()
        return node_embed
    
    def graph_forward(self, x, edge_index, focus_nodes):
        updated_x = self.graph_model(x, edge_index)
        focus_nodes_embed = updated_x[focus_nodes]
        
        return self.graph_projector(focus_nodes_embed)
        
    def text_forward(self, texts):
        tokens = self.text_tokenizer(texts, max_length=256, return_tensors='pt', truncation=True, padding=True).to(self.device)
        text_embeds = self.text_model(**tokens)[0][:, 0, :]
        
        torch.cuda.empty_cache()
        return text_embeds
    
    def forward(self, batch_data, graph_data):
        centric_nodes, nodes_text, neighbor_text = batch_data["id"], batch_data["text"], batch_data["neighbor_text"]
        
        # Flat neighboring text
        flatten_neighbor_text = []
        for i in range(len(neighbor_text[0])):
            for j in range(len(neighbor_text)):
                flatten_neighbor_text.append(neighbor_text[j][i])
        
        node_embed = self.graph_forward(graph_data.x, graph_data.edge_index, centric_nodes.to(self.device))
        text_embed = self.text_forward(nodes_text) 
        neigh_text_embed = self.text_forward(flatten_neighbor_text)

        neigh_text_embed = neigh_text_embed.reshape(node_embed.shape[0], self.num_neigh_samples, text_embed.shape[1])
        neigh_text_embed = torch.mean(neigh_text_embed, dim=1, keepdim=False)
        
        node_embed = node_embed / node_embed.norm(dim=1, keepdim=True)
        text_embed = text_embed / text_embed.norm(dim=1, keepdim=True) 
        neigh_text_embed = neigh_text_embed / neigh_text_embed.norm(dim=1, keepdim=True)
        
        labels = torch.arange(node_embed.shape[0]).to(self.device)
        
        node2text_loss = compute_contrastive_loss(node_embed, text_embed, labels)
        node2neightext_loss = compute_contrastive_loss(node_embed, neigh_text_embed, labels)
        text2neightext_loss = compute_contrastive_loss(text_embed, neigh_text_embed, labels)
        
        return node2text_loss, node2neightext_loss, text2neightext_loss        
