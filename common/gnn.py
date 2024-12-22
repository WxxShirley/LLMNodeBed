import torch.nn.functional as F 
from torch_geometric.nn import GCNConv, GATConv, TransformerConv, SAGEConv, GINConv
import torch.nn as nn


def build_conv(conv_type: str):
    """Return the specific gnn as`conv_type`"""
    if conv_type == "GCN":
        return GCNConv
    elif conv_type == "GIN":
        return lambda i, h: GINConv(
            nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, h))
        )
    elif conv_type == "GAT":
        return GATConv
    elif conv_type == "TransformerConv":
        return TransformerConv
    elif conv_type == "SAGE":
        return SAGEConv
    else:
        raise KeyError("GNN_TYPE can only be GAT, GCN, SAGE, GIN, and TransformerConv")


class GNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2, gnn_type="GCN", dropout=0.0, use_softmax=0, 
                 batch_norm=0, residual_conn=0, jump_knowledge=0):
        super().__init__()

        conv = build_conv(gnn_type)

        self.gnn_type = gnn_type
        self.hidden_dim = hidden_dim 
        self.output_dim = output_dim
        self.dropout = dropout
        self.act = F.leaky_relu
        self.use_softmax = use_softmax
        self.batch_norm = batch_norm
        self.residual_conn = residual_conn
        self.jump_knowledge = jump_knowledge
        
        if n_layers == 1:
            self.conv_layers = nn.ModuleList([conv(input_dim, output_dim)])
        elif n_layers == 2:
            self.conv_layers = nn.ModuleList([conv(input_dim, hidden_dim), conv(hidden_dim, output_dim)])
        else:
            self.conv_layers = nn.ModuleList([conv(input_dim, hidden_dim)])
            for _ in range(n_layers - 2):
                self.conv_layers.append(conv(hidden_dim, hidden_dim))
            self.conv_layers.append(conv(hidden_dim, output_dim))

        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(n_layers-1)])
        
    def reset_parameters(self):
        for conv in self.conv_layers:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
    
    def forward(self, x, edge_index):
        global_x = 0 

        for i, graph_conv in enumerate(self.conv_layers[:-1]):
            # Check whether it needs to add residual connection
            if self.residual_conn and i > 0:
                x = graph_conv(x, edge_index) + x 
            else:
                x = graph_conv(x, edge_index)
            
            # Check whether it needs to add batch normalization
            if self.batch_norm:
                x = self.bns[i](x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            global_x += x 
        
        if self.jump_knowledge:
            x = global_x

        x = self.conv_layers[-1](x, edge_index)
        
        if self.use_softmax:
            return x.log_softmax(dim=-1)
        
        return x
