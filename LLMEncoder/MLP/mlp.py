import torch 
import torch.nn as nn 
import torch.nn.functional as F 


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2, dropout=0.0, use_softmax=0):
        super(MLP, self).__init__()
        
        self.linear_layers = nn.ModuleList()
        self.linear_layers.append(nn.Linear(input_dim, hidden_dim))
        
        self.bns = torch.nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        
        for _ in range(n_layers - 2):
            self.linear_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
            
        self.linear_layers.append(nn.Linear(hidden_dim, output_dim))
        self.dropout = dropout 
        self.act = F.leaky_relu
        self.use_softmax = use_softmax
    
    def reset_parameters(self):
        for linear_layer in self.linear_layers:
            linear_layer.reset_parameters()
        
        for bn in self.bns: 
            bn.reset_parameters()
    
    def forward(self, x):
        for i, linear_layer in enumerate(self.linear_layers[:-1]):
            x = linear_layer(x)
            x = self.bns[i](x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        x = self.linear_layers[-1](x)
        
        if self.use_softmax: 
            x.log_softmax(dim=-1)
        
        return x
        