import torch.nn as nn
import torch.nn.functional as F 
import torch 
import numpy as np


def compute_contrastive_loss(graph_feats, text_feats, labels):
    logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07)).exp()
    logits = logit_scale * graph_feats @ text_feats.t()
    logits_left = F.cross_entropy(logits, labels)
    logits_right = F.cross_entropy(logits.T, labels)
    
    return (logits_left + logits_right) / 2 
