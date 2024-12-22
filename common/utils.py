import numpy as np 
import random
import torch
import datetime
import time
import pytz


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)  # gpu
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def get_cur_time(timezone='Asia/Shanghai', t_format='%m-%d %H:%M:%S'):
    return datetime.datetime.fromtimestamp(int(time.time()), pytz.timezone(timezone)).strftime(t_format)


def array_mean_std(numbers):
    array = np.array(numbers)
    return np.round(np.mean(array), 3), np.round(np.std(array), 3)


def normalize_adj_matrix(edge_index, num_nodes, device):
    edge_index_self_loops = torch.stack(
        [torch.arange(num_nodes), torch.arange(num_nodes)], dim=0
    ).to(device)
    edge_index = torch.cat([edge_index, edge_index_self_loops], dim=1)

    adj = torch.sparse_coo_tensor(edge_index, torch.ones(edge_index.shape[1]).to(device), (num_nodes, num_nodes))

    deg = torch.sparse.sum(adj, dim=1).to_dense()
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0.

    adj_normalized = adj
    deg_inv_sqrt_mat = torch.sparse_coo_tensor(
        torch.arange(num_nodes).unsqueeze(0).repeat(2, 1).to(device), 
        deg_inv_sqrt, (num_nodes, num_nodes))
    
    adj_normalized = torch.sparse.mm(
            deg_inv_sqrt_mat, torch.sparse.mm(adj_normalized, deg_inv_sqrt_mat))

    return adj_normalized


def prepare_edge_list(edge_index, num_nodes):
    """Convert [torch.LongTensor] edge_index into [List] edge_list"""
    row, col = edge_index
    edge_list = [[] for _ in range(num_nodes)] 
    
    row, col = row.numpy(), col.numpy()
    for i in range(row.shape[0]):
        edge_list[row[i]].append(int(col[i]))
    return edge_list 
