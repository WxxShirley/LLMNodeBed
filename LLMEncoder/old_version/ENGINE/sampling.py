import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, k_hop_subgraph, to_edge_index
import copy
from torch_sparse import SparseTensor


def collect_subgraphs(selected_id, graph, walk_steps=20, restart_ratio=0.5):
    graph  = copy.deepcopy(graph) # modified on the copy
    edge_index = graph.edge_index
    node_num = graph.x.shape[0]
    start_nodes = selected_id # only sampling selected nodes as subgraphs
    graph_num = start_nodes.shape[0]
    
    value = torch.arange(edge_index.size(1))

    if type(edge_index) == SparseTensor:
        adj_t = edge_index
    else:
        adj_t = SparseTensor(row=edge_index[0], col=edge_index[1],
                                    value=value,
                                    sparse_sizes=(node_num, node_num)).t()
    
    current_nodes = start_nodes.clone()
    history = start_nodes.clone().unsqueeze(0)
    signs = torch.ones(graph_num, dtype=torch.bool).unsqueeze(0)
    for i in range(walk_steps):
        seed = torch.rand([graph_num])
        nei = adj_t.sample(1, current_nodes).squeeze()
        sign = seed < restart_ratio
        nei[sign] = start_nodes[sign]
        history = torch.cat((history, nei.unsqueeze(0)), dim=0)
        signs = torch.cat((signs, sign.unsqueeze(0)), dim=0)
        current_nodes = nei
    history = history.T
    signs = signs.T
    
    graph_list = []
    for i in range(graph_num):
        path = history[i]
        sign = signs[i]
        node_idx = path.unique()
        sources = path[:-1].numpy().tolist()
        targets = path[1:].numpy().tolist()
        sub_edges = torch.IntTensor([sources, targets]).long()
        sub_edges = sub_edges.T[~sign[1:]].T
        # undirectional
        if sub_edges.shape[1] != 0:
            sub_edges = to_undirected(sub_edges)
        view = adjust_idx(sub_edges, node_idx, graph, path[0].item())

        graph_list.append(view)
    return graph_list
       
        
def adjust_idx(edge_index, node_idx, full_g, center_idx):
    '''re-index the nodes and edge index

    In the subgraphs, some nodes are droppped. We need to change the node index in edge_index in order to corresponds 
    nodes' index to edge index
    '''
    node_idx_map = {j : i for i, j in enumerate(node_idx.numpy().tolist())}
    sources_idx = list(map(node_idx_map.get, edge_index[0].numpy().tolist()))
    target_idx = list(map(node_idx_map.get, edge_index[1].numpy().tolist()))

    edge_index = torch.IntTensor([sources_idx, target_idx]).long()
    x_view = Data(edge_index=edge_index, x=full_g.x[node_idx], center=node_idx_map[center_idx], original_idx=node_idx, y=full_g.y[center_idx], root_n_index=node_idx_map[center_idx])
    return x_view


def ego_graphs_sampler(node_idx, data, hop=2):
    ego_graphs = []
    edge_index  = data.edge_index

    for idx in node_idx:
        subset, sub_edge_index, mapping, _ = k_hop_subgraph([idx], hop, edge_index, relabel_nodes=True)
        sub_x = data.x[subset]
        g = Data(x=sub_x, edge_index=sub_edge_index, root_n_index=mapping, y=data.y[idx], original_idx=subset) # note: there we use root_n_index to record the index of target node, because `PyG` increments attributes by the number of nodes whenever their attribute names contain the substring :obj:`index`
        ego_graphs.append(g)
    return ego_graphs
