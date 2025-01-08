import os
import shutil
from typing import Callable, List, Optional, Tuple, Union
import torch
import torch.nn.functional as F
import os.path as osp
import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score, average_precision_score
from torch_geometric.datasets import DGraphFin, EllipticBitcoinDataset, HeterophilousGraphDataset, TUDataset
from torch_geometric.data import InMemoryDataset, Data, download_url, extract_zip
from torch_geometric.utils import degree, to_undirected, from_scipy_sparse_matrix, coalesce
from dgl import load_graphs
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy import io
from model import TextModel
import tqdm
from ogb.nodeproppred import PygNodePropPredDataset

import requests


class DataWrapper:
    def __init__(self, data, args):
        self._data = data
        self.args = args
        self.x = self._data.x
        self.raw_texts = self._data.raw_texts
        self.label_text = self._data.label_name

    @property
    def data(self):
        return self._data

    def label_embedding(self):
        text_model = TextModel(self.args.text_encoder)
        text_features = []
        raw_texts = self.data.label_name
        for text in tqdm.tqdm(raw_texts, desc="Processing label texts"):
            text_features.append(text_model(text).unsqueeze(dim=0).cpu())
        self.data.label_emb = torch.cat(text_features, dim=0)

    def feature_embedding(self):
        emb_file = f"saved_embs/{self.data.x.shape[0]}.pt"
        if not os.path.exists(emb_file):
            text_model = TextModel(self.args.text_encoder)
            text_features = []
            raw_texts = self.data.raw_texts

            for text in tqdm.tqdm(raw_texts, desc="Processing node texts"):
                text_features.append(text_model(text).unsqueeze(dim=0).cpu())
            self.data.x = torch.cat(text_features, dim=0)
            torch.save(self.data.x, emb_file)
        else:
            self.data.x = torch.load(emb_file)

