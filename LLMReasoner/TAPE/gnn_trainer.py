import torch
from time import time
import numpy as np
from copy import deepcopy
import torch.nn.functional as F
import sys
sys.path.append("../..")
from common import GNNEncoder, compute_acc_and_f1
from common import load_graph_dataset_for_tape, set_seed


class GNNTrainer():
    def __init__(self, cfg, feature_type):
        self.seed = cfg.seed
        self.device = torch.device("cuda:0" if cfg.device > 0 else "cpu")
        self.dataset_name = cfg.dataset
        self.gnn_model_name = cfg.gnn.model.name
        self.lm_model_name = cfg.lm.model.name
        self.hidden_dim = cfg.gnn.model.hidden_dim
        self.num_layers = cfg.gnn.model.num_layers
        self.dropout = cfg.gnn.train.dropout
        self.lr = cfg.gnn.train.lr
        self.feature_type = feature_type
        self.epochs = cfg.gnn.train.epochs
        self.patience = cfg.gnn.train.early_stop

        # Load data
        set_seed(self.seed)
        data, num_classes, _ = load_graph_dataset_for_tape(cfg.dataset, self.device, re_split=cfg.re_split)

        self.num_nodes = data.y.shape[0]
        self.num_classes = num_classes
        data.y = data.y.squeeze()
        
        re_split_prefix, re_split_suffix = '_s_' if cfg.re_split else '', f'-seed{self.seed}'
        # Init gnn feature
        if self.feature_type == 'TA':
            print("Loading pretrained LM features (title and abstract) ...")
            LM_emb_path = f"../../results/TAPE/{self.dataset_name}{re_split_prefix}/{cfg.lm.model.name}{re_split_suffix}.emb"
            print(f"LM_emb_path: {LM_emb_path}")
            features = torch.from_numpy(np.array(
                np.memmap(LM_emb_path, mode='r',
                          dtype=np.float16,
                          shape=(self.num_nodes, 1024)))
            ).to(torch.float32)
        elif self.feature_type == 'E':
            print("Loading pretrained LM features (explanations) ...")
            LM_emb_path = f"../../results/TAPE/{self.dataset_name}{re_split_prefix}{cfg.lm.train.llm_name}/{cfg.lm.model.name}{re_split_suffix}.emb"
            print(f"LM_emb_path: {LM_emb_path}")
            features = torch.from_numpy(np.array(
                np.memmap(LM_emb_path, mode='r',
                          dtype=np.float16,
                          shape=(self.num_nodes, 1024)))
            ).to(torch.float32)
        else:
            print(
                f'Feature type {self.feature_type} not supported. Loading OGB features...')
            self.feature_type = 'ogb'
            features = data.x

        self.features = features.to(self.device)
        self.data = data.to(self.device)

        # ! Trainer init
        self.model = GNNEncoder(input_dim=self.features.shape[1],
                                hidden_dim=self.hidden_dim,
                                output_dim=self.num_classes,
                                n_layers=self.num_layers,
                                gnn_type=self.gnn_model_name,
                                dropout=self.dropout,
                                # for pubmed (semi-supervised), batch_norm=0
                                batch_norm=1 # if self.dataset_name != "pubmed" else 0
                                ).to(self.device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        print(f"\nNumber of GNN parameters: {trainable_params}")
        self.ckpt = f"../../results/TAPE/{self.dataset_name}/{self.gnn_model_name}.pt"

    def _train(self):
        # the same in ../main.py where we train GNNs
        self.model.train()
        self.optimizer.zero_grad()

        output = self.model(self.features, self.data.edge_index)
        loss = F.cross_entropy(output[self.data.train_mask], self.data.y[self.data.train_mask])
        loss.backward()
        self.optimizer.step()
        return float(loss)

    @ torch.no_grad()
    def _evaluate(self):
        # the same in ../main.py where we evaluate GNNs
        self.model.eval()
        logits = self.model(self.features, self.data.edge_index)
        pred = logits.argmax(dim=1)

        accuracy, macrof1_scores, weightf1_scores = [], [], []
        for mask in [self.data.train_mask, self.data.val_mask, self.data.test_mask]:
            acc, macro_f1, weight_f1 = compute_acc_and_f1(pred[mask].cpu().numpy(), self.data.y[mask].cpu().numpy())
            accuracy.append(acc)
            macrof1_scores.append(macro_f1)
            weightf1_scores.append(weight_f1)
        
        return accuracy, macrof1_scores, weightf1_scores, logits

    def train(self):
        # ! Training
        best_eval_acc = best_test_acc = 0.0
        best_eval_f1 = best_test_f1 = 0.0
        timer, counter, best_logits = [], 0, None
        for epoch in range(1, 1+self.epochs):
            loss = self._train()
            accuracy, f1_scores, weightf1_scores, cur_logits = self._evaluate()

            train_acc, val_acc, test_acc = accuracy
            train_f1, val_f1, test_f1 = f1_scores
            train_weightf1, val_weightf1, test_weightf1 = weightf1_scores

            if val_acc > best_eval_acc:
                best_eval_acc = val_acc
                best_test_acc = test_acc
                counter = 0
                best_logits = deepcopy(cur_logits)
                best_eval_f1, best_test_f1 = val_f1, test_f1 
                best_eval_weightf1, best_test_weightf1 = val_weightf1, test_weightf1
            else:
                counter += 1
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch:03d} Loss {loss:.4f}  Train acc {train_acc:.3f} Val acc {val_acc:.3f} Test acc {test_acc:.3f}  Train F1 {train_f1:.3f} Val F1 {val_f1:.3f} Test F1 {test_f1:.4f}")
            
            # Early stopping
            if counter >= self.patience:
                break
    
        print(f'\nTest Acc {best_test_acc:.3f}  Test F1 {best_test_f1:.3f}\n')
        return {
            "test_acc": best_test_acc, 
            "test_f1": best_test_f1,
            "val_acc": best_eval_acc,
            "val_f1": best_eval_f1,
            "val_weight_f1": best_eval_weightf1,
            "test_weight_f1": best_test_weightf1
        }, best_logits


class EnsembleTrainer():
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device("cuda:0" if cfg.device > 0 else "cpu")
        self.dataset_name = cfg.dataset
        self.gnn_model_name = cfg.gnn.model.name
        self.lm_model_name = cfg.lm.model.name
        self.hidden_dim = cfg.gnn.model.hidden_dim
        self.num_layers = cfg.gnn.model.num_layers

        self.dropout = cfg.gnn.train.dropout
        self.lr = cfg.gnn.train.lr
        self.feature_type = cfg.gnn.train.feature_type
        self.epochs = cfg.gnn.train.epochs
        self.weight_decay = cfg.gnn.train.weight_decay

        # ! Load data 
        set_seed(cfg.seed)
        data, _, _ = load_graph_dataset_for_tape(self.dataset_name, self.device, re_split=cfg.re_split)

        data.y = data.y.squeeze()
        self.data = data.to(self.device)
        self.TRAINER = GNNTrainer

    @torch.no_grad()
    def eval(self, logits):
        pred = logits.argmax(dim=1)

        accuracy, macrof1_scores, weightf1_scores = [], [], []
        for mask in [self.data.val_mask, self.data.test_mask]:
            acc, macro_f1, weight_f1 = compute_acc_and_f1(pred[mask].cpu().numpy(), self.data.y[mask].cpu().numpy())
            accuracy.append(acc)
            macrof1_scores.append(macro_f1)
            weightf1_scores.append(weight_f1)
        
        val_acc, test_acc = accuracy
        val_macrof1, test_macrof1 = macrof1_scores 
        val_weightf1, test_weightf1 = weightf1_scores
        print(
            f'({self.feature_type}) ValAcc: {val_acc:.4f}, TestAcc: {test_acc:.4f}, Valf1: {val_macrof1:.4f}, Testf1: {test_macrof1:.4f}\n')
        res = {'val_acc': val_acc, 'test_acc': test_acc,
               'val_f1': val_macrof1, 'test_f1':test_macrof1,
               'val_weight_f1': val_weightf1, 'test_weight_f1': test_weightf1}
        return res

    def train(self):
        all_pred = []
        all_score = {}
        feature_types = self.feature_type.split('_')
        for feature_type in feature_types:
            trainer = self.TRAINER(self.cfg, feature_type)
            cur_score, logits = trainer.train()
            all_pred.append(logits)
            all_score[feature_type] = cur_score
        
        # print("All Pred", all_pred)
        pred_ensemble = sum(all_pred) / len(all_pred)
        # print("Pred Ensemble", pred_ensemble)
        acc_ensemble = self.eval(pred_ensemble)
        all_score['ensemble'] = acc_ensemble
        return all_score
