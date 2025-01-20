import torch.nn as nn
import torch
import copy
import sys
from transformers import AutoTokenizer, AutoModel,  AutoModelForCausalLM
import torch.nn.functional as F
from peft import LoraModel, LoraConfig
from transformers import  AutoTokenizer, AutoModel
sys.path.append("../../")
from common import normalize_adj_matrix, compute_acc_and_f1
from common import MODEL_PATHs as encoder_dict, ZeroG_DESC as descriptions


class ZeroGModel(nn.Module):
    def __init__(self, args):
        super(ZeroGModel, self).__init__()
        self.args = args
        
        assert args.text_encoder in encoder_dict.keys()
        encoder_fullname = encoder_dict[args.text_encoder]

        if args.text_encoder in ["SentenceBert", "roberta", "e5-large"]:
            self.tokenizer = AutoTokenizer.from_pretrained(encoder_fullname)
            self.textmodel = AutoModel.from_pretrained(encoder_fullname)
            if args.text_encoder in ['SentenceBert']:
                self.target_modules = ["q_lin", "v_lin"]
            elif args.text_encoder in ["roberta", "e5-large"]:
                self.target_modules = ["query", "value"]
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(encoder_fullname)
            self.text_model = AutoModelForCausalLM.from_pretrained(encoder_fullname, torch_dtype=torch.float16).to(self.device)
            self.text_model.config.pad_token_id = self.text_model.config.eos_token_id
            self.target_modules = ["q_proj", "v_proj"]
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.config = LoraConfig(
            task_type="SEQ_CLS",
            r=4,
            lora_alpha=16,
            target_modules=self.target_modules,
            lora_dropout=0.1,
        )
        self.lora_model = LoraModel(self.textmodel, self.config, "default")
        self.descriptions = descriptions

        self.criteria = nn.CrossEntropyLoss()

    def text_forward(self, text):
        tokens = self.tokenizer(text, max_length=256, return_tensors='pt', truncation=True, padding=True).to(self.args.device)
        if self.args.text_encoder in ["SentenceBert", "roberta", "e5-large"]:
            text_embeds = self.lora_model(**tokens)[0][:, 0, :]
        else:
            outputs = self.lora_model(**tokens, output_hidden_states=True)
            # text_embeds = mean_pooling_llm(outputs.hidden_states[-1], tokens["attention_mask"])
            # ZeroG - official code 
            text_embeds = outputs.hidden_states[-1][:, 0, :]
        return text_embeds

    def forward(self, data):
        # insert a description node
        virtual_node_description = self.descriptions[data.dataset_name]
        node_embeds = self.text_forward(data.raw_text + [virtual_node_description])
        label_embeds = self.text_forward(data.label_text)

        if self.args.if_norm:
            node_embeds = (node_embeds - node_embeds.mean(0)) / node_embeds.std(0)
            label_embeds = (label_embeds - label_embeds.mean(0)) / label_embeds.std(0)

        # change the adj matrix
        num_existing_nodes = data.y.shape[0] + 1
        virtual_node_index = data.y.shape[0]
        if data.dataset_name in ["citeseer", "arxiv"]:
            new_edges_to_virtual = [[node_idx, virtual_node_index] for node_idx in range(num_existing_nodes-1)]
        else:
            new_edges_to_virtual = []
            for node_idx in range(num_existing_nodes-1):
                new_edges_to_virtual.append([node_idx, virtual_node_index])
                new_edges_to_virtual.append([virtual_node_index, node_idx])
        new_edge_index = torch.cat([data.edge_index.t(), torch.tensor(
            new_edges_to_virtual, dtype=torch.long).to(self.args.device)], dim=0).t()

        adj_normed = normalize_adj_matrix(new_edge_index, num_existing_nodes, self.args.device)
        for _ in range(self.args.R):
            node_embeds = torch.mm(adj_normed, node_embeds)
        new_node_embeds = node_embeds[:-1, :]
        logits = torch.mm(new_node_embeds, label_embeds.transpose(1, 0))
        logits = torch.div(logits, 1)

        labels = data.y.long().to(self.args.device) if data.y.dim() == 1 else data.y.squeeze(1).long().to(self.args.device)
        CL_loss = self.criteria(logits, labels)

        return CL_loss

    def zero_shot_eval(self, node_embeds, label_embeds, data):
        if self.args.if_norm:
            node_embeds = (node_embeds - node_embeds.mean(0)) / node_embeds.std(0)
            label_embeds = (label_embeds - label_embeds.mean(0)) / label_embeds.std(0)

        # change the adj matrix
        num_existing_nodes = data.y.shape[0] + 1
        virtual_node_index = data.y.shape[0]
        
        new_edges_to_virtual = []
        for node_idx in range(num_existing_nodes-1):
            new_edges_to_virtual.extend([[node_idx, virtual_node_index], [virtual_node_index, node_idx]])
    
        new_edge_index = torch.cat([data.edge_index.t(), torch.tensor(new_edges_to_virtual, dtype=torch.long).to(self.args.device)], dim=0).t()
        adj_normed = normalize_adj_matrix(new_edge_index, num_existing_nodes, self.args.device)

        labels = data.y.long().to(self.args.device)
        score_dict = {}
        test_node_embeds = copy.deepcopy(node_embeds)
        
        for cur_R in range(self.args.R):
            test_label_embeds = copy.deepcopy(label_embeds)
            test_label_embeds /= test_label_embeds.norm(dim=-1, keepdim=True)
            comp_node_embeds = copy.deepcopy(test_node_embeds[:-1, :])
            comp_node_embeds /= comp_node_embeds.norm(dim=-1, keepdim = True)

            cur_dists = torch.einsum('bn,cn->bc', comp_node_embeds, test_label_embeds)
            preds = torch.argmax(cur_dists, dim=1)
            test_acc, test_f1, _ = compute_acc_and_f1(preds[data.test_mask].cpu(), labels[data.test_mask].cpu())
            score_dict[cur_R] = [test_acc, test_f1]

            node_embeds = torch.mm(adj_normed, node_embeds)
            test_node_embeds = torch.mm(adj_normed, test_node_embeds)

        node_embeds = node_embeds[:-1, :]
        node_embeds /= node_embeds.norm(dim=-1, keepdim=True).to(self.args.device)
        label_embeds /= label_embeds.norm(dim=-1, keepdim=True)
        dists = torch.einsum('bn,cn->bc', node_embeds, label_embeds)
        preds = torch.argmax(dists, dim=1)
        labels = data.y.long().to(self.args.device)

        test_acc, test_f1, _ = compute_acc_and_f1(preds[data.test_mask].cpu(), labels[data.test_mask].cpu())
        return [test_acc,test_f1], score_dict
    