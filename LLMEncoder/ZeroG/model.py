import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification
from peft import LoraModel, LoraConfig
import sys 
sys.path.append("../../")
from common import normalize_adj_matrix, compute_acc_and_f1, mean_pooling_llm
from common import MODEL_PATHs as encoder_dict, ZeroG_DESC as descriptions


class TextLoraModel(nn.Module):
    def __init__(self, args):
        super(TextLoraModel, self).__init__()
        self.args = args 
        self.device = torch.device(args.device)
        
        assert args.text_encoder in encoder_dict.keys()
        encoder_fullname = encoder_dict[args.text_encoder]
        
        if args.text_encoder in ["SentenceBert", "roberta", "e5-large", "MiniLM"]:
            self.tokenizer = AutoTokenizer.from_pretrained(encoder_fullname)
            self.text_model = AutoModel.from_pretrained(encoder_fullname).to(self.device)
            if args.text_encoder in ['SentenceBert']:
                self.target_modules = ["q_lin", "v_lin"]
            elif args.text_encoder in ["roberta", "e5-large", "MiniLM"]:
                self.target_modules = ["query", "value"]
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(encoder_fullname)
            self.text_model = AutoModelForSequenceClassification.from_pretrained(encoder_fullname, torch_dtype=torch.float16).to(self.device)
            self.text_model.config.pad_token_id = self.text_model.config.eos_token_id
            print(self.text_model)
            self.target_modules = ["q_proj", "v_proj"]
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.config = LoraConfig(
            task_type="SEQ_CLS",
            r=4,
            lora_alpha=8,
            target_modules=self.target_modules,
            lora_dropout=0.1
        )

        self.lora_model = LoraModel(self.text_model, self.config, "default") if args.use_lora else self.text_model
        self.criteria = nn.CrossEntropyLoss()

    def text_forward(self, text):
        text = "Empty Text" if len(text) == 0 else text
        # print(len(text))
        tokens = self.tokenizer(text, max_length=256, return_tensors='pt', truncation=True, padding=True).to(self.device)
        if self.args.text_encoder in ["SentenceBert", "roberta", "e5-large", "MiniLM"]:
            text_embeds = self.lora_model(**tokens)[0][:, 0, :]
        # elif self.args.text_encoder == "MiniLM":
        #     text_embeds = mean_pooling(self.lora_model(**tokens), tokens["attention_mask"])
        else:
            outputs = self.lora_model(**tokens, output_hidden_states=True)
            text_embeds = mean_pooling_llm(outputs.hidden_states[-1], tokens["attention_mask"])
        del tokens
        torch.cuda.empty_cache()
        return text_embeds

    def forward(self, batch_data, label_texts):
        node_embeds = self.text_forward(batch_data["text"])
        label_embeds = self.text_forward(label_texts)

        if self.args.if_norm:
            node_embeds = (node_embeds - node_embeds.mean(0)) / node_embeds.std(0)
            label_embeds = (label_embeds - label_embeds.mean(0)) / label_embeds.std(0)
        
        logits = torch.mm(node_embeds, label_embeds.transpose(1, 0))
        logits = torch.div(logits, 1)
        cl_loss = self.criteria(logits, batch_data["label"].long())

        return cl_loss
    
    def zero_shot_eval(self, node_embeds, label_embeds, data):
        if self.args.if_norm:
            # print(f"hit feature normalization")
            node_embeds = (node_embeds - node_embeds.mean(0)) / node_embeds.std(0)
            label_embeds = (label_embeds - label_embeds.mean(0)) / label_embeds.std(0)
        
        num_exist_nodes = data.y.shape[0] + 1 
        virtual_node_idx = data.y.shape[0]
        new_edges_to_virtual = []
        for node_idx in range(num_exist_nodes-1):
            new_edges_to_virtual.extend([[node_idx, virtual_node_idx], [virtual_node_idx, node_idx]])
        
        new_edge_index = torch.cat([data.edge_index.t(), torch.tensor(new_edges_to_virtual, dtype=torch.long).to(self.device)], dim=0).t()

        adj_normed = normalize_adj_matrix(new_edge_index, num_exist_nodes, self.device)

        for _ in range(self.args.test_R):
            node_embeds = torch.mm(adj_normed, node_embeds)
        node_embeds = node_embeds[:-1, :]
        node_embeds /= node_embeds.norm(dim=-1, keepdim=True)
        label_embeds /= label_embeds.norm(dim=-1, keepdim=True)

        dists = torch.einsum('bn,cn->bc', node_embeds, label_embeds)
        preds = torch.argmax(dists, dim=1)

        test_acc, test_f1 = compute_acc_and_f1(preds[data.test_mask].cpu(), data.y[data.test_mask].cpu())
        eval_acc, eval_f1 = compute_acc_and_f1(preds[data.val_mask].cpu(), data.y[data.val_mask].cpu())
        del node_embeds, label_embeds, dists, preds
        torch.cuda.empty_cache()
        return [test_acc, test_f1], [eval_acc, eval_f1]
