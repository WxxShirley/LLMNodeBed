import torch
import torch.nn as nn
import numpy as np
from transformers import PreTrainedModel, AutoTokenizer, AutoModel, AutoModelForCausalLM
from transformers.modeling_outputs import TokenClassifierOutput
from tqdm import tqdm
from .model_path import MODEL_PATHs


def mean_pooling(model_output, attention_mask):
    # from Sentence-Transformers official code "https://huggingface.co/sentence-transformers/all-mpnet-base-v2"
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def mean_pooling_llm(token_embeddings, attention_mask):
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class TextEncoder(nn.Module):
    def __init__(self, encoder_name, encoder_type, device):
        super(TextEncoder, self).__init__()

        self.encoder_name = encoder_name
        self.device = device

        if encoder_type == "LM":
            lm_path = MODEL_PATHs[encoder_name]
            self.tokenizer = AutoTokenizer.from_pretrained(lm_path)
            self.model = AutoModel.from_pretrained(lm_path).to(device)
            self.encoder_type = "LM"
        else:
            llm_path = MODEL_PATHs[encoder_name]

            llm_tokenizer = AutoTokenizer.from_pretrained(llm_path)
            llm_tokenizer.pad_token_id = 0
            llm_tokenizer.padding_side = "right"
            llm_tokenizer.truncation_side = "right"
            self.tokenizer = llm_tokenizer
            self.encoder_type = "LLM"
            
            self.model = AutoModelForCausalLM.from_pretrained(llm_path, torch_dtype=torch.float16).to(device)
            print(self.model)
            self.model.config.pad_token_id = self.model.config.eos_token_id
        
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[{encoder_name}] Number of parameters {trainable_params}")
    
    def engine_forward(self, all_text, max_length=512, pool="cls"):
        layer_dict = {
            "MiniLM": 6, "SentenceBert": 6, "e5-large": 24, "roberta": 24,
            "Qwen-3B": 36, "Qwen-7B": 28,  "Mistral-7B": 32, "Llama-8B": 32
        }
        num_layers = layer_dict[self.encoder_name]
        layers = [[] for _ in range(num_layers+1)]
        
        with torch.no_grad():
            for input_text in tqdm(all_text, desc="Preparing All Hidden States"):
                input_text = "Empty text" if len(input_text) == 0 else input_text
                encoded_input = self.tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(self.device)
                output = self.model(**encoded_input, output_hidden_states=True) 
                hidden_states = output["hidden_states"]
                
                for i, layer_hid in enumerate(hidden_states):
                    if pool == "cls":
                        layer_node_hid = layer_hid[:, 0, :]
                    else:
                        layer_node_hid = mean_pooling_llm(layer_hid, encoded_input["attention_mask"])
                    layers[i].append(layer_node_hid.cpu())

        layers_hid = [torch.cat(xs).float() for xs in layers]
        layers_shape = [obj.shape for obj in layers_hid]
        print(layers_shape)
        return layers_hid
            
    def forward(self, input_text, pooling="cls", max_length=512):
        input_text = "Empty text" if len(input_text) == 0 else input_text
        if self.encoder_type == "LM":
            encoded_input = self.tokenizer(input_text, return_tensors='pt', padding=True, truncation=True, max_length=max_length).to(self.device)
            
            output = self.model(**encoded_input)
            if pooling == "cls":
                text_emb = output.last_hidden_state[:, 0, :]
            else:
                text_emb = mean_pooling(output, encoded_input["attention_mask"])
        elif self.encoder_type == "LLM":
            encoded_input = self.tokenizer(input_text, add_special_tokens=False, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(self.device)
            outputs = self.model(**encoded_input, output_hidden_states=True)
            text_emb = mean_pooling_llm(outputs.hidden_states[-1], encoded_input["attention_mask"])
        
        return text_emb


class BertClassifier(PreTrainedModel):
    def __init__(self, model, n_labels, dropout=0.0, cla_bias=True, feat_shrink=''):
        super().__init__(model.config)
        # model is an instance of `AutoModel.from_pretrained(lm_model_name)`
        self.bert_encoder = model
        self.dropout = nn.Dropout(dropout)
        self.feat_shrink = feat_shrink
        hidden_dim = model.config.hidden_size
        self.loss_func = nn.CrossEntropyLoss(label_smoothing=0.3, reduction='mean')

        if feat_shrink:
            self.feat_shrink_layer = nn.Linear(model.config.hidden_size, int(feat_shrink), bias=cla_bias)
            hidden_dim = int(feat_shrink)
        self.classifier = nn.Linear(hidden_dim, n_labels, bias=cla_bias)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                return_dict=None,
                preds=None):

        outputs = self.bert_encoder(input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    return_dict=return_dict,
                                    output_hidden_states=True)
        # outputs[0]=last hidden state
        emb = self.dropout(outputs['hidden_states'][-1])
        # Use CLS Emb as sentence emb.
        cls_token_emb = emb.permute(1, 0, 2)[0]
        if self.feat_shrink:
            cls_token_emb = self.feat_shrink_layer(cls_token_emb)
        logits = self.classifier(cls_token_emb)

        if labels.shape[-1] == 1:
            labels = labels.squeeze()
        loss = self.loss_func(logits, labels)

        return TokenClassifierOutput(loss=loss, logits=logits)


class BertClaInfModel(PreTrainedModel):
    def __init__(self, model, emb, pred, feat_shrink=''):
        super().__init__(model.config)
        self.bert_classifier = model
        self.emb, self.pred = emb, pred
        self.feat_shrink = feat_shrink
        self.loss_func = nn.CrossEntropyLoss(label_smoothing=0.3, reduction='mean')

    @torch.no_grad()
    def forward(self,
                input_ids=None,
                attention_mask=None,
                labels=None,
                return_dict=None,
                node_id=None):

        # Extract outputs from the model
        bert_outputs = self.bert_classifier.bert_encoder(input_ids=input_ids,
                                                         attention_mask=attention_mask,
                                                         return_dict=return_dict,
                                                         output_hidden_states=True)
        # outputs[0]=last hidden state
        emb = bert_outputs['hidden_states'][-1]
        # Use CLS Emb as sentence emb.
        cls_token_emb = emb.permute(1, 0, 2)[0]
        if self.feat_shrink:
            cls_token_emb = self.bert_classifier.feat_shrink_layer(
                cls_token_emb)
        logits = self.bert_classifier.classifier(cls_token_emb)

        # Save prediction and embeddings to disk (memmap)
        batch_nodes = node_id.cpu().numpy()
        self.emb[batch_nodes] = cls_token_emb.cpu().numpy().astype(np.float16)
        self.pred[batch_nodes] = logits.cpu().numpy().astype(np.float16)

        if labels.shape[-1] == 1:
            labels = labels.squeeze()
        loss = self.loss_func(logits, labels)

        return TokenClassifierOutput(loss=loss, logits=logits)
