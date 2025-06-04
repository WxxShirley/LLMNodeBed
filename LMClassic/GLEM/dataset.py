import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import AutoTokenizer


class TextDataset(Dataset):
    def __init__(self, tokenizer_name, raw_texts, labels, train_mask):
        r"""The Text Attributed Graph datasets from the `"Learning on Large-scale Text-attributed Graphs via Variational Inference" <https://arxiv.org/abs/2210.14709>`_ paper."""
        self.raw_texts = raw_texts
        self.labels = labels 
        self.num_nodes = len(raw_texts)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.train_mask = train_mask # is_gold flag
        self.all_tokens = self._token_nodes()

    def _token_nodes(self, batch_size: int = 256):
        token_keys = ['input_ids', 'attention_mask']
        
        all_encoded_token = {k: [] for k in token_keys}
        pbar = tqdm(total=self.num_nodes, desc='Tokenizing Text Attributed Graph')
        for i in range(0, self.num_nodes, batch_size):
            end_index = min(self.num_nodes, i+batch_size)
            token = self.tokenizer(self.raw_texts[i:end_index], padding='max_length', truncation=True, max_length=512, return_tensors="pt")
            for k in token_keys:
                all_encoded_token[k].append(token[k])
            pbar.update(end_index - i)
        pbar.close()

        all_encoded_token = {k: torch.cat(v) for k, v in all_encoded_token.items() if len(v) > 0}
        return all_encoded_token
    
    def __len__(self):
        return self.num_nodes
    
    def __getitem__(self, idx):
        item = {} 
        item['input'] = {k: v[idx].squeeze(0) for k, v in self.all_tokens.items()}
        item['labels'] = self.labels[idx] # ground-truth label idx
        item['is_gold'] = self.train_mask[idx] # 1 if in golden training set, 0 otherwise
        item['n_id'] = idx  # node idx

        return item 
    