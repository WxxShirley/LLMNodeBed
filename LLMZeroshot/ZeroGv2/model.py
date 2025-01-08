import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import GCNConv, GINConv, GATConv, global_add_pool
from torch_geometric.nn.inits import uniform

from transformers import BertModel, BertTokenizer, RobertaTokenizer, RobertaModel, AutoTokenizer, AutoModel, AutoModelForMaskedLM,RobertaTokenizer, RobertaModel, T5Tokenizer, T5Model,T5EncoderModel




class TextModel(nn.Module):
    def __init__(self, encoder):
        super(TextModel, self).__init__()
        self.encoder = encoder
        if self.encoder == 'Bert' or  self.encoder == 'bert':
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.textmodel = BertModel.from_pretrained('bert-base-uncased')


        if self.encoder == 'Roberta' or  self.encoder == 'roberta' :
            self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
            self.textmodel = RobertaModel.from_pretrained('roberta-base')
        if self.encoder == 'SentenceBert':
            self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/multi-qa-distilbert-cos-v1")
            self.textmodel = AutoModel.from_pretrained("sentence-transformers/multi-qa-distilbert-cos-v1")
        if self.encoder == 'SimCSE':
            self.tokenizer = AutoTokenizer.from_pretrained('princeton-nlp/sup-simcse-bert-base-uncased')
            self.textmodel = AutoModel.from_pretrained('princeton-nlp/sup-simcse-bert-base-uncased')
        if self.encoder == 'e5':
            self.tokenizer = AutoTokenizer.from_pretrained('intfloat/e5-base-v2')
            self.textmodel = AutoModel.from_pretrained('intfloat/e5-base-v2')
        if self.encoder  == 't5':
            self.tokenizer = T5Tokenizer.from_pretrained("t5-large")
            self.textmodel = T5EncoderModel.from_pretrained("t5-large")
       

    
    def forward(self, input):        
        inputs = self.tokenizer(input, return_tensors='pt', truncation=True, padding=True).to(self.textmodel.device)

        with torch.no_grad():
            outputs = self.textmodel(**inputs)

        text_embedding = outputs[0][:,0,:].squeeze()
        return text_embedding
