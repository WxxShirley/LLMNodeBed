import torch 
import torch.nn as nn 
from transformers import AutoModelForCausalLM, AutoTokenizer
import contextlib
import sys 
sys.path.append("../..")
from common import BOS, EOS_USER, EOS, DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN, IGNORE_INDEX


class GraphGPTModel(torch.nn.Module):
    def __init__(self, args, llm_path, graph_embedding, stage="matching"):
        super().__init__()
        
        self.stage = stage
        if stage == "matching":
            self.max_txt_len = args.s1_max_txt_length 
            self.max_new_tokens = args.s1_max_ans_length 
        else:
            self.max_txt_len = args.s2_max_txt_length 
            self.max_new_tokens = args.s2_max_ans_length 

        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        self.tokenizer.pad_token_id = 0 
        self.tokenizer.padding_side = 'left'
        
        kwargs = {
            "max_memory": {0: "80GiB"},
            "device_map": "auto"
        }
        model = AutoModelForCausalLM.from_pretrained(llm_path, torch_dtype=torch.float16, **kwargs)
        for _, param in model.named_parameters():
            param.requires_grad = False 
            
        self.model = model 
        self.word_embedding = self.model.get_input_embeddings()
        self.graph_embedding = graph_embedding
        print(f"Finish loading pre-trained {args.llm} model!")
        
        input_dim = self.graph_embedding.shape[1]
        self.graph_projector = nn.Linear(input_dim, args.output_dim).to(self.model.device)
        
    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.model.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()
    
    def encode_graphs(self, node_index):
        node_embeds = self.graph_embedding[node_index]
        return self.graph_projector(node_embeds)
    
    def forward(self, samples):
        queries = self.tokenizer(samples["query"], add_special_tokens=False) # input query 
        labels = self.tokenizer(samples["label"], add_special_tokens=False)
        
        eos_tokens = self.tokenizer(EOS, add_special_tokens=False)
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        gstart_embeds = self.word_embedding(self.tokenizer(DEFAULT_G_START_TOKEN, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.model.device))
        gend_embeds = self.word_embedding(self.tokenizer(DEFAULT_G_END_TOKEN, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.model.device))
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.model.device))
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(self.model.device)).unsqueeze(0)
        
        batch_size = len(samples["id"])
        
        batch_input_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []
        
        batch_node_indexes  = samples["nodes"].to(self.model.device)
        
        for i in range(batch_size):
            label_input_ids = labels.input_ids[i][:self.max_new_tokens] + eos_tokens.input_ids 
            
            input_ids = queries.input_ids[i][:self.max_txt_len] + eos_user_tokens.input_ids + label_input_ids
            input_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            
            node_embeds = self.encode_graphs(batch_node_indexes[i, :])
            graph_embeds = torch.cat([gstart_embeds, node_embeds, gend_embeds], dim=0)
            input_embeds = torch.cat([bos_embeds, graph_embeds, input_embeds], dim=0)
            
            batch_input_embeds.append(input_embeds)
            batch_attention_mask.append([1] * input_embeds.shape[0])
            label_input_ids = [IGNORE_INDEX] * (input_embeds.shape[0] - len(label_input_ids)) + label_input_ids
            batch_label_input_ids.append(label_input_ids)
        
        max_length = max([x.shape[0] for x in batch_input_embeds]) 
        for i in range(batch_size):
            pad_length = max_length - batch_input_embeds[i].shape[0]
            batch_input_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_input_embeds[i]])
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]
            
            batch_label_input_ids[i] = [IGNORE_INDEX] * pad_length + batch_label_input_ids[i]
            
        input_embeds = torch.stack(batch_input_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
        label_input_ids = torch.tensor(batch_label_input_ids).to(self.model.device)
        
        with self.maybe_autocast():
            outputs = self.model(
                inputs_embeds=input_embeds, 
                attention_mask=attention_mask,
                return_dict=True,
                labels=label_input_ids
            )
            
        return outputs.loss
    
    def inference(self, samples):
        queries = self.tokenizer(samples["query"])
        
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        gstart_embeds = self.word_embedding(self.tokenizer(DEFAULT_G_START_TOKEN, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.model.device))
        gend_embeds = self.word_embedding(self.tokenizer(DEFAULT_G_END_TOKEN, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.model.device))
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.model.device))
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(self.model.device)).unsqueeze(0)
        
        batch_size = len(samples["id"])
        
        batch_input_embeds = []
        batch_attention_mask = []

        batch_node_indexes  = samples["nodes"].to(self.model.device)
        
        for i in range(batch_size):
            input_ids = queries.input_ids[i][:self.max_txt_len] + eos_user_tokens.input_ids 
            input_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            
            node_embeds = self.encode_graphs(batch_node_indexes[i, :])
            graph_embeds = torch.cat([gstart_embeds, node_embeds, gend_embeds], dim=0)
            input_embeds = torch.cat([bos_embeds, graph_embeds, input_embeds], dim=0)
            
            batch_input_embeds.append(input_embeds)
            batch_attention_mask.append([1] * input_embeds.shape[0])
        
        max_length = max([x.shape[0] for x in batch_input_embeds])
        
        for i in range(batch_size):
            pad_length = max_length - batch_input_embeds[i].shape[0]
            batch_input_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_input_embeds[i]], dim=0)
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]
        
        input_embeds = torch.stack(batch_input_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
        
        with self.maybe_autocast():
            outputs = self.model.generate(
                inputs_embeds=input_embeds, 
                max_new_tokens=self.max_new_tokens,
                attention_mask=attention_mask,
                use_cache=False
            )
        
        pred = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return samples["id"], pred 
    
    def print_trainable_params(self):
        trainable_params = 0
        all_param = 0 

        for _, param in self.named_parameters():
            num_params = param.numel()

            all_param += num_params
            if param.requires_grad:
                trainable_params += num_params
        
        return trainable_params, all_param
        