import torch 
import torch.nn as nn 
from transformers import AutoModelForCausalLM, AutoTokenizer
import contextlib
import sys 
sys.path.append("../..")
from common import BOS, EOS_USER, EOS, IGNORE_INDEX, DEFAULT_GRAPH_PAD_ID
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


class LLaGAModel(torch.nn.Module):
    def __init__(self, args, llm_path, graph_embedding, structure_embedding):
        super().__init__() 
        
        self.max_txt_len = args.max_txt_length
        self.max_new_tokens = args.max_ans_length
        self.neighbor_template = args.neighbor_template
        self.neighbor_desc_mean_pooling = args.nd_mean
        
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        self.tokenizer.pad_token_id = 0
        self.tokenizer.padding_side = 'left'
        
        # TODO: please change the following configurations based on your own device
        kwargs = {
            "max_memory": {args.gpu_id: '80GiB'},
            "device_map": "auto",
        }
        if args.num_gpus == 2:
            kwargs = {
                "max_memory": {0: '48GiB', 1: '48GiB'},
                "device_map": "auto",
            }
        model = AutoModelForCausalLM.from_pretrained(llm_path, torch_dtype=torch.float16, **kwargs)
        
        # Freeze LLM
        if args.llm_freeze:
            for _, param in model.named_parameters(): 
                param.requires_grad = False 
        else:
            model = prepare_model_for_kbit_training(model) 
            lora_config = LoraConfig(r=8, 
                                     lora_alpha=16, 
                                     target_modules=["q_proj", "v_proj"], 
                                     lora_dropout=0.05, 
                                     bias="none", 
                                     task_type="CAUSAL_LM")
            model = get_peft_model(model, lora_config)
            
        self.model = model 
        # TODO: fix device based on args
        self.device = self.model.device
        self.word_embedding = self.model.model.get_input_embeddings()
        self.graph_embedding = graph_embedding
        self.position_embedding = structure_embedding
        print(f"Finish loading pre-trained {args.llm} model!")
        
        # Build Linear Projection Layers
        input_dim = self.graph_embedding.shape[1] + self.position_embedding.shape[1] if self.neighbor_template == "ND" else self.graph_embedding[0].shape[1]
        hidden_dim, output_dim = args.hidden_dim, args.output_dim
        linear_layers = []
        assert args.n_linear_layer >= 2, "# Layers in Linear Projection should be greater than 2, Please fix the configration!"
        for i in range(args.n_linear_layer):
            if i == 0:
                linear_layers.append(nn.Linear(input_dim, hidden_dim))
                linear_layers.append(nn.LeakyReLU())
            elif i != args.n_linear_layer - 1:
                linear_layers.append(nn.Linear(hidden_dim, hidden_dim))
                linear_layers.append(nn.LeakyReLU())
            else:
                linear_layers.append(nn.Linear(hidden_dim, output_dim))
        self.graph_projector = nn.Sequential(*linear_layers).to(self.device)
    
    def maybe_autocast(self, dtype=torch.bfloat16):
        # if on cpu, don't use autocast
        # if on gpu, use autocast with dtype if provided, otherwise use torch.float16
        enable_autocast = self.device != torch.device("cpu")

        if enable_autocast:
            return torch.cuda.amp.autocast(dtype=dtype)
        else:
            return contextlib.nullcontext()
    
    def encode_subgraph_hopfield(self, graph_index):
        graph_index = graph_index.item() # Actually, node_index
        neighbor_embs = []
        for cur_layer_emb in self.graph_embedding: 
            neighbor_embs.append(cur_layer_emb[graph_index, :])
        graph_emb = torch.stack(neighbor_embs, dim=0)
        graph_features = self.graph_projector(graph_emb) 
        # print(graph_emb.shape, graph_features.shape)
        return graph_features
    
    def encode_subgraph_neighbordesc(self, graph_indexes):
        mask = graph_indexes != DEFAULT_GRAPH_PAD_ID 
        masked_graph_emb = self.graph_embedding[graph_indexes[mask]]
        
        s, d = graph_indexes.shape[0], masked_graph_emb.shape[1]
        graph_embed = torch.zeros((s, d)).to(self.device)
        graph_embed[mask] = masked_graph_emb 
        graph_embed = torch.cat([graph_embed, self.position_embedding], dim=1)
        
        graph_features = self.graph_projector(graph_embed)
        graph_features[graph_indexes == DEFAULT_GRAPH_PAD_ID] = 0. 
        
        if self.neighbor_desc_mean_pooling:
            graph_features = torch.mean(graph_features, dim=0, keepdim=True)
            
        return graph_features 
    
    def forward(self, samples):
        queries = self.tokenizer(samples["query"], add_special_tokens=False) # input query
        labels = self.tokenizer(samples["label"], add_special_tokens=False) # output ground-truth label
        
        eos_tokens = self.tokenizer(EOS, add_special_tokens=False)
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.device))
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(self.device)).unsqueeze(0)
        
        batch_size = len(samples['id'])
        
        batch_inputs_embeds = []
        batch_attention_mask = []
        batch_label_input_ids = []
        
        if self.neighbor_template == "ND":
            batch_graph_indexes = torch.stack(samples["graph"]).to(self.device)
        else:
            batch_graph_indexes = samples["id"]
        
        for i in range(batch_size):
            label_input_ids = labels.input_ids[i][:self.max_new_tokens] + eos_tokens.input_ids
            input_ids = queries.input_ids[i][:self.max_txt_len] + eos_user_tokens.input_ids + label_input_ids
            
            input_embeds = self.word_embedding(torch.tensor(input_ids).to(self.device))
            
            # For Debugging
            # print(f'Node ID {samples["id"][i]}', f' Graph Info {batch_graph_indexes[:, i]}')
            if self.neighbor_template == "ND":
                graph_embeds = self.encode_subgraph_neighbordesc(batch_graph_indexes[:, i])
            else:
                graph_embeds = self.encode_subgraph_hopfield(batch_graph_indexes[i])
            
            input_embeds = torch.cat([bos_embeds, graph_embeds, input_embeds], dim=0)
            
            batch_inputs_embeds.append(input_embeds)
            batch_attention_mask.append([1] * input_embeds.shape[0])
            label_input_ids = [IGNORE_INDEX] * (input_embeds.shape[0] - len(label_input_ids)) + label_input_ids
            batch_label_input_ids.append(label_input_ids)
        
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]
            
            batch_label_input_ids[i] = [IGNORE_INDEX] * pad_length + batch_label_input_ids[i]
        
        input_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
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
        queris = self.tokenizer(samples["query"], add_special_tokens=False)
        
        eos_user_tokens = self.tokenizer(EOS_USER, add_special_tokens=False)
        bos_embeds = self.word_embedding(self.tokenizer(BOS, add_special_tokens=False, return_tensors='pt').input_ids[0].to(self.device))
        pad_embeds = self.word_embedding(torch.tensor(self.tokenizer.pad_token_id).to(self.device)).unsqueeze(0)
        
        batch_size = len(samples["id"])
        
        batch_inputs_embeds = []
        batch_attention_mask = []
        
        if self.neighbor_template == "ND":
            batch_graph_indexes = torch.stack(samples["graph"]).to(self.device)
        else:
            batch_graph_indexes = samples["id"]
        
        for i in range(batch_size):
            input_ids = queris.input_ids[i][:self.max_txt_len] + eos_user_tokens.input_ids
            input_embeds = self.word_embedding(torch.tensor(input_ids).to(self.model.device))
            
            if self.neighbor_template == "ND":
                graph_embeds = self.encode_subgraph_neighbordesc(batch_graph_indexes[:, i])
            else:
                graph_embeds = self.encode_subgraph_hopfield(batch_graph_indexes[i])
            input_embeds = torch.cat([bos_embeds, graph_embeds, input_embeds], dim=0)
            
            batch_inputs_embeds.append(input_embeds)
            batch_attention_mask.append([1] * input_embeds.shape[0])
        
        max_length = max([x.shape[0] for x in batch_inputs_embeds])
        for i in range(batch_size):
            pad_length = max_length - batch_inputs_embeds[i].shape[0]
            batch_inputs_embeds[i] = torch.cat([pad_embeds.repeat(pad_length, 1), batch_inputs_embeds[i]])
            batch_attention_mask[i] = [0] * pad_length + batch_attention_mask[i]
            
        input_embeds = torch.stack(batch_inputs_embeds, dim=0).to(self.model.device)
        attention_mask = torch.tensor(batch_attention_mask).to(self.model.device)
        
        with self.maybe_autocast():
            outputs = self.model.generate(
                inputs_embeds=input_embeds,
                max_new_tokens=16, 
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
    