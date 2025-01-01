import torch 
import argparse
import sys
import tqdm
import torch.nn as nn 
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
sys.path.append("../..")
from common import compute_acc_and_f1, mean_pooling_llm, normalize_adj_matrix
from common import load_graph_dataset, MODEL_PATHs as encoder_dict, ZeroG_DESC as descriptions
from common import set_seed


class MiniZeroG(nn.Module):
    def __init__(self, args):
        super(MiniZeroG, self).__init__()
        self.args = args 
        self.device = device
        self.add_virtual_node = args.virtual_node
        
        assert args.text_encoder in encoder_dict.keys()
        encoder_fullname = encoder_dict[args.text_encoder]

        if args.text_encoder in ["SentenceBert", "roberta", "e5-large", "MiniLM"]:
            self.tokenizer = AutoTokenizer.from_pretrained(encoder_fullname)
            self.text_model = AutoModel.from_pretrained(encoder_fullname).to(self.device)
        else:
            # Load local LLM
            self.tokenizer = AutoTokenizer.from_pretrained(encoder_fullname)
            self.text_model = AutoModelForCausalLM.from_pretrained(encoder_fullname, torch_dtype=torch.float16).to(self.device)
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
    def text_forward(self, text):
        text = "Empty text" if len(text) == 0 else text
        tokens = self.tokenizer(text, max_length=256, return_tensors='pt', truncation=True, padding=True).to(self.device)
        if self.args.text_encoder in ["SentenceBert", "roberta", "e5-large"]:
            text_embeds = self.text_model(**tokens)[0][:, 0, :]
        else:
            # For LLMs, we use the sentence-level mean pooling
            outputs = self.text_model(**tokens, output_hidden_states=True)
            text_embeds = mean_pooling_llm(outputs.hidden_states[-1], tokens["attention_mask"])
        del tokens
        
        return text_embeds
    
    def zero_shot_eval(self, node_embeds, label_embeds, data, R):
        if self.args.if_norm:
            # print(f"hit feature normalization")
            node_embeds = (node_embeds - node_embeds.mean(0)) / node_embeds.std(0)
            label_embeds = (label_embeds - label_embeds.mean(0)) / label_embeds.std(0)
        
        if self.add_virtual_node:
            num_exist_nodes = data.y.shape[0] + 1 
            virtual_node_idx = data.y.shape[0]
            new_edges_to_virtual = []
            for node_idx in range(num_exist_nodes-1):
                new_edges_to_virtual.extend([[node_idx, virtual_node_idx], [virtual_node_idx, node_idx]])
        
            new_edge_index = torch.cat([data.edge_index.t(), torch.tensor(new_edges_to_virtual, dtype=torch.long).to(self.device)], dim=0).t()
            adj_normed = normalize_adj_matrix(new_edge_index, num_exist_nodes, self.device)
        else:
            adj_normed = normalize_adj_matrix(data.edge_index, data.num_nodes, self.device)

        for _ in range(R):
            node_embeds = torch.mm(adj_normed, node_embeds)
        
        node_embeds = node_embeds[:-1, :] if args.virtual_node else node_embeds
        node_embeds /= node_embeds.norm(dim=-1, keepdim=True).to(self.device)
        label_embeds /= label_embeds.norm(dim=-1, keepdim=True)

        dists = torch.einsum('bn,cn->bc', node_embeds, label_embeds)
        preds = torch.argmax(dists, dim=1)

        test_acc, test_f1, _ = compute_acc_and_f1(preds[data.test_mask].cpu(), data.y[data.test_mask].cpu())
        
        return [test_acc, test_f1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_datasets", type=str, default="cora,citeseer,pubmed,wikics,instagram,photo")
    parser.add_argument("--if_norm", action="store_true", default=True, help="Indicator of normalization")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--text_encoder", type=str, default="SentenceBert", help="Type of text encoder")
    parser.add_argument("--R", type=int, default=6)
    parser.add_argument("--virtual_node", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    device = torch.device(args.device)
    print("=" * 20)
    print(args)

    test_names, test_graphs = args.test_datasets.split(","), []
    
    set_seed(args.seed)
    for dataset_name in test_names:
        test_graph = load_graph_dataset(dataset_name, device, re_split=True)
        test_graphs.append(test_graph)

    model = MiniZeroG(args)
    model.eval()

    res_list, score_dict = [], {}
    for idx, test_graph_data in enumerate(test_graphs):
        # Zero-shot Prediction 
        test_graph_name = test_names[idx]
        with torch.no_grad():
            text_features = []
            for text in tqdm.tqdm(test_graph_data.raw_texts, desc=f"Processing {test_graph_name} node texts"):
                cur_text_feature = model.text_forward(text).cpu()
                text_features.append(cur_text_feature)
            
            if args.virtual_node:
                desc = descriptions[test_graph_name]
                text_features.append(model.text_forward(desc).cpu())
                
            node_embeds = torch.cat(text_features, dim=0).to(device)

            label_features = []
            label_text = test_graph_data.label_name
            for text in tqdm.tqdm(label_text, desc=f"Processing {test_graph_name} label texts"):
                cur_label_feature = model.text_forward(text).cpu()
                label_features.append(cur_label_feature)
            label_embeds = torch.cat(label_features, dim=0).to(device)
            
            init_test_acc, init_test_f1 = model.zero_shot_eval(node_embeds, label_embeds, test_graph_data, R=0)
            test_acc, test_f1 = model.zero_shot_eval(node_embeds, label_embeds, test_graph_data, R=args.R)
            
        print(f"{test_graph_name} {args.text_encoder} Classification -  Acc {init_test_acc} / Macro-F1 {init_test_f1}")
        print(f"{test_graph_name} {args.text_encoder}  + SGC Result  -  Acc {test_acc} / Macro-F1 {test_f1}\n")

    print("=" * 20, "\n\n")
