import json 
import argparse
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer
import torch
from main import descriptions, classes
import os
import sys 
sys.path.append("../..")
from common import set_seed, load_graph_dataset_for_zerog, compute_acc_and_f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--llm", type=str, default="Qwen-3B")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--load_lora", type=bool, default=True)
    parser.add_argument("--re_split", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_generation_length", type=int, default=512)
    args = parser.parse_args()
    print(args, "\n")
    set_seed(args.seed)
    
    device = torch.device(args.device)
    llm_path = {
        "Qwen-3B": "/root/autodl-tmp/models/qwen/Qwen2___5-3B-Instruct", # 3B
        "Mistral-7B": "/root/autodl-tmp/models/Mistral-7B/snapshots/Mistral-7B-Instruct-v0.2", # 7B
    }[args.llm]
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Automatically search for available checkpoints
    save_dir = f"output/{args.dataset}_{args.llm}" 
    files = os.listdir(save_dir)
    lora_ckpt_dir = None
    for file in files:
        file_path = os.path.join(save_dir, file) 
        
        if os.path.isdir(file_path) and "checkpoint-" in file_path:
            lora_ckpt_dir = file_path 
    
    # print(lora_ckpt_dir)
    assert lora_ckpt_dir is not None, "Current output folder does not have a valid LORA model!"
    
    model = AutoPeftModelForCausalLM.from_pretrained(lora_ckpt_dir).to(device)
    graph_data = load_graph_dataset_for_zerog(dataset_name=args.dataset, device=device, prefix="../..", re_split=args.re_split)
    test_node_ids = graph_data.test_mask.nonzero(as_tuple=False).squeeze().detach().cpu().numpy().tolist()
     
    write_dir = "prediction"
    os.makedirs(write_dir, exist_ok=True)
    write_file = open(f"{write_dir}/{lora_ckpt_dir.split('/')[1]}.json", "w")
    
    pred_labels, gt_labels = [], []
    for test_node in test_node_ids: 
        origin_txt = graph_data.raw_texts[test_node][:2048] # cut up the over-length text
        cur_prompt = descriptions[args.dataset].replace("{{node}}", origin_txt)
        
        tokens = tokenizer(cur_prompt, return_tensors="pt")
        input_ids = tokens['input_ids'].to(device)
        ground_truth_label = classes[args.dataset][graph_data.y[test_node].item()]
        try: 
            output = model.generate(input_ids, max_length=args.max_generation_length, temperature=0.9)
            decode_output = tokenizer.batch_decode(output, skip_special_tokens=True)[0]
            llm_pred = decode_output.replace(cur_prompt, "")
        except Exception as e:
            print(f"Node {test_node} Encounter Error {e}")
            pred_labels.append("Error Prediction")
            gt_labels.append(ground_truth_label)
            continue
        
        pred_label = llm_pred[:llm_pred.index("</s>")] if "</s>" in llm_pred  else llm_pred
        write_content = {
            "id": test_node, 
            "origin_txt": origin_txt, 
            "ground-truth": ground_truth_label, 
            "pred": llm_pred
        }
        pred_labels.append(pred_label) 
        gt_labels.append(ground_truth_label)
        
        write_file.write(json.dumps(write_content) + "\n")
        write_file.flush()
    
    acc, f1 = compute_acc_and_f1(pred_labels, gt_labels)
    print(f"Accuracy {acc:.3f}  F1-Score {f1:.3f}")
    