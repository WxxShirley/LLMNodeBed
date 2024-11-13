import argparse 
from peft import LoraConfig, get_peft_model 
from transformers import TrainingArguments, Trainer, IntervalStrategy, AutoModelForCausalLM, AutoTokenizer
import random
import torch
from dataset import TextDataset
import os
import sys 
import json
import csv
import time
sys.path.append("../..")
from common import set_seed, load_graph_dataset_for_zerog, compute_acc_and_f1


# TODO: this is duplicate with LLaGA/dataset.py 
# TODO: abstract these parts into the `common` folder
descriptions = {
    "cora": "Given a node-centered graph with centric node description: {{node}}, each node represents a paper, we need to classify the center node into 7 classes: Case_Based, Genetic_Algorithms, Neural_Networks, Probabilistic_Methods, Reinforcement_Learning, Rule_Learning, Theory, please tell me which class the center node belongs to?",
    "pubmed": "Given a node-centered graph with centric node description: {{node}}, each node represents a paper about Diabetes, we need to classify the center node into 3 classes: Experimentally induced diabetes, Type 1 diabetes, Type 2 diabetes, please tell me which class the center node belongs to?",
    "arxiv": "Given a node-centered graph with centric node description: {{node}}, we need to classify the center node into 40 classes: cs.NA(Numerical Analysis), cs.MM(Multimedia), cs.LO(Logic in Computer Science), cs.CY(Computers and Society), cs.CR(Cryptography and Security), cs.DC(Distributed, Parallel, and Cluster Computing), cs.HC(Human-Computer Interaction), cs.CE(Computational Engineering, Finance, and Science), cs.NI(Networking and Internet Architecture), cs.CC(Computational Complexity), cs.AI(Artificial Intelligence), cs.MA(Multiagent Systems), cs.GL(General Literature), cs.NE(Neural and Evolutionary Computing), cs.SC(Symbolic Computation), cs.AR(Hardware Architecture), cs.CV(Computer Vision and Pattern Recognition), cs.GR(Graphics), cs.ET(Emerging Technologies), cs.SY(Systems and Control), cs.CG(Computational Geometry), cs.OH(Other Computer Science), cs.PL(Programming Languages), cs.SE(Software Engineering), cs.LG(Machine Learning), cs.SD(Sound), cs.SI(Social and Information Networks), cs.RO(Robotics), cs.IT(Information Theory), cs.PF(Performance), cs.CL(Computational Complexity), cs.IR(Information Retrieval), cs.MS(Mathematical Software), cs.FL(Formal Languages and Automata Theory), cs.DS(Data Structures and Algorithms), cs.OS(Operating Systems), cs.GT(Computer Science and Game Theory), cs.DB(Databases), cs.DL(Digital Libraries), cs.DM(Discrete Mathematics), please tell me which class the center node belongs to?",
    "citeseer": "Given a node-centered graph with centric node description: {{node}}, each node represents a paper, we need to classify the center node into 6 classes: Agents, ML (Machine Learning), IR (Information Retrieval), DB (Databases), HCI (Human-Computer Interaction), AI (Artificial Intelligence), please tell me which class the center node belongs to?",
    "wikics": "Given a node-centered graph with centric node description: {{node}}, each node represents an entity, we need to classify the center node into 10 classes: Computational Linguistics, Databases, Operating Systems, Computer Architecture, Computer Security, Internet Protocols, Computer File Systems, Distributed Computing Architecture, Web Technology, Programming Language Topics, please tell me which class the center node belongs to?",
    "reddit": "Given a node-centered graph with centric node description: {{node}}, each node represents an user, we need to classify the center node into 2 classes: Normal Users and Popular Users, please tell me which class the center node belongs to?",
    "instagram": "Given a node-centered graph with centric node description: {{node}}, each node represents an user, we need to classify the center node into 2 classes: Normal Users and Commercial Users, please tell me which class the center node belongs to?",
}

classes = {
    "arxiv": ["cs.NA(Numerical Analysis)", "cs.MM(Multimedia)", "cs.LO(Logic in Computer Science)", "cs.CY(Computers and Society)", "cs.CR(Cryptography and Security)", "cs.DC(Distributed, Parallel, and Cluster Computing)", "cs.HC(Human-Computer Interaction)", "cs.CE(Computational Engineering, Finance, and Science)", "cs.NI(Networking and Internet Architecture)", "cs.CC(Computational Complexity)", "cs.AI(Artificial Intelligence)", "cs.MA(Multiagent Systems)", "cs.GL(General Literature)", "cs.NE(Neural and Evolutionary Computing)", "cs.SC(Symbolic Computation)", "cs.AR(Hardware Architecture)", "cs.CV(Computer Vision and Pattern Recognition)", "cs.GR(Graphics)", "cs.ET(Emerging Technologies)", "cs.SY(Systems and Control)", "cs.CG(Computational Geometry)", "cs.OH(Other Computer Science)", "cs.PL(Programming Languages)", "cs.SE(Software Engineering)", "cs.LG(Machine Learning)", "cs.SD(Sound)", "cs.SI(Social and Information Networks)", "cs.RO(Robotics)", "cs.IT(Information Theory)", "cs.PF(Performance)", "cs.CL(Computational Complexity)", "cs.IR(Information Retrieval)", "cs.MS(Mathematical Software)", "cs.FL(Formal Languages and Automata Theory)", "cs.DS(Data Structures and Algorithms)", "cs.OS(Operating Systems)", "cs.GT(Computer Science and Game Theory)", "cs.DB(Databases)", "cs.DL(Digital Libraries)", "cs.DM(Discrete Mathematics)"],
    "cora": ['Rule_Learning', 'Neural_Networks', 'Case_Based', 'Genetic_Algorithms', 'Theory', 'Reinforcement_Learning', 'Probabilistic_Methods'],
    "pubmed": ['Experimentally induced diabetes', 'Type 1 diabetes', 'Type 2 diabetes'],
    "citeseer": ['Agents', 'ML (Machine Learning)', 'IR (Information Retrieval)', 'DB (Databases)', 'HCI (Human-Computer Interaction)', 'AI (Artificial Intelligence)'],
    "wikics": ['Computational Linguistics', 'Databases', 'Operating Systems', 'Computer Architecture', 'Computer Security', 'Internet Protocols', 'Computer File Systems', 'Distributed Computing Architecture', 'Web Technology', 'Programming Language Topics'],
    "reddit": ['Normal Users', 'Popular Users'],
    "instagram": ['Normal Users', 'Commercial Users']
}


BOS = '<s>[INST]'
EOS_USER = '[/INST]'
EOS = '</s>'
IGNORE_INDEX = -100 


def prepare_graph_instruction_tuning_data(graph_data, data_type="train"):
    focus_mask = {"train": graph_data.train_mask, "val": graph_data.val_mask, "test": graph_data.test_mask}[data_type]
    focus_nodes = focus_mask.nonzero(as_tuple=False).squeeze().detach().cpu().numpy().tolist()
    
    data_contents = []
    for node_id in focus_nodes: 
        origin_txt = graph_data.raw_texts[node_id]
        label = classes[args.dataset][graph_data.y[node_id].item()]
        
        data_contents.append({
            "id": node_id, 
            "input": origin_txt, 
            "output": label
        })
    
    if data_type != "test":      
        random.shuffle(data_contents)
    # print(data_contents[0])
    return data_contents


def tokenizer_instruction_tuning_data(raw_data, max_txt_length=128, max_origin_txt_length=128, max_ans_length=16):
    # Full Input: BOS + Query + EOS_USER + Label + EOS
    bos_tokens = tokenizer(BOS, add_special_tokens=False)
    eos_user_tokens = tokenizer(EOS_USER, add_special_tokens=False)
    eos_tokens = tokenizer(EOS, add_special_tokens=False)
    
    full_input_ids, full_attention_masks, full_labels = [], [], [] 
    # input_length, label_length = [], []
    for sample in raw_data:
        input_left, input_right = descriptions[args.dataset].split("{{node}}")
        # print(input_left, input_right)
        tokenized_input_left, tokenizer_input_right = tokenizer(input_left, add_special_tokens=False), tokenizer(input_right, add_special_tokens=False)
        tokenizer_input_right_ids = tokenizer_input_right.input_ids[:max_txt_length]
        
        tokenized_origin_txt = tokenizer(sample["input"], add_special_tokens=False)
        origin_txt_ids = tokenized_origin_txt.input_ids[:max_origin_txt_length] 
        
        tokenied_label = tokenizer(sample["output"], add_special_tokens=False)
        label_ids = tokenied_label.input_ids[:max_ans_length] + eos_tokens.input_ids 
        input_ids = bos_tokens.input_ids + tokenized_input_left.input_ids + origin_txt_ids + tokenizer_input_right_ids + eos_user_tokens.input_ids + label_ids
        label_ids = [IGNORE_INDEX] * (len(input_ids) - len(label_ids)) + label_ids
        
        full_input_ids.append(input_ids)
        full_attention_masks.append([1] * len(input_ids))
        full_labels.append(label_ids)
        
        # Statistics for Choosing Suitable Maximum Lengths
        # input_length.append(len(tokenized_origin_txt.input_ids) + len(tokenizer_input_right.input_ids))
        # label_length.append(len(tokenied_label.input_ids))
        
    max_length = max([len(x) for x in full_input_ids])
    # print(f"Avg Input Length {sum(input_length)/len(input_length):.4f}  Avg Output Length {sum(label_length)/len(label_length):.4f}  Maximum Output Length {max_length}")
    
    for i in range(len(full_input_ids)):
        pad_length = max_length - len(full_input_ids[i])
        full_input_ids[i] = [0] * pad_length + full_input_ids[i]
        full_attention_masks[i] = [0] * pad_length + full_attention_masks[i]
        full_labels[i] = [IGNORE_INDEX] * pad_length + full_labels[i]
        
    input_ids = torch.tensor(full_input_ids).to(device)
    attention_mask = torch.tensor(full_attention_masks).to(device)
    label_input_ids = torch.tensor(full_labels).to(device)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": label_input_ids
    }


def tokenizer_test_data(batch_data, max_txt_length=128, max_origin_txt_length=128):
    bos_tokens = tokenizer(BOS, add_special_tokens=False)
    eos_user_tokens = tokenizer(EOS_USER, add_special_tokens=False)

    full_input_ids, full_attention_masks = [], []
    hint_prompt = " Only output the most possible category without any extra contents. "
    for sample in batch_data:
        input_left, input_right = descriptions[args.dataset].split("{{node}}")
        tokenized_input_left, tokenizer_input_right = tokenizer(input_left, add_special_tokens=False), tokenizer(input_right + hint_prompt, add_special_tokens=False)
        tokenizer_input_right_ids = tokenizer_input_right.input_ids[:max_txt_length]
        
        tokenized_origin_txt = tokenizer(sample["input"], add_special_tokens=False)
        origin_txt_ids = tokenized_origin_txt.input_ids[:max_origin_txt_length] 
        
        input_ids = bos_tokens.input_ids + tokenized_input_left.input_ids + origin_txt_ids + tokenizer_input_right_ids + eos_user_tokens.input_ids
        full_input_ids.append(input_ids)
        full_attention_masks.append([1] * len(input_ids))

    max_length = max([len(x) for x in full_input_ids])

    for i in range(len(full_input_ids)):
        pad_length = max_length - len(full_input_ids[i])
        full_input_ids[i] = torch.tensor([0] * pad_length + full_input_ids[i]).to(device)
        full_attention_masks[i] = torch.LongTensor([0] * pad_length + full_attention_masks[i]).to(device)
    
    input_ids = torch.stack(full_input_ids).to(device)
    attention_mask = torch.stack(full_attention_masks).to(device)
    
    return input_ids, attention_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default='cora')
    parser.add_argument('--llm', type=str, default="Qwen-3B")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--max_txt_length', type=int, default=128, help="Maximum length of query prompt") 
    parser.add_argument('--max_origin_txt_length', type=int, default=128, help="Maximum length of node's original text")
    parser.add_argument('--max_ans_length', type=int, default=16, help="Maximum length of answer")
    parser.add_argument('--num_epoch', type=int, default=2)
    parser.add_argument('--re_split', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=8) 
    args = parser.parse_args()
    print(args, "\n")
    
    set_seed(args.seed)
    device = torch.device(args.device) 
    
    llm_path = {
        "Qwen-3B": "/root/autodl-tmp/models/qwen/Qwen2___5-3B-Instruct", # 3B
        "Mistral-7B": "/root/autodl-tmp/models/Mistral-7B/snapshots/Mistral-7B-Instruct-v0.2", # 7B
    }[args.llm]
    
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    tokenizer.pad_token_id = 0 
    tokenizer.padding_side = 'left'
    # TODO: you can adjust the GPU setting based on your own device
    kwargs = {'max_memory': {0: '48GiB'}, 'device_map': "auto"}
    model = AutoModelForCausalLM.from_pretrained(llm_path, **kwargs)
    
    peft_config = LoraConfig(
        task_type="CAUSAL_LM", 
        inference_mode=False, 
        r=8, 
        lora_alpha=16, 
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )
    ft_model = get_peft_model(model, peft_config)
    ft_model.print_trainable_parameters()
    
    # Prepare training and validation data <origin_txt, label>
    graph_data = load_graph_dataset_for_zerog(dataset_name=args.dataset, device=device, prefix="../..", re_split=args.re_split)
    train_contents = prepare_graph_instruction_tuning_data(graph_data, "train")
    val_contents = prepare_graph_instruction_tuning_data(graph_data, "val") 
    test_contents = prepare_graph_instruction_tuning_data(graph_data, "test")
    
    train_encodings = tokenizer_instruction_tuning_data(train_contents, max_txt_length=args.max_txt_length, max_origin_txt_length=args.max_origin_txt_length, max_ans_length=args.max_ans_length)
    train_dataset = TextDataset(train_encodings) 
    val_encodings = tokenizer_instruction_tuning_data(val_contents, max_txt_length=args.max_txt_length, max_origin_txt_length=args.max_origin_txt_length, max_ans_length=args.max_ans_length)
    val_dataset = TextDataset(val_encodings)

    print(len(train_dataset), len(val_dataset), len(test_contents))
    
    save_dir = f"output/{args.dataset}_{args.llm}"
    
    training_args = TrainingArguments(
        output_dir=save_dir, 
        learning_rate=1e-5, 
        per_device_eval_batch_size=args.batch_size * 2, 
        per_device_train_batch_size=args.batch_size, 
        num_train_epochs=args.num_epoch, 
        weight_decay=0.01, 
        eval_steps=20, 
        evaluation_strategy=IntervalStrategy.STEPS,
        save_steps=20,
        save_total_limit=1,
        load_best_model_at_end=True
    )
    
    trainer = Trainer(
        model=ft_model, 
        args=training_args,
        train_dataset=train_dataset, 
        eval_dataset=val_dataset,  
        tokenizer=tokenizer
    )
    
    st_time = time.time()
    trainer.train()
    train_secs = time.time() - st_time

    # Load well-trained model 
    batch_size = args.batch_size * 2
    write_dir = "prediction"
    os.makedirs(write_dir, exist_ok=True)
    write_file = open(f"{write_dir}/{args.dataset}_{args.llm}.json", "w")
    pred_labels, gt_labels = [], [] 
    
    st_time = time.time()
    for i in range(0, len(test_contents), batch_size):
        batch_data = test_contents[i: min(i+batch_size, len(test_contents))]
        batch_input_ids, batch_attention_mask = tokenizer_test_data(batch_data, max_txt_length=args.max_txt_length, max_origin_txt_length=args.max_origin_txt_length)
        output = ft_model.generate(input_ids=batch_input_ids, 
                                   attention_mask=batch_attention_mask, 
                                   max_new_tokens=args.max_ans_length)
        decode_output = tokenizer.batch_decode(output, skip_special_tokens=True)
        origin_prompts = tokenizer.batch_decode(batch_input_ids, skip_special_tokens=True)
        
        # print(origin_prompts)
        for idx, pred_content in enumerate(decode_output):
            origin_prompt = origin_prompts[idx] 
            label, node_id = batch_data[idx]["output"], batch_data[idx]["id"]
            
            pred_label = pred_content.replace(origin_prompt, "")
            write_content = {
                "idx": node_id, 
                "ground-truth": label,
                "pred": pred_label
            }
            print(write_content)
            write_file.write(json.dumps(write_content) + "\n")
            write_file.flush()
            
            pred_labels.append(pred_label.strip(" "))
            gt_labels.append(label.strip(" "))
    inference_secs = time.time() - st_time
    
    acc, f1 = compute_acc_and_f1(pred_labels, gt_labels)
    print(f"Accuracy {acc:.3f}  F1-Score {f1:.3f}")
    
    with open(f"summary.csv", 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([args.dataset, args.llm, acc, f1, "Semi" if not args.re_split else "Supervised", f"Seed-{args.seed}", f"Batch Size-{args.batch_size}", f"Epoch-{args.num_epoch}", f"Train Hours-{train_secs/3600:.3f}", f"Inference Seconds-{inference_secs:.2f}"])
    