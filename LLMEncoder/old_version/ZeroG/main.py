from model import TextLoraModel, descriptions
import argparse
import torch 
from dataset import MyTextDataset
from torch_geometric.data import DataLoader
import time
import sys 
sys.path.append("../../")
from common import load_graph_dataset_for_zerog, get_cur_time


def build_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument("--model_dir", type=str, default="ckpts", help="Folder to save model")
    
    parser.add_argument("--epoch", type=int, default=15)
    parser.add_argument("--if_norm", action="store_true", default=True, help="Indicator of normalization")
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--text_encoder", type=str, default="SentenceBert", help="Type of text encoder")
    parser.add_argument("--test_R", type=int, default=4)

    parser.add_argument("--lr", type=float, default=5e-3, help="Learning rate of optimizer")
    parser.add_argument("--batch_size", type=int, default=32)
    
    parser.add_argument("--use_lora", type=int, default=1)
    
    args = parser.parse_args()
    
    if args.use_lora == 0:
        args.lr = 2e-5
    elif args.use_lora == 1 and args.text_encoder in ["roberta", "e5-large"]:
        args.lr = 1e-3

    return args 


def eval(test_gdata):
    model.eval()
    with torch.no_grad():
        text_features = []
        # for text in tqdm.tqdm(test_gdata.raw_texts, desc="[MODEL EVAL] Processing node texts"):
        for text in test_gdata.raw_texts:
            cur_text_feature = model.text_forward(text).cpu()
            text_features.append(cur_text_feature)
        
        desc = descriptions[args.dataset]
        text_features.append(model.text_forward(desc).cpu())
        node_embeds = torch.cat(text_features, dim=0).to(device)

        label_features = []
        # for text in tqdm.tqdm(test_gdata.label_text, desc="[MODEL EVAL] Processing label texts"):
        for text in test_gdata.label_text:
            cur_label_feature = model.text_forward(text).cpu()
            label_features.append(cur_label_feature)
        label_embeds = torch.cat(label_features, dim=0).to(device)

        res = model.zero_shot_eval(node_embeds, label_embeds, test_gdata)
        return res


if __name__ == "__main__":
    args = build_args()
    device = torch.device(args.device)
    
    print('= ' * 20)
    print('## Starting Time:', get_cur_time(), flush=True)
    print(args, "\n")

    # Step 1 - Load all Test Graphs
    graph_data = load_graph_dataset_for_zerog(args.dataset, device)
    print(f"[STAGE 1] Loading {args.dataset}'s graph ...")
    
    # Step 2 - Load Wrapped (L)LM Encoder Model
    model = TextLoraModel(args)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[STAGE 2] Preparing Text Encoder {args.text_encoder} with # Parameters {total_params} ...")

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=5e-4)
    
    # Step 3 - Preparing Training Data
    train_corpus = MyTextDataset(graph_data)
    train_dataloader = DataLoader(train_corpus, batch_size=args.batch_size, shuffle=True)
    print(f"[STAGE 3] Loading Training Pairs <text, label> from {args.dataset}, forming {len(train_dataloader)} train-loaders ...")
    
    # [Optional] Step 4 - Static (or Zero-shot) Evaluation
    res = eval(graph_data)
    print(f"[STAGE 4] Finish Zero-shot Evaluation with Performance {res}")
    
    # Step 5 - Model Training 
    best_eval_acc = 0 
    best_test_acc, best_test_f1 = 0, 0
    for i in range(args.epoch):
        model.train()

        start_time = time.time()
        epoch_loss = 0.0 
        for step, batch in enumerate(train_dataloader):
            # print(batch)
            optimizer.zero_grad()
            loss = model(batch, graph_data.label_text)
      
            if torch.isnan(loss).any():
                print(loss)
                break 
            
            epoch_loss += loss.item()
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
        # torch.cuda.empty_cache()
        print(f"[MODEL TRAINING] Epoch {i+1:03d} Loss {epoch_loss:.4f} Cost Times {time.time() - start_time:.3f}s")
        
        test_scores, eval_scores = eval(graph_data)
        if eval_scores[0] > best_eval_acc:
            best_eval_acc = eval_scores[0]
            best_test_acc, best_test_f1 = test_scores
        print(f"[MODEL EVAL] Epoch {i+1:03d} with Test Scores {test_scores}; Eval Scores {eval_scores}")
    
    print("\n\n")
    print(f"[{args.dataset}] Best Test Acc {best_test_acc:.3f} Best Test F1 {best_test_f1:.3f}")
    
    print('\n## Finishing Time:', get_cur_time(), flush=True)
    print('= ' * 20)
    print("Done!\n\n")
    