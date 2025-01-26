import torch
import numpy as np
import os
from transformers import AutoTokenizer, AutoModel, TrainingArguments, Trainer, IntervalStrategy
import sys
sys.path.append("../..")
from common import BertClassifier, BertClaInfModel
from common import load_graph_dataset_for_tape, compute_acc_and_f1, set_seed
from common import MODEL_PATHs as lm_paths


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx])
                for key, val in self.encodings.items()}
        item['node_id'] = idx
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])

        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def compute_metrics(p):
    from sklearn.metrics import accuracy_score
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    return {"accuracy": accuracy}


class LMTrainer():
    def __init__(self, cfg):
        self.dataset_name = cfg.dataset
        self.seed = cfg.seed
        
        self.model_name = cfg.lm.model.name
        self.feat_shrink = cfg.lm.model.feat_shrink

        self.weight_decay = cfg.lm.train.weight_decay
        self.dropout = cfg.lm.train.dropout
        self.att_dropout = cfg.lm.train.att_dropout
        self.cla_dropout = cfg.lm.train.cla_dropout
        self.batch_size = cfg.lm.train.batch_size
        self.epochs = cfg.lm.train.epochs
        self.warmup_epochs = cfg.lm.train.warmup_epochs
        self.eval_patience = cfg.lm.train.eval_patience
        self.grad_acc_steps = cfg.lm.train.grad_acc_steps
        self.lr = cfg.lm.train.lr

        self.use_gpt_str = cfg.lm.train.llm_name if cfg.lm.train.use_gpt else ""

        local_folder = f"../../results/TAPE"
        if cfg.re_split:
            self.re_split_str = '_s_' 
            self.output_dir = f'{local_folder}/{self.dataset_name}{self.re_split_str}{self.use_gpt_str}/{cfg.lm.model.name}-seed{self.seed}'
            self.ckpt_dir = f'{local_folder}/{self.dataset_name}{self.re_split_str}{self.use_gpt_str}/{cfg.lm.model.name}-seed{self.seed}'
        else:
            self.output_dir = f'{local_folder}/{self.dataset_name}{self.use_gpt_str}/{cfg.lm.model.name}-seed{self.seed}'
            self.ckpt_dir = f'{local_folder}/{self.dataset_name}{self.use_gpt_str}/{cfg.lm.model.name}-seed{self.seed}'
        
        if not os.path.exists(local_folder):
            os.makedirs(local_folder, exist_ok=True)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir, exist_ok=True)

        # Preprocess data
        self.device = torch.device("cuda:0" if cfg.device > 0 else "cpu")
        set_seed(self.seed)
        data, num_classes, text = load_graph_dataset_for_tape(self.dataset_name, 
                                                              self.device, 
                                                              use_gpt=cfg.lm.train.use_gpt, 
                                                              gpt_name=cfg.lm.train.llm_name, 
                                                              re_split=cfg.re_split)
        self.data = data
        self.num_nodes = data.y.size(0)
        self.n_labels = num_classes

        tokenizer = AutoTokenizer.from_pretrained(lm_paths[cfg.lm.model.name])
        if type(text)!=list:
            text = text.tolist()
        X = tokenizer(text, padding=True, truncation=True, max_length=cfg.lm.train.max_length)

        dataset = Dataset(X, data.y.tolist())
        self.inf_dataset = dataset

        self.train_dataset = torch.utils.data.Subset(
            dataset, self.data.train_mask.nonzero().squeeze().tolist())
        self.val_dataset = torch.utils.data.Subset(
            dataset, self.data.val_mask.nonzero().squeeze().tolist())
        self.test_dataset = torch.utils.data.Subset(
            dataset, self.data.test_mask.nonzero().squeeze().tolist())

        # Define pretrained tokenizer and model
        # bert_model = AutoModel.from_pretrained(self.model_name)
        bert_model = AutoModel.from_pretrained(lm_paths[cfg.lm.model.name])
        self.model = BertClassifier(bert_model,
                                    n_labels=self.n_labels,
                                    feat_shrink=self.feat_shrink)

        self.model.config.dropout = self.dropout
        self.model.config.attention_dropout = self.att_dropout

        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"\nNumber of parameters: {trainable_params}")

    def train(self):
        # Define training parameters
        # If there are more than 1 GPUs, multiply the number of GPUs
        eq_batch_size = self.batch_size * 1
        train_steps = self.num_nodes // eq_batch_size + 1
        eval_steps = self.eval_patience // eq_batch_size
        warmup_steps = int(self.warmup_epochs * train_steps)

        # Define Trainer
        args = TrainingArguments(
            output_dir=self.output_dir,
            do_train=True,
            do_eval=True,
            eval_steps=eval_steps,
            evaluation_strategy=IntervalStrategy.STEPS,
            save_steps=eval_steps,
            learning_rate=self.lr,
            weight_decay=self.weight_decay,
            save_total_limit=1,
            load_best_model_at_end=True,
            gradient_accumulation_steps=self.grad_acc_steps,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.batch_size*8,
            warmup_steps=warmup_steps,
            num_train_epochs=self.epochs,
            dataloader_num_workers=1,
            fp16=True,
            dataloader_drop_last=False,
        )
        self.trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.train_dataset,
            eval_dataset=self.val_dataset,
            compute_metrics=compute_metrics,
        )

        # Train pre-trained model
        self.trainer.train()
        # torch.save(self.model.state_dict(), f"{self.ckpt_dir}.ckpt")
        # print(f'LM saved to {self.ckpt_dir}.ckpt')

    @torch.no_grad()
    def eval_and_save(self):
        # if LM is set to SentenceBert, the dimension is 768
        emb = np.memmap(f"{self.ckpt_dir}.emb",
                        dtype=np.float16,
                        mode='w+',
                        shape=(self.num_nodes, self.feat_shrink if self.feat_shrink else 1024))
        pred = np.memmap(f"{self.ckpt_dir}.pred",
                         dtype=np.float16,
                         mode='w+',
                         shape=(self.num_nodes, self.n_labels))

        inf_model = BertClaInfModel(
            self.model, emb, pred, feat_shrink=self.feat_shrink)
        inf_model.eval()
        inference_args = TrainingArguments(
            output_dir=self.output_dir,
            do_train=False,
            do_predict=True,
            per_device_eval_batch_size=self.batch_size*4,
            dataloader_drop_last=False,
            dataloader_num_workers=1,
            fp16_full_eval=True,
        )

        trainer = Trainer(model=inf_model, args=inference_args)
        trainer.predict(self.inf_dataset)
        
        accuracy, macrof1_scores, weightf1_scores = [], [], []
        pred = torch.tensor(pred).to(self.device)
        for mask in [self.data.train_mask, self.data.val_mask, self.data.test_mask]:
            acc, macro_f1, weight_f1 = compute_acc_and_f1(np.argmax(pred[mask].cpu().numpy(), -1), self.data.y[mask].cpu().numpy())
            accuracy.append(acc)
            macrof1_scores.append(macro_f1)
            weightf1_scores.append(weight_f1)
        
        train_acc, val_acc, test_acc = accuracy
        train_f1, val_f1, test_f1 = macrof1_scores
        train_weightf1, val_weightf1, test_weightf1 = weightf1_scores

        print(f'[LM] TrainAcc: {train_acc:.3f}, ValAcc: {val_acc:.3f}, TestAcc: {test_acc:.3f}')
        print(f'[LM] TrainF1 {train_f1:.3f}, ValF1 {val_f1:.3f}, TestF1 {test_f1:.3f}')
        print(f'[LM] Train WeightF1 {train_weightf1:.3f}, Val WeightF1 {val_weightf1:.3f}, Test WeightF1 {test_weightf1:.3f}')
        return {'TrainAcc': train_acc, 'ValAcc': val_acc, 'TestAcc': test_acc,
                'TrainF1': train_f1, 'ValF1': val_f1, 'TestF1': test_f1,
                'Train WeightF1': train_weightf1, 'Val WeightF1': val_weightf1, 'Test WeightF1': test_weightf1}
