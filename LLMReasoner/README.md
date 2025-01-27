# LLM-as-Reasoner Method 

This folder contains the LLM-as-Reasoner method: [TAPE](https://arxiv.org/abs/2305.19523). 


## 🚀 Usage 

### Step 0 - Environment Setup 

In addition to the required Python packages like PyTorch, PyG, and Transformers, you need to download the following models:
* **LM**: Default is [RoBERTa-355M](https://huggingface.co/sentence-transformers/all-roberta-large-v1)
* **LLM**: Default is [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)

The model paths are configured in `common/model_path.py`, and you can update them based on the actual paths on your device.
 

### Step 1 - Explanation Generation 

We provide code for generating explanations using both open-source and closed-source LLMs:

* `explanation_gpt.py`: Invoke OpenAI GPT models to generate reasoning text 
* `explanation_llm.py`: Load local LLMs like Mistral-7B to generate reasoning text 

**Example Script:**

Specify the `dataset`, `llm`, and `batch_size` (which should be determined based on your GPU device, e.g., 16 is suitable for an NVIDIA A6000 48G GPU):
```shell 
cd TAPE 

python3 explanation_llm.py --dataset=cora --llm=Mistral-7B --batch_size=16
```

The generated explanation files will be automatically saved under the `results/TAPE/{llm}` folder, named as `{dataset}.json`.


### Step 2 - Fine-tuning LMs based on Original and Generated Texts 

Fine-tune LMs with original text attributes 
```shell 
WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False  CUDA_VISIBLE_DEVICES=0,1 python -m trainLM dataset cora runs 4 lm.train.epochs 10 

# Supervised Settings 
WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=0,1 python -m trainLM dataset cora runs 4 re_split 1 lm.train.epochs 6
```


Fine-tune LMs with LLM-generated text attributes 

Set `lm.train.use_gpt` to `True` and speficy the LLM name in `lm.train.llm_name`

```shell 
WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=0,1 python -m trainLM dataset cora runs 4  lm.train.epochs 10 lm.train.use_gpt True   lm.train.llm_name Mistral-7B 
```

**Notice**:

1.  Set the `lm.train.batch_size` to adjust the batch size according to your device. For example, the H100-80G can use the default value of `32`, while two A6000-48G GPUs require a reduction of the batch size to `16`.
    ```shell
    WANDB_DISABLED=True  CUDA_VISIBLE_DEVICES=0,1 TOKENIZERS_PARALLELISM=False  python -m trainLM dataset cora runs 4 lm.train.epochs 10  lm.train.batch_size 16 
    ```

2. If you wish to change the default LM encoder to another one, you need to modify the following three parts:

   * Change the encoder name argument in `lm.model.name`:
     ```shell
     WANDB_DISABLED=True  CUDA_VISIBLE_DEVICES=0,1 TOKENIZERS_PARALLELISM=False  python -m trainLM dataset cora runs 4 lm.train.epochs 10  lm.model.name SentenceBert
     ```
   * Adjust the embedding dimension according to the updated encoder's output embedding dimension in `TAPE/lm_trainer.py`:
     ```python 
     # Line 156
     def eval_and_save(self):
        # if LM is set to SentenceBert, the dimension is 768
        # if LM is set to roberta, the dimension is 1024
        emb = np.memmap(f"{self.ckpt_dir}.emb",
                        dtype=np.float16,
                        mode='w+',
                        shape=(self.num_nodes, self.feat_shrink if self.feat_shrink else 768))
     ``` 

     In `TAPE/gnn_trainer.py`
     ```python 
     # Line 37
     # change the embedding dimension based on speific encoder
     if self.feature_type == 'TA':
        print("Loading pretrained LM features (title and abstract) ...")
        # ... 
        features = torch.from_numpy(np.array(
        np.memmap(LM_emb_path, mode='r',
                  dtype=np.float16,
                  shape=(self.num_nodes, 768)))
        ).to(torch.float32)
     ```
  

### Step 3 - Training GNNs for Classification 

You need to speficy both the `feature_type` and `llm_name` as follows: 
```shell 
python -m trainEnsemble dataset cora runs 4 gnn.train.feature_type TA_E  lm.train.llm_name Mistral-7B 

# The GNN configuration can be customized (for more settings, refer to `config.py`)
python -m trainEnsemble dataset cora runs 4 gnn.train.feature_type TA_E  lm.train.llm_name Mistral-7B  gnn.model.num_layers 3 gnn.model.name SAGE 

```


### Reproducibility 

We provide reproducibility scripts in `run.sh`. We will release both GPT-4o and Mistral-7B generated texts soon.


## 🙏 Acknowledgement 

We extend our heartfelt thanks to the [official implementation](https://github.com/XiaoxinHe/TAPE) of the TAPE method!
