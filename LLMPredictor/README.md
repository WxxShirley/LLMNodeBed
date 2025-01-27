# LLM-as-Predictor Methods 

This folder contains three LLM-as-Predictor methods: `InstructionTuning` for LLM Instruction Tuning, `GraphGPT` for the [GraphGPT](https://arxiv.org/abs/2310.13023) algorithm, and `LLaGA` for the [LLaGA](https://arxiv.org/abs/2402.08170) algorithm. 


## 🚀 Usage 

### InstructionTuning 

The code is located in the `InstructionTuning` folder.

You can execute LLM instruction tuning by running `main.py`.


**Example Scripts:**
```shell
cd InstructionTuning 

# Semi-supervised setting on 2 NVIDIA A6000-48G GPUs
WANDB_DISABLED=True python main.py --num_epoch=20 --llm=Mistral-7B  --batch_size=16  --re_split=0 --dataset=cora

# Supervised setting
WANDB_DISABLED=True python main.py --num_epoch=4 --llm=Mistral-7B  --batch_size=16  --re_split=1  --dataset=cora
```

For additional hyperparameter settings (e.g., number of epochs, batch size, maximum input/output text lengths), please refer to `reproduce.sh`.

You can specify the available GPUs using `CUDA_VISIBLE_DEVICES=1,2` (for GPUs 1 and 2)

Besides, in `InstructionTuning/main.py`, you can adjust the GPU configurations based on your own device: 
```python 
# Lines 150 - 154 in InstructionTuning/main.py
kwargs = {'max_memory': {0: '80GiB'}, 'device_map': "auto"}
if args.num_gpus == 2:
    # Example for 2 x A6000-48G GPUs
    kwargs = {'max_memory': {0: '48GiB', 1: '48GiB'}, 'device_map': "auto"}
model = AutoModelForCausalLM.from_pretrained(llm_path, **kwargs) 
```

The example commands in `reproduce.sh` can be successfully executed on a single H100-80G GPU or two A6000-48G GPUs. If you encounter CUDA OOM errors, please **adjust the `batch_size`**.


### GraphGPT 

***Preparation*** Before running `main.py` for GraphGPT, you need to prepare LM-encoded node attributes for the dataset, which will be fed into the LLMs as graph context.
```shell 
cd LLMEncoder/GNN

# Use RoBERTa-355M to encode nodes
python3 embedding.py --dataset=cora --encoder_name=roberta
```

***Execution*** The three-stages (text-graph grounding, graph matching, and classification insturcition tuning) are optional. 
```shell 
cd GraphGPT 

# Skip the text-graph grounding stage
#  GraphMatching runs for 1 epoch and InstructionTuning runs for 20 epochs (in semi-supervised settings)
python3 -u main.py --dataset=cora --do_stage1=1 --seed=0 --load_ground_embedding=0 --re_split=0 --s2_epoch=20 --gpu_id=0 --s2_batch_size=16 
```

For more hyperparameter settings for each dataset, please refer to `reproduce.sh`.


### LLaGA 
Similar to GraphGPT, before running `main.py` for LLaGA, you need to **prepare LM-encoded node attributes** for the dataset, which will be fed into the LLMs as graph context.


**Example Scripts:**

For the semi-supervised setting on Cora, the epochs are set to 12 with a batch size of 64. 
```shell
cd LLaGA 

python3 -u main.py --neighbor_template=HO --dataset=cora --re_split=0  --num_epochs=12 --llm=Mistral-7B --patience=4 
```
For additional hyperparameter settings for each dataset, please refer to `run.sh`, which contains configurations using Mistral-7B or Qwen-3B as the backbone. Configurations for other LLMs will be provided soon.

You can change the GPU configurations in `LLaGA/llaga_model.py` based on your own devices: 
```python
# TODO: please change the following configurations based on your own device
kwargs = { 
    "max_memory": {args.gpu_id: '80GiB'}, # 1 H100-80G
    "device_map": "auto",
}

if args.num_gpus == 2:
    kwargs = {
        "max_memory": {0: '48GiB', 1: '48GiB'}, # 2 A6000-48G
        "device_map": "auto",
    }
```


## 🙏 Acknowledgements 

We express our gratitude for the official implementations of [GraphGPT](https://github.com/HKUDS/GraphGPT) and [LLaGA](https://github.com/VITA-Group/LLaGA).
