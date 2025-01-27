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
For additional hyperparameter settings for each dataset, please refer to `reproduce.sh`, which contains configurations using Mistral-7B as the backbone. Configurations for other LLMs will be provided soon.



## 🙏 Acknowledgements 

We express our gratitude for the official implementations of [GraphGPT](https://github.com/HKUDS/GraphGPT) and [LLaGA](https://github.com/VITA-Group/LLaGA).
