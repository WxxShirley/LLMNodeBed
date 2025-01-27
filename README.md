# LLMNodeBed

This repository is the official implementation of the paper: **A Comprehensive Analysis on LLM-based Node Classification Algorithms**. It provides a standardized framework for evaluating LLM-based node classification methods, including **10 datasets**, **8 LLM-based algorithms**, and **3 learning paradigms**. Designed for flexibility, researchers can easily integrate new datasets or algorithms.


## 📝 Table of Contents
- [🚀 Quick Start](#-quick-start)
- [📖 Code Structure](#-code-structure)
- [🔧 Supported Methods](#-supported-methods)
- [🙏 Acknowledgements](#-acknowledgements)

---


## 🚀 Quick Start 

### 0. Environment Setup
Create a conda environment and install dependencies:
```bash
conda create -n NodeBed python=3.10
conda activate NodeBed
pip install torch torch_geometric transformers peft pytz scikit-learn torch_scatter torch_sparse
``` 


### 1. LLM Preparation 

* **Close-source LLMs like GPT-4o, DeepSeek-Chat**: 

  Add API keys to `LLMZeroShot/Direct/api_keys.py`

* **Open-source LLMs like Mistral-7B, Qwen**:

  Download models from HuggingFace (e.g., [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)). Then, update model paths in `common/model_path.py`.

  Example paths: 
  ```python 
  MODEL_PATHs = {
    "MiniLM": "sentence-transformers/all-MiniLM-L6-v2",
    "Mistral-7B": "mistralai/Mistral-7B-Instruct-v0.2",
    "Llama-8B": "meta-llama/Llama-3.1-8B-Instruct",
    # See full list in common/model_path.py
  }
  ```


### 2. Datasets 

Download datasets from [Google Drive](https://drive.google.com/file/d/14GmRVwhP1pUD_OIhoJU3oATZWTnklhPG/view) and unzip into the `datasets` folder.

Before running LLM-based algorithms, please generate LM / LLM-encoded embeddings as follows: 
```shell 
cd LLMEncoder/GNN
python3 embedding.py --dataset=cora --encoder_name=roberta      # LM embeddings
python3 embedding.py --dataset=cora --encoder_name=Mistral-7B  # LLM embeddings
```


### 3. (Optional) Deploy Local LLMs

For LLM Direct Inference using open-source LLMs, we depoly them as local services based on the [FastChat](https://github.com/lm-sys/FastChat) framework.

```shell 
# Install dependencies
pip install vllm "fschat[model_worker,webui]"

# Start services
python3 -m fastchat.serve.controller --host 127.0.0.1
CUDA_VISIBLE_DEVICES=0 python3 -m fastchat.serve.vllm_worker --model-path mistralai/Mistral-7B-Instruct-v0.2 --host 127.0.0.1
python3 -m fastchat.serve.openai_api_server --host 127.0.0.1 --port 8008
```

Then, the Mistral-7B model can be invoked via the url `http://127.0.0.1:8008/v1/chat/completions`. 



### 4. Run Algorithms


Refer to method-specific READMEs for execution details:

* **LLM-as-Encoder**: `LLMEncoder/`

* **LLM-as-Predictor**: `LLMPredictor/`

* **LLM-as-Reasoner**: `LLMReasoner/`

* **Zero-shot Methods**: `LLMZeroShot/`



## 📖 Code Structure 

```
LLMNodeBed/
├── LLMEncoder/           # LLM-as-Encoder (GNN, ENGINE)
├── LLMPredictor/         # LLM-as-Predictor (GraphGPT, LLaGA, Instruction Tuning)
├── LLMReasoner/          # LLM-as-Reasoner (TAPE)
├── LLMZeroShot/          # Zero-shot Methods (Direct Inference, ZeroG)
├── common/               # Shared utilities
├── datasets/             # Dataset storage
├── results/              # Experiment outputs
└── requirements.txt
```


## 🔧 Supported Methods 


| Method                       | Veneue   | Official Implementation                    | Our Implementation                |
| ---------------------------- | -------- | ------------------------------------------ | --------------------------------- |
| TAPE                         | ICLR'24  | [link](https://github.com/XiaoxinHe/TAPE)  | `LLMReasoner/TAPE`                |
| ENGINE                       | IJCAI'24 | [link](https://github.com/ZhuYun97/ENGINE) | `LLMEncoder/ENGINE`               |
| GraphGPT                     | SIGIR'24 | [link](https://github.com/HKUDS/GraphGPT)  | `LLMPredictor/GraphGPT`           |
| LLaGA                        | ICML'24  | [link](https://github.com/VITA-Group/LLaGA)| `LLMPredictor/LLaGA`              |
| ZeroG                        | KDD'24   | [link](https://github.com/NineAbyss/ZeroG) | `LLMZeroShot/ZeroG`               |
| $\text{GNN}_{\text{LLMEmb}}$ | -        | Ours Proposed                              | `LLMEncoder/GNN`                  |
| LLM Instruction Tuning       | -        | Ours Implemented                           | `LLMPredictor/Instruction Tuning` |
| Direct Inference             | -        | Ours Implemented                           | `LLMZeroShot/Direct`              |

## 🙏 Acknowledgements

We thank the authors of TAPE, ENGINE, GraphGPT, LLaGA, and ZeroG for their open-source implementations. Part of our framework is inspired by [GLBench](https://github.com/NineAbyss/GLBench).
