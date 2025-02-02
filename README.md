# LLMNodeBed
<p align="center">
   <a href="https://llmnodebed.github.io/"><img src="https://img.shields.io/badge/🌐-Website-red" height="25"></a>
  <a href=""><img src="https://img.shields.io/badge/📝-Paper-blue" height="25"></a>
</p>

This repository is the official implementation of the paper: **A Comprehensive Analysis on LLM-based Node Classification Algorithms**. It provides a standardized framework for evaluating LLM-based node classification methods, including **10 datasets**, **8 LLM-based algorithms**, and **3 learning paradigms**. 


> Please consider citing or giving a 🌟 if our repository is helpful to your work!


```bibtext

```

### 🎙️ News 

📅 [2025-02-03] The [code](https://github.com/WxxShirley/LLMNodeBed) for LLMNodebed, along with the [project pages](https://llmnodebed.github.io/) and [paper](), has now been released! 🧨


---



## 📝 Table of Contents
- [🚀 Quick Start](#-quick-start)
- [📖 Code Structure](#-code-structure)
- [🔧 Supported Methods](#-supported-methods)
- [📮 Contact](#-contact)
- [🙏 Acknowledgements](#-acknowledgements)



## 🚀 Quick Start 

### 0. Environment Setup
To get started, follow these steps to set up your Python environment: 

```bash
conda create -n NodeBed python=3.10
conda activate NodeBed
pip install torch torch_geometric transformers peft pytz scikit-learn torch_scatter torch_sparse
```

Some packages might be missed for specific algorithms. Check the algorithm READMD or error logs to identify any missing dependencies and install them accordingly.


### 1. LLM Preparation 

* **Close-source LLMs like GPT-4o, DeepSeek-Chat**: 

  Add API keys to `LLMZeroShot/Direct/api_keys.py`

* **Open-source LLMs like Mistral-7B, Qwen**:

  Download models from HuggingFace (e.g., [Mistral-7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2)). Then, update model paths in `common/model_path.py` as you actual saving paths.

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

* **LLM-as-Encoder**: [`LLMEncoder/README.md`](https://github.com/WxxShirley/LLMNodeBed/tree/main/LLMEncoder)

* **LLM-as-Predictor**: [`LLMPredictor/README.md`](https://github.com/WxxShirley/LLMNodeBed/tree/main/LLMPredictor)

* **LLM-as-Reasoner**: [`LLMReasoner/README.md`](https://github.com/WxxShirley/LLMNodeBed/tree/main/LLMReasoner)

* **Zero-shot Methods**: [`LLMZeroShot/README.md`](https://github.com/WxxShirley/LLMNodeBed/tree/main/LLMZeroshot)



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


## 📮 Contact 

If you have any further questions about usage, reproducibility, or would like to discuss, please feel free to open an issue or contact the authors via email at xxwu@se.cuhk.edu.hk.



## 🙏 Acknowledgements

We thank the authors of TAPE, ENGINE, GraphGPT, LLaGA, and ZeroG for their open-source implementations. Part of our framework is inspired by [GLBench](https://github.com/NineAbyss/GLBench).

