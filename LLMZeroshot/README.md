# Zero-shot Learning Methods 

This folder contains two zero-shot learning methods: `Direct` for LLM Direct Inference and `ZeroG` for the [ZeroG](https://arxiv.org/abs/2402.11235) algorithm. 

## 🚀 Usage 

### LLM Direct Inference

To use the Direct Inference, you need to first **fill in the LLM API Key** in the `Direct/api_keys.py` file.

**Supported LLMs:**
* DeepSeek-Chat
* GPT-4o
* Open-source LLMs: LLaMA3.1-8B, Mistral-7B  
  *Note: Open-source LLMs must be deployed locally using FastChat + vLLM.*

  ```shell
  # Example scripts for deploying local LLM services
  #  - Install dependencies
  pip install vllm "fschat[model_worker,webui]"

  #  - Start services
  python3 -m fastchat.serve.controller --host 127.0.0.1
  CUDA_VISIBLE_DEVICES=0 python3 -m fastchat.serve.vllm_worker --model-path mistralai/Mistral-7B-Instruct-v0.2 --host 127.0.0.1
  python3 -m fastchat.serve.openai_api_server --host 127.0.0.1 --port 8008
  ```

**Supported Prompt Templates:**
* Direct
* Chain-of-Thought (CoT)
* Tree-of-Thought (ToT)
* ReAct
* w. Neighbor
* w. Summary


**Example Script:**

`dataset` specifies the dataset, `model_name` specifies the LLM name, `prompt_type` specifies the prompt template.
For more information, please refer to [`Direct/run.sh`](https://github.com/WxxShirley/LLMGraphBench/blob/main/LLMZeroshot/Direct/run.sh) file.
```shell
# LLM (DeepSeek-Chat) direct inference with "Direct" prompt on the Cora dataset
cd Direct
python3 -u main.py --dataset=cora --model_name=deepseek-chat --prompt_type=direct
```

### ZeroG 

For ZeroG, you need to download the LM model and modify the `common/model_path.py` file to point to the local model path. The default model path is set to [`sentence-transformers/multi-qa-distilbert-cos-v1`](https://huggingface.co/sentence-transformers/multi-qa-distilbert-cos-v1).


**Example Script**

`train_dataset` specifies the training dataset, `test_dataset` specifies the testing dataset.
```shell
# Script for ZeroG Reproduction
cd ZeroG
python3 -u main.py --train_dataset arxiv --test_dataset cora wikics --k 2 --if_norm --device 0 --epoch 2 
python3 -u main.py --train_dataset reddit --test_dataset instagram --k 2 --if_norm --device 0 --epoch 1 
python3 -u main.py --train_dataset computer --test_dataset photo --k 2 --if_norm --device 0 --epoch 1
```

## 🙏 Acknowledgement 

Special thanks to the [official implementation](https://github.com/NineAbyss/ZeroG) of the ZeroG algorithm!
