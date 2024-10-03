# LLM as Enhancers 

## LLM for Embedding Generation 

Relevant files are `embedding.py` and `main.py`:

* **`embedding.py`** This file utilizes various LMs and LLMs to generate node embeddings, saving the results in the `/datasets/` folder.

* **`main.py`** This file runs GNNs on a specified dataset for node classification tasks, allowing you to evaluate performance. You can specify the node encoder, which can be either an LM or an LLM.

### Example Scripts 

Please refer to `run.sh` for example commands:

```bash
# Generate Embedding 
#  - Specify dataset's name (`dataset`), encoder's name (`encoder_name`), and pooling method (`use_cls`)
#  - For LLMs, please remember to set `use_cls=0`
python3 embedding.py --encoder_name=e5-large --dataset=cora 
python3 embedding.py --encoder_name=Qwen-3B --dataset=cora  --use_cls=0


# Run GNNs 
# Note that the default encoder is Shallow Embedding, e.g.,
python3 main.py --dataset=cora 
# To use the generated node embeddings from a specified encoder, run:
# Ensure that you have executed `embedding.py` before this command.
python3 main.py --dataset=cora --encoder_name=e5-large 
# Additional GNN configurations
python3 main.py --dataset=cora --encoder_name=e5-large --gnn_type=GCN --n_layers=2 --hidden_dim=128 --dropout=0.5
```


### (TODO) Performance and Observations 



