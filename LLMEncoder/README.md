# LLM-as-Encoder Methods 

This folder contains three methods: `GNN` for GNNs, `ENGINE` for the [ENGINE](https://arxiv.org/abs/2401.15569) algorithm, and `MLP` for the MLPs.

Both GNNs and MLPs can accept **arbitrary node embeddings**, including shallow embeddings or those generated from LM or LLMs. 


## Usage 

### $GNN_{\text{LLMEmb}}$

To utilize the GNN method with LLM-generated embeddings, you first need to run `embedding.py` to encode the node embeddings for your specified dataset. 

```shell
cd GNN 

# Generate Embeddings
# Specify the dataset name (`dataset`), encoder name (`encoder_name`), and pooling method (`use_cls`)
python embedding.py --encoder_name=roberta --dataset=cora --use_cls=1 
python embedding.py --encoder_name=Mistral-7B --dataset=cora 
```

After execution, an embedding file will automatically be saved in the `datasets/{encoder_name}` folder, named `{dataset}.pt`.

With the generated embeddings, you can run the GNN methods using various initialized embeddings. The GNN can be customized flexibly:

```shell
cd GNN 

# Run GNN with default shallow embeddings
python main.py --dataset=cora 

# Run GNN with Mistral-7B embeddings
python main.py --dataset=cora --encoder_name=Mistral-7B  

# More configurations for GNNs  
python3 main.py --dataset=cora --encoder_name=Mistral-7B --gnn_type=GCN --n_layers=2 --hidden_dim=64 --dropout=0.5 
python3 main.py --dataset=cora --encoder_name=Mistral-7B --gnn_type=SAGE --n_layers=2 --hidden_dim=64 --dropout=0.7 --batch_norm=0
```

For additional arguments, refer to `main.py`. 


### MLP 
The MLP method is similar to the GNN, allowing for customization with different numbers of layers, hidden dimensions, and dropout ratios. Input embeddings can be set from various encoders.

Example Script:
```shell
cd MLP 

python3 main.py --dataset=cora --encoder_name=Mistral-7B --hidden_dim=128 --n_layers=3 --dropout=0.6
```


### ENGINE 
To run the ENGINE algorithm, you must first execute `cache.py` to save the intermediate embeddings from each LLM layer, which will be used as node embeddings.

```shell
# Generate embeddings for ENGINE 
cd ENGINE 

python cache.py --dataset=cora --encoder=Mistral-7B
```


Once the embeddings are generated, you can run `main.py` to train and evaluate the ENGINE method.

Specific hyperparameter settings are provided in `reproduce.sh`. For example, for the Cora dataset, you might set the number of layers to 2 and the hidden dimension to 64:
```shell
# Cora configuration 
python3 -u main.py --gnn_type=GCN --dataset=cora --encoder=Mistral-7B --run_times=4 --hidden=64 --n_layers=2  
  
# Citeseer configuration 
python3 -u main.py --gnn_type=GCN --dataset=citeseer --hidden=128 --n_layers=1 --sampler=khop --k=2
```


## Acknowledgement

Thanks to the [official implementation](https://github.com/ZhuYun97/ENGINE) of the ENGINE algorithm!
