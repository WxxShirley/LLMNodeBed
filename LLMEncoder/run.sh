#!/bin/bash

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


# Commands of GridSearch (take GCN and cora dataset as an example)
HIDDEN_DIMS=(64 128 256 512)
LAYERS=(2 3)
DROPOUTS=(0.3 0.5 0.7)
for HIDDEN_DIM in "${HIDDEN_DIMS[@]}"; do 
    for LAYER in "${LAYERS[@]}"; do 
        for DROPOUT in "${DROPOUTS[@]}"; do  
            python main.py --dataset=cora --encoder_name=e5-large --hidden_dim=$HIDDEN_DIM --n_layers=$LAYER --dropout=$DROPOUT 
            # architecture (batch_norm, residual_conn, etc) search
            # reference from "Classic GNNs are Strong Baselines: Reassessing GNNs for Node Classification"
            python main.py --dataset=cora --encoder_name=e5-large --hidden_dim=$HIDDEN_DIM --n_layers=$LAYER --dropout=$DROPOUT --batch_norm=1 
            python main.py --dataset=cora --encoder_name=e5-large --hidden_dim=$HIDDEN_DIM --n_layers=$LAYER --dropout=$DROPOUT --batch_norm=0 --residual_conn=1 
        done
    done 
done 
