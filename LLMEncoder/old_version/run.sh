#!/bin/bash

# Generate Embedding 
#  - Specify dataset's name (`dataset`), encoder's name (`encoder_name`), and pooling method (`use_cls`)
#  - For LLMs, please remember to set `use_cls=0`
python3 embedding.py --encoder_name=Qwen-3B --dataset=cora  --use_cls=0


# Run GNNs 
# Note that the default encoder is Shallow Embedding, e.g.,
python3 main.py --dataset=cora 
# You can specify the encoder's name to use this encoder generated node embeddings.
# Note that you have to run the `embedding.py` file before running this command.
python3 main.py --dataset=cora --encoder_name=SentenceBert
# Other configurations for GNNs
python3 main.py --dataset=cora --encoder_name=Qwen-3B --gnn_type=GCN --n_layers=2 --hidden_dim=64 --dropout=0.5


# Commands of GridSearch (take GCN and cora dataset as an example)
HIDDEN_DIMS=(32 64 128 256)
LAYERS=(2 3 4)
DROPOUTS=(0.3 0.5 0.7)
for HIDDEN_DIM in "${HIDDEN_DIMS[@]}"; do 
    for LAYER in "${LAYERS[@]}"; do 
        for DROPOUT in "${DROPOUTS[@]}"; do  
            python main.py --dataset=cora --encoder_name=Qwen-3B --hidden_dim=$HIDDEN_DIM --n_layers=$LAYER --dropout=$DROPOUT 
        done
    done 
done 
