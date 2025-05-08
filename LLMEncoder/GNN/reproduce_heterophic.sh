#!/bin/bash

ENCODERS=("shallow" "SentenceBert" "roberta" "Qwen-3B" "Mistral-7B")
# Important: grid-search over hidden_dim, n_layers, and learning rate
HIDDEN_DIMS=(32 64 128 256)
LAYERS=(1 2 3)
GNNs=("HeteroGNN")
DEVICE="cuda:0"

SUPERVISED=0 # Semi-supervised setting
DATASETS=("cornell" "texas" "wisconsin" "washington")
for GNN in "${GNNs[@]}"; do 
    for DATASET in "${DATASETS[@]}"; do 
        for ENCODER in "${ENCODERS[@]}"; do  
            for HIDDEN_DIM in "${HIDDEN_DIMS[@]}"; do 
                for LAYER in "${LAYERS[@]}"; do 
                    # Search over learning rate  
                    # The patience is increased to 200 to avoid early stopping
                    python3 -u main.py --gnn_type=$GNN --re_split=$SUPERVISED --dataset=$DATASET --encoder_name=$ENCODER --hidden_dim=$HIDDEN_DIM --n_layers=$LAYER   --print_freq=50 --device=$DEVICE  --learning_rate=1e-2  --patience=200 
                    python3 -u main.py --gnn_type=$GNN --re_split=$SUPERVISED --dataset=$DATASET --encoder_name=$ENCODER --hidden_dim=$HIDDEN_DIM --n_layers=$LAYER   --print_freq=50 --device=$DEVICE  --learning_rate=1e-3  --patience=200 
                    python3 -u main.py --gnn_type=$GNN --re_split=$SUPERVISED --dataset=$DATASET --encoder_name=$ENCODER --hidden_dim=$HIDDEN_DIM --n_layers=$LAYER   --print_freq=50 --device=$DEVICE  --learning_rate=5e-3  --patience=200 
       
                done 
            done 
        done 
    done 
done 
