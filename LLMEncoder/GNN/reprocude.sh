#!/bin/bash

ENCODERS=("shallow" "Mistral-7B")
HIDDEN_DIMS=(64 128 256)
LAYERS=(2 3)
DROPOUTS=(0.5 0.7)
GNNs=("GCN")
DEVICE="cuda:0"

# Semi-supervised Settings 
#   - For Cora, Citeseer, and Pubmed, the `batch_norm` is set 0 
#   - For remaining datasets, `batch_norm` is set to 1
SUPERVISED=0
DATASETS=("cora" "citeseer" "pubmed")
for GNN in "${GNNs[@]}"; do 
    for DATASET in "${DATASETS[@]}"; do 
        for ENCODER in "${ENCODERS[@]}"; do  
            for HIDDEN_DIM in "${HIDDEN_DIMS[@]}"; do 
                for LAYER in "${LAYERS[@]}"; do 
                    for DROPOUT in "${DROPOUTS[@]}"; do  
                        python3 -u main.py --gnn_type=$GNN --re_split=$SUPERVISED --dataset=$DATASET --encoder_name=$ENCODER --hidden_dim=$HIDDEN_DIM --n_layers=$LAYER --dropout=$DROPOUT  --print_freq=50 --batch_norm=0 --device=$DEVICE
                    done
                done 
            done 
        done 
    done 
done 

DATASETS=("computer" "photo" "history" "wikics" "instagram" "reddit" "arxiv") 
for GNN in "${GNNs[@]}"; do 
    for DATASET in "${DATASETS[@]}"; do 
        for ENCODER in "${ENCODERS[@]}"; do  
            for HIDDEN_DIM in "${HIDDEN_DIMS[@]}"; do 
                for LAYER in "${LAYERS[@]}"; do 
                    for DROPOUT in "${DROPOUTS[@]}"; do  
                        python3 -u main.py --gnn_type=$GNN --re_split=$SUPERVISED --dataset=$DATASET --encoder_name=$ENCODER --hidden_dim=$HIDDEN_DIM --n_layers=$LAYER --dropout=$DROPOUT  --print_freq=50 --batch_norm=1 --device=$DEVICE
                    done
                done 
            done 
        done 
    done 
done 


SUPERVISED=1
DATASETS=("cora" "citeseer" "pubmed" "computer" "photo" "history" "wikics" "instagram" "reddit") 
for GNN in "${GNNs[@]}"; do 
    for DATASET in "${DATASETS[@]}"; do 
        for ENCODER in "${ENCODERS[@]}"; do  
            for HIDDEN_DIM in "${HIDDEN_DIMS[@]}"; do 
                for LAYER in "${LAYERS[@]}"; do 
                    for DROPOUT in "${DROPOUTS[@]}"; do  
                        python3 -u main.py --gnn_type=$GNN --re_split=$SUPERVISED --dataset=$DATASET --encoder_name=$ENCODER --hidden_dim=$HIDDEN_DIM --n_layers=$LAYER --dropout=$DROPOUT  --print_freq=50 --batch_norm=1 --learning_rate=1e-2 
                    done
                done 
            done 
        done 
    done 
done 
