#!/bin/bash

ENCODERS=("shallow" "SentenceBert" "roberta" "Qwen-3B" "Mistral-7B")
HIDDEN_DIMS=(256 512)

SUPERVISED=0
DATASETS=("cora" "citeseer" "pubmed" "wikics" "instagram" "reddit" "photo" "history" "computer" "arxiv")
for DATASET in "${DATASETS[@]}"; do 
    for ENCODER in "${ENCODERS[@]}"; do  
        for HIDDEN_DIM in "${HIDDEN_DIMS[@]}"; do 
            python3 -u main.py --re_split=$SUPERVISED --dataset=$DATASET --encoder_name=$ENCODER --hidden_dim=$HIDDEN_DIM --n_layers=3 --dropout=0.5  --print_freq=50 
            python3 -u main.py --re_split=$SUPERVISED --dataset=$DATASET --encoder_name=$ENCODER --hidden_dim=$HIDDEN_DIM --n_layers=4 --dropout=0.6  --print_freq=50 
            python3 -u main.py --re_split=$SUPERVISED --dataset=$DATASET --encoder_name=$ENCODER --hidden_dim=$HIDDEN_DIM --n_layers=4 --dropout=0.7  --print_freq=50 
        done 
    done
done 


SUPERVISED=1
DATASETS=("cora" "citeseer" "pubmed" "wikics" "instagram" "reddit" "photo" "history" "computer")
HIDDEN_DIMS=(128 256 512)
for DATASET in "${DATASETS[@]}"; do 
    for ENCODER in "${ENCODERS[@]}"; do  
        for HIDDEN_DIM in "${HIDDEN_DIMS[@]}"; do 
            python3 -u main.py --re_split=$SUPERVISED --dataset=$DATASET --encoder_name=$ENCODER --hidden_dim=$HIDDEN_DIM --n_layers=3 --dropout=0.5  --print_freq=50 
            python3 -u main.py --re_split=$SUPERVISED --dataset=$DATASET --encoder_name=$ENCODER --hidden_dim=$HIDDEN_DIM --n_layers=4 --dropout=0.6  --print_freq=50 
        done 
    done
done 
