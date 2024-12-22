for DATASET in cora citeseer pubmed wikics instagram reddit photo history computer arxiv; do 
    for ENCODER in Mistral-7B; do 
        python3 cache.py --dataset=$DATASET --encoder=$ENCODER
    done 
done 
