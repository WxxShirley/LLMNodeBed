for DATASET in cora pubmed citeseer wikics arxiv instagram ; do
   for MODELNAME in qwen-turbo; do
        python zeroshot_llm.py --model_name=$MODELNAME --dataset=$DATASET
    done
done
