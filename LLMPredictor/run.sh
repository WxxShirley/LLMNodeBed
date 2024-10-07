for DATASET in cora citeseer instagram ; do
   for MODELNAME in deepseek-chat; do
        python zeroshot_llm_main.py --model_name=$MODELNAME --dataset=$DATASET
    done
done
