for DATASET in cora citeseer instagram ; do
   for MODELNAME in deepseek-chat; do
        python zeroshot_llm_main.py --model_name=$MODELNAME --dataset=$DATASET
    done
done
# source /data/fge2/try/.venv/bin/activate.csh
# nohup python3 -u zeroshot_llm_main.py >>../results/LLMPredictor/llm_zero_shot/gpt-4/cora.log
# nohup ./run.sh >../results/LLMPredictor/llm_gnn_neighbors/run.log