for DATASET in  cora; do
    for MODELNAME in gpt-4o; do
        for PREDICTION_TYPE in lm gnn llm; do
            python zeroshot_llm_main.py --prediction_type=$PREDICTION_TYPE --model_name=$MODELNAME --dataset=$DATASET
        done
    done
done



# source /data/fge2/try/.venv/bin/activate.csh
# nohup python3 -u zeroshot_llm_main.py >>../results/LLMPredictor/llm_zero_shot/gpt-4/cora.log
# nohup ./run.sh >/data/fge2/try/LLMGraphBench/results/zero_shot/none/cora.log

