for DEVICE in cuda:1; do
    for DATASET in  pubmed; do
        for MODELNAME in mistral-7b; do
            for PREDICTION_TYPE in none cot tot react lm gnn llm; do
                python zeroshot_llm_main.py --prediction_type=$PREDICTION_TYPE --model_name=$MODELNAME --dataset=$DATASET --device=$DEVICE
            done
        done
    done
done



# source /data/fge2/try/.venv/bin/activate.csh
# nohup python3 -u zeroshot_llm_main.py >>../results/LLMPredictor/llm_zero_shot/gpt-4/cora.log
# nohup ./run.sh >/data/fge2/LLMGraphBench/results/zero_shot/none/ig.log & nohup ./run_copy.sh >/data/fge2/LLMGraphBench/results/zero_shot/none/pubmed.log & nohup ./run_2.sh >/data/fge2/LLMGraphBench/results/zero_shot/none/check.log &

