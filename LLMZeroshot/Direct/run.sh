LOG_DIR=../../results/DirectInference/logs
mkdir $LOG_DIR

python3 -u main.py --dataset=cora --model_name=deepseek-chat --prompt_type=direct >>$LOG_DIR/cora_direct.log 


for DATASET in cora wikics; 
    for LLM in deepseek-chat gpt-4o ; do 
        for PROMPT in direct cot summary ; do 
            python3 -u main.py --dataset=$DATASET --model_name=$LLM --prompt_type=$PROMPT >>$LOG_DIR/$DATASET+$LLM+$PROMPT.log 
        done 
    done 
done 
