GPU=1

for LLM in Llama-8B ; do 
    for DATASET in cora citeseer ; do 
         for SEED in 0 1 2 3 4 5 6 7 8 9; do 
            python3 -u main.py --neighbor_template=HO --dataset=$DATASET --seed=$SEED --re_split=0  --num_epochs=12 --llm=$LLM --patience=4 --gpu_id=$GPU  >>../../results/LLaGA/logs/$LLM+$DATASET+Semi.log 
            python3 -u main.py --neighbor_template=HO --dataset=$DATASET --seed=$SEED --re_split=1  --num_epochs=10 --llm=$LLM --patience=4 --gpu_id=$GPU >>../../results/LLaGA/logs/$LLM+$DATASET.log 
         done 
    done 
    
    # middle-scale ~ 10,000
    for DATASET in wikics instagram pubmed; do 
        for SEED in 0 1 2 3 ; do 
             python3 -u main.py --neighbor_template=HO --dataset=$DATASET --re_split=0  --num_epochs=10 --llm=$LLM  --patience=2 --seed=$SEED --gpu_id=$GPU >>../../results/LLaGA/logs/$LLM+$DATASET+Semi.log 
             python3 -u main.py --neighbor_template=HO --dataset=$DATASET --re_split=1  --num_epochs=8 --llm=$LLM  --patience=2 --seed=$SEED --gpu_id=$GPU >>../../results/LLaGA/logs/$LLM+$DATASET.log 
        done 
    done 
    
    # middle-scale ~ 40,000 
    for DATASET in reddit photo computer history; do 
        for SEED in 0 ; do 
            python3 -u main.py --neighbor_template=HO --dataset=$DATASET --re_split=0  --num_epochs=8 --llm=$LLM  --patience=2 --seed=$SEED  --gpu_id=$GPU  >>../../results/LLaGA/logs/$LLM+$DATASET+Semi.log 
            python3 -u main.py --neighbor_template=HO --dataset=$DATASET --re_split=1  --num_epochs=6 --llm=$LLM  --patience=2 --seed=$SEED  --gpu_id=$GPU  >>../../results/LLaGA/logs/$LLM+$DATASET.log 
        done 
    done 

done 
