GPU=0

mkdir ../../results/LLaGA/logs

for LLM in Mistral-7B ; do 
    # small-scale
    for DATASET in cora citeseer ; do 
         for SEED in 0 1 2 3 4 5 6 7 8 9 ; do 
            python3 -u main.py --neighbor_template=HO --dataset=$DATASET --seed=$SEED --re_split=0  --num_epochs=12 --llm=$LLM --patience=4 --gpu_id=$GPU  >>../../results/LLaGA/logs/$LLM+$DATASET+Semi.log 
            python3 -u main.py --neighbor_template=HO --dataset=$DATASET --seed=$SEED --re_split=1  --num_epochs=10 --llm=$LLM --patience=4 --gpu_id=$GPU  >>../../results/LLaGA/logs/$LLM+$DATASET.log 
         done 
    done 
    
    # middle-scale ~ 10,000
    for DATASET in wikics instagram pubmed; do 
        for SEED in 0 1 2 3  ; do 
             python3 -u main.py --neighbor_template=HO --dataset=$DATASET --re_split=0  --num_epochs=10 --llm=$LLM  --patience=2 --seed=$SEED --gpu_id=$GPU   >>../../results/LLaGA/logs/$LLM+$DATASET+Semi.log 
             python3 -u main.py --neighbor_template=HO --dataset=$DATASET --re_split=1  --num_epochs=8 --llm=$LLM  --patience=2 --seed=$SEED --gpu_id=$GPU   >>../../results/LLaGA/logs/$LLM+$DATASET.log 
        done 
    done 
    
    # middle-scale ~ 40,000 
    for DATASET in reddit photo computer history; do 
        for SEED in 0 1 2 3 ; do 
            python3 -u main.py --neighbor_template=HO --dataset=$DATASET --re_split=0  --num_epochs=8 --llm=$LLM  --patience=2 --seed=$SEED  --gpu_id=$GPU --num_gpus=2  >>../../results/LLaGA/logs/$LLM+$DATASET+Semi.log 
            python3 -u main.py --neighbor_template=HO --dataset=$DATASET --re_split=1  --num_epochs=6 --llm=$LLM  --patience=2 --seed=$SEED  --gpu_id=$GPU --num_gpus=2  >>../../results/LLaGA/logs/$LLM+$DATASET.log 
        done 
    done 
    
    # arXiv 
    # TODO: adjust the batch size for different LLMs 
    #       e.g., Mistral-7B batch_size=16 is ok
    python3 -u main.py --neighbor_template=HO --seed=0 --gpu_id=$GPU --dataset=arxiv --re_split=0  --num_epochs=2 --llm=$LLM --patience=1 --batch_size=12 --max_txt_length=460 --max_ans_length=20 --eval_batch_size=32 --num_gpus=1  >>../../results/LLaGA/logs/$LLM+arxiv.log 

done 


# For Qwen-series, e.g., Qwen-3B 
for LLM in Qwen-3B ; do 
    for DATASET in cora citeseer; do 
        for SEED in 0 1 2 3 5 6 7 8 9 ; do 
            python3 -u main.py --neighbor_template=HO --dataset=$DATASET --seed=$SEED --re_split=0  --num_epochs=12 --llm=$LLM --patience=4 --batch_size=32 --eval_batch_size=64  --gpu_id=$GPU --lr=5e-4 >>../../results/LLaGA/logs/$LLM+$DATASET+Semi.log 
            python3 -u main.py --neighbor_template=HO --dataset=$DATASET --seed=$SEED --re_split=1  --num_epochs=10 --llm=$LLM --patience=4 --batch_size=32 --eval_batch_size=64  --gpu_id=$GPU >>../../results/LLaGA/logs/$LLM+$DATASET.log 
        done 
    done 

    for SEED in 0 4 5 6 ; do 
        for DATASET in instagram wikics pubmed ; do 
            python3 -u main.py --neighbor_template=HO --dataset=$DATASET --re_split=0  --num_epochs=10 --llm=$LLM  --patience=4 --batch_size=32  --seed=$SEED --eval_batch_size=64 --gpu_id=$GPU >>../../results/LLaGA/logs/$LLM+$DATASET+Semi.log 
            python3 -u main.py --neighbor_template=HO --dataset=$DATASET --re_split=1  --num_epochs=8 --llm=$LLM  --patience=4 --seed=$SEED --batch_size=32 --eval_batch_size=64 --gpu_id=$GPU >>../../results/LLaGA/logs/$LLM+$DATASET.log 
        done 
    done 

    for SEED in 0 4 5 ; do 
        for DATASET in history photo; do 
            python3 -u main.py --neighbor_template=HO --dataset=$DATASET --re_split=1  --num_epochs=4 --llm=$LLM  --patience=2 --batch_size=32  --seed=$SEED --eval_batch_size=64 --gpu_id=$GPU >>../../results/LLaGA/logs/$LLM+$DATASET.log 
            python3 -u main.py --neighbor_template=HO --dataset=$DATASET --re_split=1  --num_epochs=2 --llm=$LLM  --patience=2 --batch_size=32  --seed=$SEED --eval_batch_size=64 --gpu_id=$GPU >>../../results/LLaGA/logs/$LLM+$DATASET.log 
        done 
    done 

    python3 -u main.py --seed=0  --neighbor_template=HO --dataset=arxiv --re_split=0  --num_epochs=1 --llm=$LLM --patience=2 --batch_size=8 --max_txt_length=400 --eval_batch_size=24 --gpu_id=$GPU >>../../results/LLaGA/logs/$LLM+arxiv.log 

done 
