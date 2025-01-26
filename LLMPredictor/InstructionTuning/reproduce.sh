# [Example Script] Run on 2 A100 40G GPU device 
WANDB_DISABLED=True python main.py --num_epoch=20 --llm=Mistral-7B  --batch_size=16  --re_split=0 --dataset=cora --num_gpus=2
WANDB_DISABLED=True python main.py --num_epoch=4 --llm=Mistral-7B  --batch_size=16  --re_split=1  --dataset=cora --num_gpus=2



# [Reprocude Scripts] Run on 1 H100 80G GPU device

mkdir ../../results/InstructionTuning/logs

for DATASET in cora citeseer ; do 
    for SEED in 0 1 2 3  ; do 
        WANDB_DISABLED=True python3 -u main.py --num_epoch=20 --llm=Mistral-7B  --batch_size=12  --re_split=0 --dataset=$DATASET --max_txt_length=80 --max_origin_txt_length=200 --seed=$SEED  >>../../results/InstructionTuning/logs/Mistral-7B+$DATASET+Semi.log 
        WANDB_DISABLED=True python3 -u main.py --num_epoch=4 --llm=Mistral-7B  --batch_size=12  --re_split=1 --dataset=$DATASET --max_txt_length=80 --max_origin_txt_length=200  --seed=$SEED  >>../../results/InstructionTuning/logs/Mistral-7B+$DATASET.log 
    done 
done 
   

for DATASET in wikics pubmed; do 
    for SEED in 0 1 2 3  ; do 
        # Pubmed: Avg Query Prompt Length 51.0000 | Avg OriginTxT Length 425.6667 |  Avg Output Length 4.3333
        # WikiCS: Avg Query Prompt Length 76.0000 | Avg OriginTxT Length 597.9836 |  Avg Output Length 3.1882
        WANDB_DISABLED=True python3 -u main.py --num_epoch=8 --llm=Mistral-7B  --batch_size=10  --re_split=0 --dataset=$DATASET --max_txt_length=80 --max_origin_txt_length=200 --seed=$SEED  >>../../results/InstructionTuning/logs/Mistral-7B+$DATASET+Semi.log 
        WANDB_DISABLED=True python3 -u main.py --num_epoch=2 --llm=Mistral-7B  --batch_size=10  --re_split=1 --dataset=$DATASET --max_txt_length=80 --max_origin_txt_length=200  --seed=$SEED  >>../../results/InstructionTuning/logs/Mistral-7B+$DATASET.log 
    done 
done 


for DATASET in instagram; do 
    for SEED in 0 1 2 3 ; do 
        # Instagram: Avg Query Prompt Length 37.0000 | Avg OriginTxT Length 54.8707 |  Avg Output Length 2.0000
        WANDB_DISABLED=True python3 -u main.py --num_epoch=8 --llm=Mistral-7B  --batch_size=16  --re_split=0 --dataset=$DATASET --max_txt_length=80 --max_origin_txt_length=100 --seed=$SEED  >>../../results/InstructionTuning/logs/Mistral-7B+$DATASET+Semi.log 
        WANDB_DISABLED=True python3 -u main.py --num_epoch=2 --llm=Mistral-7B  --batch_size=16  --re_split=1 --dataset=$DATASET --max_txt_length=80 --max_origin_txt_length=100  --seed=$SEED  >>../../results/InstructionTuning/logs/Mistral-7B+$DATASET.log 
    done 
done 


for DATASET in reddit photo computer history; do 
    for SEED in 0 1 2 3 ; do 
        WANDB_DISABLED=True python3 -u main.py --num_epoch=2 --llm=Mistral-7B  --batch_size=10  --re_split=0 --dataset=$DATASET --max_txt_length=80 --max_origin_txt_length=200 --seed=$SEED  >>../../results/InstructionTuning/logs/Mistral-7B+$DATASET+Semi.log 
        WANDB_DISABLED=True python3 -u main.py --num_epoch=1 --llm=Mistral-7B  --batch_size=10  --re_split=1 --dataset=$DATASET --max_txt_length=80 --max_origin_txt_length=200 --seed=$SEED  >>../../results/InstructionTuning/logs/Mistral-7B+$DATASET.log 
    done 
done 


WANDB_DISABLED=True python3 -u main.py --num_epoch=1 --llm=Mistral-7B --re_split=0 --dataset=arxiv --max_txt_length=256 --max_origin_txt_length=200 --batch_size=6 >>../../results/InstructionTuning/Mistral-7B+arXiv.log 
