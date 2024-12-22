for DATASET in citeseer ; do 
    WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False CUDA_VISIBLE_DEVICES=0  python -m trainLM dataset $DATASET runs 4 >>../../results/TAPE/logs/$DATASET+LM.log 
    python -m trainGNN dataset $DATASET runs 4 >>../../results/TAPE/logs/$DATASET+GNN.log 
done 
