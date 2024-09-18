cd TAPE 

# Shallow version (LM+GNN Decoupled)
#   - First fine-tune a LM (roberta-355M)
#   - Then train a GNN (default: 2 layers GCN)
for DATASET in cora citeseer instagram reddit ; do 
    WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False  python -m trainLM dataset $DATASET runs 2  >> ../../results/LLMEncoder/TAPE/$DATASET+LM.log 

    python -m trainGNN dataset $DATASET runs 5 >> ../../results/LLMEncoder/TAPE/$DATASET+GNN.log 
done 
 
for DATASET in pubmed arxiv ; do 
    WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False  python -m trainLM dataset $DATASET runs 1  lm.train.max_length 1024  lm.train.batch_size 16 >> ../../results/LLMEncoder/TAPE/$DATASET+LM.log 

    python -m trainGNN dataset $DATASET runs 5 >> ../../results/LLMEncoder/TAPE/$DATASET+GNN.log 
done 


# Generate explanations 
for DATASET in cora pubmed citeseer wikics  instagram reddit ; do 
    python3 -u explanation_llm.py --dataset=$DATASET
done 


# Full version 
for DATASET in cora ; do 
    WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False  python -m trainLM dataset $DATASET runs 2  >> ../../results/LLMEncoder/TAPE/$DATASET+LM.log 
    WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False  python -m trainLM dataset $DATASET runs 2 lm.train.use_gpt True  lm.train.epochs 2 >> ../../results/LLMEncoder/TAPE/$DATASET+LM+2.log 
    
    python -m trainEnsemble dataset $DATASET runs 5 gnn.train.feature_type TA_E >> ../../results/LLMEncoder/TAPE/$DATASET+Ensemble.log 
done 
