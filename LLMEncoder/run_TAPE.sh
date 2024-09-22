cd TAPE 

# Shallow version (LM+GNN Decoupled)
#   - First fine-tune a LM (roberta-355M)
#   - Then train a GNN (default: 2 layers GCN)
for DATASET in cora citeseer pubmed wikics instagram reddit ; do 
    WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False  python -m trainLM dataset $DATASET runs 3  >> ../../results/LLMEncoder/TAPE/$DATASET+LM.log 

    python -m trainGNN dataset $DATASET runs 5 >> ../../results/LLMEncoder/TAPE/$DATASET+GNN.log 
done 

for DATASET in arxiv ; do 
    WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False  python -m trainLM dataset $DATASET runs 1   >> ../../results/LLMEncoder/TAPE/$DATASET+LM.log 
    python -m trainGNN dataset $DATASET runs 5 >> ../../results/LLMEncoder/TAPE/$DATASET+GNN.log 
done 



# Preparation - Generation explanations via LLMs
for DATASET in cora citeseer pubmed wikics instagram reddit  ; do 
    # python3 -u explanation_api.py --dataset=$DATASET
    python3 -u explanation_qwen.py --dataset=$DATASET
done 

# Full TAPE  (LM+GNN Decoupled)
#   - First fine-tune a LM (roberta-355M) based on LLM's generated explanations
#   - Then train a GNN (default: 2 layers GCN)
for LLM in Qwen2.5-3B   DeepSeek-Chat ; do 
    for DATASET in citeseer ; do  
         WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False  python -m trainLM dataset $DATASET runs 3  lm.train.use_gpt True   lm.train.llm_name $LLM  >> ../../results/LLMEncoder/TAPE/$DATASET+LM+$LLM.log 
    
         python -m trainEnsemble dataset $DATASET runs 5 gnn.train.feature_type TA_E lm.train.llm_name $LLM   >> ../../results/LLMEncoder/TAPE/$DATASET+Ensemble+$LLM.log 
    done 
done 
