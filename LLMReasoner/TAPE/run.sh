mkdir ../../results/TAPE/logs_supervised/

# Before running the following commands, please run the `explanation_llm.py` to prepare generated texts. 
# The following commands are exected on a single NVIDIA H100-80G.

# Small-scale
for DATASET in cora citeseer wikics instagram ; do 
    WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False  python -m trainLM dataset $DATASET runs 4 re_split 1 lm.train.epochs 10 >> ../../results/TAPE/logs_supervised/$DATASET+LM.log 
    # Vanilla version without LLM
    python -m trainGNN dataset $DATASET runs 4 re_split 1  >> ../../results/TAPE/logs_supervised/$DATASET+GNN.log 

    for LLM in Mistral-7B ; do 
         WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False  python -m trainLM dataset $DATASET runs 4  lm.train.use_gpt True   lm.train.llm_name $LLM   re_split 1  >> ../../results/TAPE/logs_supervised/$DATASET+$LLM+LM.log 
         # Full TAPE 
         python -m trainEnsemble dataset $DATASET runs 4 gnn.train.feature_type TA_E  lm.train.llm_name $LLM re_split 1   >> ../../results/TAPE/logs_supervised/$DATASET+Ensemble+$LLM.log 
    done 
done 


# Middle
for DATASET in pubmed reddit ; do 
    WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False  python -m trainLM dataset $DATASET runs 4 re_split 1  lm.train.epochs 6 >> ../../results/TAPE/logs_supervised/$DATASET+LM.log 
    python -m trainGNN dataset $DATASET runs 4 re_split 1  >> ../../results/TAPE/logs_supervised/$DATASET+GNN.log 

    for LLM in Mistral-7B ; do 
         WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False  python -m trainLM dataset $DATASET runs 4  lm.train.use_gpt True   lm.train.llm_name $LLM re_split 1  >> ../../results/TAPE/logs_supervised/$DATASET+$LLM+LM.log 
         python -m trainEnsemble dataset $DATASET runs 4 gnn.train.feature_type TA_E  lm.train.llm_name $LLM re_split 1   >> ../../results/TAPE/logs_supervised/$DATASET+Ensemble+$LLM.log 
    done 
done 


# Large
for DATASET in photo history computer ; do 
    WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False  python -m trainLM dataset $DATASET runs 4  re_split 1  lm.train.epochs 2 >> ../../results/TAPE/logs_supervised/$DATASET+LM.log 
    python -m trainGNN dataset $DATASET runs 4 re_split 1  >> ../../results/TAPE/logs_supervised/$DATASET+GNN.log 

    for LLM in Mistral-7B  gpt-4o-mini ; do 
         WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False  python -m trainLM dataset $DATASET runs 4  lm.train.use_gpt True   lm.train.llm_name $LLM re_split 1  lm.train.epochs 2 >> ../../results/TAPE/logs_supervised/$DATASET+$LLM+LM.log 
         python -m trainEnsemble dataset $DATASET runs 4 gnn.train.feature_type TA_E  lm.train.llm_name $LLM  re_split 1  >> ../../results/TAPE/logs_supervised/$DATASET+Ensemble+$LLM.log 
    done 
done 


# arxiv - single
for DATASET in arxiv ; do 
    WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False  python -m trainLM dataset $DATASET runs 4 lm.train.epochs 2   >> ../../results/TAPE/logs/$DATASET+LM.log 
    python -m trainGNN dataset $DATASET runs 4  >> ../../results/TAPE/logs/$DATASET+GNN.log 

     for LLM in Mistral-7B  gpt-4o-mini ; do 
         WANDB_DISABLED=True TOKENIZERS_PARALLELISM=False  python -m trainLM dataset $DATASET runs 4  lm.train.use_gpt True   lm.train.llm_name $LLM  lm.train.epochs 1  >> ../../results/TAPE/logs/$DATASET+$LLM+LM.log 
         python -m trainEnsemble dataset $DATASET runs 4 gnn.train.feature_type TA_E  lm.train.llm_name $LLM  >> ../../results/TAPE/logs/$DATASET+Ensemble+$LLM.log 
    done 
done 
