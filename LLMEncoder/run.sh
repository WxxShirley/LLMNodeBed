for DATASET in cora pubmed citeseer wikics instagram reddit ; do 
    for GNN_TYPE in GCN GAT SAGE TransformerConv; do 
        # non-gridsearch version 
        # python main.py --dataset=$DATASET
    
        # gridsearch for shallow embedding 
        for HIDDEN in 64 128 256 512; do 
            for LAYER in 2 3; do 
                for DROPOUT in 0.3 0.5; do 
                     python main.py --dataset=$DATASET --gnn_type=$GNN_TYPE --hidden_dim=$HIDDEN --n_layers=$LAYER --dropout=$DROPOUT 
                done 
            done 
        done 
        
        for LM in MiniLM SentenceBert roberta e5-large; do 
            for HIDDEN in 64 128 256 512; do 
                for LAYER in 2 3; do 
                    for DROPOUT in 0.3 0.5; do 
                          python main.py --dataset=$DATASET --gnn_type=$GNN_TYPE --hidden_dim=$HIDDEN --n_layers=$LAYER --dropout=$DROPOUT  --emb_type=LM --emb_model=$LM
                    done 
                done 
            done
        done

        for LLM in Mistral-7B Llama3-8B Llama-13B Vicuna-13B; do 
            for HIDDEN in 64 128 256 512; do 
                for LAYER in 2 3; do 
                    for DROPOUT in 0.3 0.5; do 
                          python main.py --dataset=$DATASET --gnn_type=$GNN_TYPE --hidden_dim=$HIDDEN --n_layers=$LAYER --dropout=$DROPOUT  --emb_type=LLM --emb_model=$LLM 
                    done 
                done 
            done
        done

    done 

    # non-gridsearch version 
    # for LM in MiniLM SentenceBert roberta e5-large; do 
    #     python main.py --emb_type=LM --emb_model=$LM --dataset=$DATASET
    # done 

    # for LLM in Mistral-7B Llama3-8B Llama-13B Vicuna-13B ; do 
    #     python main.py --emb_type=LLM --emb_model=$LLM  --dataset=$DATASET
    # done 
done 


for DATASET in arxiv; do 
   for GNN_TYPE in GCN GAT SAGE TransformerConv; do 
        python3 -u main.py --dataset=$DATASET --gnn_type=$GNN_TYPE  --write_result=1

        for LM in MiniLM SentenceBert roberta e5-large; do 
            python3 -u main.py --emb_type=LM --emb_model=$LM --dataset=$DATASET  --gnn_type=$GNN_TYPE --write_result=1 
        done 

        for LLM in Mistral-7B Llama3-8B Llama-13B Vicuna-13B ; do 
            python3 -u main.py --emb_type=LLM --emb_model=$LLM  --dataset=$DATASET --gnn_type=$GNN_TYPE --write_result=1
        done 
    done
done
