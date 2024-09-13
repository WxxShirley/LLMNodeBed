
#for DATASET in cora pubmed citeseer wikics instagram reddit; do
#    for LLM in Mistral-7B Vicuna-13B Llama3-8B Llama-13B; do
#        python embedding.py --encoder_type=LLM --llm_name=$LLM  --dataset=$DATASET --use_cls=0
#    done
#done


# #Example of GridSearch
#for HIDDEN in 64 128 256 512; do
#    for LAYER in 2 3; do
#       for DROPOUT in 0.3 0.5 0.7; do
#           # TODO: specify dataset, emb_type, emb_model
#           python main.py --hidden_dim=$HIDDEN --n_layers=$LAYER --dropout=$DROPOUT
#        done
#    done
#done

for DATASET in cora pubmed citeseer wikics instagram reddit; do
  for LMORLLM in LM; do
    for LMTYPE in MiniLM SentenceBert bert roberta e5-large; do
      for GNN in GCN GAT SAGE GIN TransformerConv; do
        for HIDDEN in 64 128 256 512; do
          for LAYER in 2 3; do
            for DROPOUT in 0.3 0.5 0.7; do
              # TODO: specify dataset, emb_type, emb_model
              python main.py --dataset=$DATASET --emb_type=$LMORLLM --emb_model=$LMTYPE --gnn_type=$GNN --hidden_dim=$HIDDEN --n_layers=$LAYER --dropout=$DROPOUT
            done
          done
        done
      done
    done
  done
done

for DATASET in cora pubmed citeseer wikics instagram reddit; do
  for LMORLLM in LLM; do
    for LLMTYPE in Llama-13B Mistral-7B Llama3-8B Vicuna-13B; do
      for GNN in GCN GAT SAGE GIN TransformerConv; do
        for HIDDEN in 64 128 256 512; do
          for LAYER in 2 3; do
            for DROPOUT in 0.3 0.5 0.7; do
              # TODO: specify dataset, emb_type, emb_model
              python main.py --dataset=$DATASET --emb_type=$LMORLLM --emb_model=$LLMTYPE --gnn_type=$GNN --hidden_dim=$HIDDEN --n_layers=$LAYER --dropout=$DROPOUT
            done
          done
        done
      done
    done
  done
done



#
## Run scripts
#for DATASET in cora citeseer pubmed wikics ; do
#   python main.py --dataset=$DATASET
#
#   for LM in MiniLM SentenceBert roberta e5-large; do
#        python main.py --emb_type=LM --emb_model=$LM --dataset=$DATASET
#    done
#
#    for LLM in Mistral-7B Vicuna-13B Llama3-8B Llama-13B; do
#        python main.py --emb_type=LLM --emb_model=$LLM  --dataset=$DATASET
#    done
#done
