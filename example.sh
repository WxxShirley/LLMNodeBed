# Generate node embeddings using the e5-large model on a CUDA environment 
# (approximately 30 seconds on a single GPU with 48GB)
python embedding.py --encoder_type=LM --lm_name=e5-large 

# Generate node embeddings using the e5-large model on a CPU 
# (approximately 4 minutes on a 16GB MacBook Air)
python embedding.py --device=cpu --encoder_type=LM --lm_name=e5-large

# Generate node embeddings using Llama3-8B on a CUDA environment 
# (for LLMs, set use_cls to 0; approximately 2 minutes on a single GPU with 48GB)
python embedding.py --encoder_type=LLM --llm_name=Llama3-8B --use_cls=0 

# Run GNNs with shallow embeddings on CPU
python main.py --device=cpu 

# Run GNNs with e5-large generated embeddings on CPU
python main.py --device=cpu --emb_type=LM --emb_model=e5-large --gnn=GCN 
