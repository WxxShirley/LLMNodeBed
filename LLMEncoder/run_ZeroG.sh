cd ZeroG

for DATASET in cora citeseer pubmed wikics instagram reddit; do 
    python3 -u main.py --text_encoder=SentenceBert --dataset=$DATASET >>logs/SentenceBert/$DATASET.log 
    python3 -u main.py --text_encoder=SentenceBert --dataset=$DATASET --use_lora=0  >>logs/SentenceBert/$DATASET.log 
    
    for ENCODER in e5-large roberta; do 
         python3 -u main.py --text_encoder=$ENCODER --dataset=$DATASET --lr=1e-3 >>logs/$ENCODER/$DATASET.log 
         python3 -u main.py --text_encoder=$ENCODER --dataset=$DATASET --lr=2e-5 --use_lora=0 >>logs/$ENCODER/$DATASET.log 
    done 
done 
