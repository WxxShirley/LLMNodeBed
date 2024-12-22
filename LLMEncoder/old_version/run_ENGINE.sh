# [Step 1] Generate Embedding 
for DATASET in cora  citeseer pubmed ; do   
   for MODEL in SentenceBert  Qwen-3B ; do 
         python cache.py --dataset=$DATASET --encoder=$MODEL 
    done 
done 


# [Step 2] ENGINE Training & Evaluation
#   - Take cora as an example 
python3 -u main.py --dataset=cora --encoder=Qwen-3B --run_times=5 --hidden=64 --n_layers=2 --epoch=300 --patience=50 
python3 -u main.py --dataset=cora --encoder=SentenceBert --run_times=5 --hidden=64 --n_layers=2 --epoch=300 --patience=50 
