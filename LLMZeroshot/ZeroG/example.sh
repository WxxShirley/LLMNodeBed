# In-domain Transfer
for R in 0 1 2 10; do 
    # Transfer within Academic Networks  
    python3 -u main.py --dataset=arxiv,citeseer --test_dataset=cora,pubmed,wikics --epoch=4 --use_train=1 --R=$R >>../../results/ZeroG/logsacademic_transfer.log 
    # Transfer within Social Networks  
    python3 -u main.py --dataset=reddit --test_dataset=instagram --use_train=1 --epoch=4 --R=$R >>../../results/ZeroG/logs/social_transfer.log 
    # Transfer within ECommerce Networks 
    python3 -u main.py --dataset=computer --test_dataset=history,photo --use_train=1 --epoch=4 --R=$R >>../../results/ZeroG/logs/ecommerce_transfer.log 
done 


# Different Combinations of Academic Transfer
for USE_TRAIN in 1 ; do
   for R in 10 4 2 ; do 
        python3 -u main.py --dataset=arxiv --test_dataset=cora,citeseer,pubmed,wikics --epoch=3 --use_train=1 --R=$R --device=cuda:0 >>../../results/ZeroG/logs/fullacademic-new.log 
        python3 -u main.py --dataset=arxiv,citeseer --test_dataset=cora,pubmed,wikics --epoch=3 --use_train=1 --R=$R --device=cuda:0 >>../../results/ZeroG/logs/fullacademic-new.log 
        python3 -u main.py --dataset=citeseer,pubmed,wikics --test_dataset=cora --use_train=$USE_TRAIN --R=$R --epoch=4 --device=cuda:0  >>../../results/ZeroG/logs/fullacademic-new.log 
        python3 -u main.py --dataset=arxiv,citeseer,pubmed,wikics --test_dataset=cora --use_train=$USE_TRAIN --R=$R --epoch=2 --device=cuda:0  >>../../results/ZeroG/logs/fullacademic-new.log 
   done 
done 
