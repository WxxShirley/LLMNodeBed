LM=roberta
LLM=Mistral-7B
DEVICE=cuda:0

# Cora
for SPLIT in  0 ; do 
   for GNN in GCN; do 
     python3 -u main.py --device=$DEVICE  --gnn_type=$GNN --re_split=$SPLIT --dataset=cora --encoder=$LM --run_times=4 --hidden=64 --n_layers=2  >>../../results/ENGINE/logs/cora+LM+Semi.log 
     python3 -u main.py --device=$DEVICE --gnn_type=$GNN --re_split=$SPLIT --dataset=cora --encoder=$LLM --run_times=4 --hidden=64 --n_layers=2  >>../../results/ENGINE/logs/cora+LLM+Semi.log 
     python3 -u main.py --device=$DEVICE --gnn_type=$GNN --re_split=$SPLIT --dataset=cora --encoder=Mistral-7B --run_times=4 --hidden=64 --n_layers=2   >>../../results/ENGINE/logs/cora+LLM+Semi.log 
   done 
done 


# Citeseer 
for SPLIT in  0 ; do 
   for GNN in GCN; do 
       python3 -u main.py --device=$DEVICE --gnn_type=$GNN --re_split=$SPLIT --dataset=citeseer --encoder=$LM --run_times=4 --hidden=128 --n_layers=1  --sampler=khop --k=2 >>../../results/ENGINE/logs/citeseer+LM+Semi.log 
       python3 -u main.py --device=$DEVICE --gnn_type=$GNN --re_split=$SPLIT --dataset=citeseer --encoder=$LLM --run_times=4 --hidden=128 --n_layers=1  --sampler=khop --k=2 >>../../results/ENGINE/logs/citeseer+LLM+Semi.log 
       python3 -u main.py --device=$DEVICE --gnn_type=$GNN --re_split=$SPLIT --dataset=citeseer --encoder=Mistral-7B --run_times=4 --hidden=128 --n_layers=1  --sampler=khop  --k=2 >>../../results/ENGINE/logs/citeseer+LLM+Semi.log 
   done 
done 


# Pubmed 
for SPLIT in  0 ; do 
   for GNN in GCN; do 
       python3 -u main.py --device=$DEVICE --gnn_type=$GNN --re_split=$SPLIT --dataset=pubmed --encoder=$LM --run_times=4 --hidden=64 --n_layers=2 --lr=0.0005  --sampler=khop   >>../../results/ENGINE/logs/pubmed+LM+Semi.log 
       python3 -u main.py --device=$DEVICE --gnn_type=$GNN --re_split=$SPLIT --dataset=pubmed --encoder=$LLM --run_times=4 --hidden=64 --n_layers=2  --lr=0.0005  --sampler=khop   >>../../results/ENGINE/logs/pubmed+LLM+Semi.log 
       python3 -u main.py --device=$DEVICE --gnn_type=$GNN --re_split=$SPLIT --dataset=pubmed --encoder=Mistral-7B --run_times=4 --hidden=64 --n_layers=2  --lr=0.0005  --sampler=khop   >>../../results/ENGINE/logs/pubmed+LLM+Semi.log 
   done 
done 


# WikiCS 
for SPLIT in  0 ; do 
   for GNN in GCN; do 
       python3 -u main.py --device=$DEVICE --gnn_type=$GNN --re_split=$SPLIT --dataset=wikics --encoder=$LM --run_times=4 --hidden=64 --n_layers=1 --batch_norm=1 --lr=0.001  --sampler=khop   >>../../results/ENGINE/logs/wikics+LM+Semi.log 
       python3 -u main.py --device=$DEVICE --gnn_type=$GNN --re_split=$SPLIT --dataset=wikics --encoder=$LLM --run_times=4 --hidden=64 --n_layers=1 --batch_norm=1 --lr=0.001  --sampler=khop   >>../../results/ENGINE/logs/wikics+LLM+Semi.log 
       python3 -u main.py --device=$DEVICE --gnn_type=$GNN --re_split=$SPLIT --dataset=wikics --encoder=Mistral-7B --run_times=4 --hidden=64 --n_layers=1 --batch_norm=1 --lr=0.001  --sampler=khop   >>../../results/ENGINE/logs/wikics+LLM+Semi.log 
   done 
done 


# Instagram 
for SPLIT in  0 ; do 
   for GNN in GCN; do 
        python3 -u main.py --device=$DEVICE  --gnn_type=$GNN --re_split=$SPLIT --dataset=instagram --encoder=$LM --run_times=4 --hidden=128 --n_layers=3  --batch_norm=1 --sampler=khop  >>../../results/ENGINE/logs/instagram+LM+Semi.log 
        python3 -u main.py --device=$DEVICE  --gnn_type=$GNN --re_split=$SPLIT --dataset=instagram --encoder=$LLM --run_times=4 --hidden=128 --n_layers=3  --batch_norm=1 --sampler=khop  >>../../results/ENGINE/logs/instagram+LLM+Semi.log 
        python3 -u main.py --device=$DEVICE  --gnn_type=$GNN --re_split=$SPLIT --dataset=instagram --encoder=Mistral-7B --run_times=4 --hidden=128 --n_layers=3  --batch_norm=1 --sampler=khop  >>../../results/ENGINE/logs/instagram+LLM+Semi.log 
   done 
done 


# Reddit 
for SPLIT in  0 ; do 
   for GNN in GCN; do 
        python3 -u main.py --device=$DEVICE  --gnn_type=$GNN --re_split=$SPLIT --dataset=reddit --encoder=$LM --run_times=4 --hidden=64 --n_layers=3  --lr=0.0005 --batch_norm=1 --sampler=khop  >>../../results/ENGINE/logs/reddit+LM+Semi.log 
        python3 -u main.py --device=$DEVICE  --gnn_type=$GNN --re_split=$SPLIT --dataset=reddit --encoder=$LLM --run_times=4 --hidden=64 --n_layers=3  --lr=0.0005 --batch_norm=1 --sampler=khop  >>../../results/ENGINE/logs/reddit+LLM+Semi.log 
        python3 -u main.py --device=$DEVICE  --gnn_type=$GNN --re_split=$SPLIT --dataset=reddit --encoder=Mistral-7B --run_times=4 --hidden=64 --n_layers=3 --lr=0.0005 --batch_norm=1 --sampler=khop  >>../../results/ENGINE/logs/reddit+LLM+Semi.log 
     done 
done 


# Photo
for SPLIT in  0 ; do 
   for GNN in GCN; do 
        python3 -u main.py --gnn_type=$GNN --re_split=$SPLIT --dataset=photo --encoder=$LM --run_times=4 --hidden=64 --n_layers=2 --lr=0.0005 --batch_norm=1 --sampler=khop  >>../../results/ENGINE/logs/photo+LM+Semi.log 
        python3 -u main.py --gnn_type=$GNN --re_split=$SPLIT --dataset=photo --encoder=$LLM --run_times=4 --hidden=64 --n_layers=2  --lr=0.0005 --batch_norm=1 --sampler=khop  >>../../results/ENGINE/logs/photo+LLM+Semi.log 
        python3 -u main.py --gnn_type=$GNN --re_split=$SPLIT --dataset=photo --encoder=Mistral-7B --run_times=4 --hidden=64 --n_layers=2  --lr=0.0005 --batch_norm=1 --sampler=khop  >>../../results/ENGINE/logs/photo+LLM+Semi.log 
     done 
done 


# History
for SPLIT in  0 ; do 
   for GNN in GCN; do 
        python3 -u main.py --gnn_type=$GNN --re_split=$SPLIT --dataset=history --encoder=$LM --run_times=4 --hidden=64 --n_layers=2  --batch_norm=1 --sampler=khop  >>../../results/ENGINE/logs/history+LM+Semi.log 
        python3 -u main.py --gnn_type=$GNN --re_split=$SPLIT --dataset=history --encoder=$LLM --run_times=4 --hidden=64 --n_layers=2  --batch_norm=1 --sampler=khop  >>../../results/ENGINE/logs/history+LLM+Semi.log 
        python3 -u main.py --gnn_type=$GNN --re_split=$SPLIT --dataset=history --encoder=Mistral-7B --run_times=4 --hidden=64 --n_layers=2 --batch_norm=1 --sampler=khop  >>../../results/ENGINE/logs/history+LLM+Semi.log 
     done 
done 


# Computer
for SPLIT in  0 ; do 
   for GNN in GCN; do 
        python3 -u main.py --gnn_type=$GNN --re_split=$SPLIT --dataset=computer --encoder=$LM --run_times=4 --hidden=64 --n_layers=2  --batch_norm=1 --sampler=khop  >>../../results/ENGINE/logs/computer+LM+Semi.log 
        python3 -u main.py --gnn_type=$GNN --re_split=$SPLIT --dataset=computer --encoder=$LLM --run_times=4 --hidden=64 --n_layers=2  --batch_norm=1 --sampler=khop  >>../../results/ENGINE/logs/computer+LLM+Semi.log 
        python3 -u main.py --gnn_type=$GNN --re_split=$SPLIT --dataset=computer --encoder=Mistral-7B --run_times=4 --hidden=64 --n_layers=2  --batch_norm=1 --sampler=khop  >>../../results/ENGINE/logs/computer+LLM+Semi.log 
     done 
done 


# arxiv
for GNN in GCN; do 
    python3 -u main.py --gnn_type=$GNN --re_split=0 --dataset=arxiv --encoder=$LM --run_times=3 --hidden=128 --n_layers=1 --epoch=300 --patience=50 --lr=0.0005 --batch_norm=1 --sampler=khop  >>../../results/ENGINE/logs/arxiv+LM.log 
    python3 -u main.py --gnn_type=$GNN --re_split=0 --dataset=arxiv --encoder=$LLM --run_times=3 --hidden=128 --n_layers=1 --epoch=300 --patience=50 --lr=0.0005 --batch_norm=1 --sampler=khop  >>../../results/ENGINE/logs/arxiv+LLM.log 
    python3 -u main.py --gnn_type=$GNN --re_split=0 --dataset=arxiv --encoder=Mistral-7B --run_times=3 --hidden=128 --n_layers=1 --epoch=300 --patience=50 --lr=0.0005 --batch_norm=1 --sampler=khop  >>../../results/ENGINE/logs/arxiv+LLM.log 
done 
