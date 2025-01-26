GPU=0

mkdir ../../results/GraphGPT/logs/

# For cora, citeseer, instagram, epochs are 20 and 10
for DATASET in cora citeseer instagram ; do 
     for SEED in 0 1 2 3  ; do
        python3 -u main.py --dataset=$DATASET --do_stage1=0 --seed=$SEED --load_ground_embedding=0 --re_split=0 --s2_epoch=20 --gpu_id=$GPU --s2_batch_size=16  >>../../results/GraphGPT/logs/$DATASET+Semi.log 
        python3 -u main.py --dataset=$DATASET --do_stage1=1 --seed=$SEED --load_ground_embedding=0 --re_split=1 --s2_epoch=10 --gpu_id=$GPU --s2_batch_size=16  >>../../results/GraphGPT/logs/$DATASET.log 
    done 
done 


# For pubmed and wikics, epochs are 10 and 6
for DATASET in wikics pubmed ; do 
     for SEED in 0 1 2 3  ; do
        python3 -u main.py --dataset=$DATASET --do_stage1=1 --s1_epoch=1 --seed=$SEED --load_ground_embedding=0 --re_split=0 --s2_epoch=10 --gpu_id=$GPU --s2_batch_size=16  >>../../results/GraphGPT/logs/$DATASET+Semi.log 
        python3 -u main.py --dataset=$DATASET --do_stage1=1 --seed=$SEED --load_ground_embedding=0 --re_split=1 --s2_epoch=6 --gpu_id=$GPU --s2_batch_size=16  >>../../results/GraphGPT/logs/$DATASET.log 
    done 
done 


for DATASET in reddit photo history computer ; do 
    for SEED in  0 1 2 3 ; do
        python3 -u main.py --dataset=$DATASET --do_stage1=1  --s1_epoch=1  --seed=$SEED --load_ground_embedding=0 --re_split=0 --s2_epoch=4 --s2_batch_size=16  >>../../results/GraphGPT/logs/$DATASET+Semi.log 
        python3 -u main.py --dataset=$DATASET --do_stage1=0  --seed=$SEED --load_ground_embedding=0 --re_split=1 --s2_epoch=2 --s2_batch_size=16  >>../../results/GraphGPT/logs/$DATASET.log 
    done 
done 


for SEED in 0 1 2 3 ; do 
   python3 -u main.py --dataset=arxiv --do_stage1=0 --seed=$SEED --load_ground_embedding=0 --re_split=0 --s2_epoch=2 >>../../results/GraphGPT/logs/arxiv.log 
done
