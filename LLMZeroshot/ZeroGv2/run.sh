for SEED in 1 2 3 4; do
    python main_TextBP_benchmark.py --seed=$SEED --train_dataset arxiv --test_dataset cora pubmed wikics --k 2 --if_norm --device 2 --epoch 1 >>../../results/ZeroG/logs/arxiv.log
    python main_TextBP_benchmark.py --seed=$SEED --train_dataset reddit --test_dataset instagram --k 2 --if_norm --device 2 --epoch 1 >>../../results/ZeroG/logs/reddit.log
    python main_TextBP_benchmark.py --seed=$SEED --train_dataset computer --test_dataset photo --k 2 --if_norm --device 2 --epoch 1 >>../../results/ZeroG/logs/computer.log
done 


# nohup ./run.sh >/data/fge2/LLMGraphBench/results/ZeroG/logs/5retest.log