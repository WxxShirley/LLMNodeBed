LOG_DIR=../../results/ZeroG/logs
mkdir $LOG_DIR

# Single dataset 
python3 -u main.py --train_dataset arxiv --test_dataset cora wikics --k 2 --if_norm --device 0 --epoch 2 >>$LOG_DIR/academic.log 
python3 -u main.py --train_dataset reddit --test_dataset instagram --k 2 --if_norm --device 0 --epoch 1 >>$LOG_DIR/social.log
python3 -u main.py --train_dataset computer --test_dataset photo --k 2 --if_norm --device 0 --epoch 1  >>$LOG_DIR/commerce.log 

# Full
python3 -u main.py --train_dataset reddit computer arxiv  --test_dataset cora wikics instagram photo --k 2 --if_norm --device 1 --epoch 1 >>$LOG_DIR/full.log 
