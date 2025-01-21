LOG_DIR=../../results/ZeroG/logs

# Comment this line if the folder already exists
mkdir $LOG_DIR

# Intra-domain Transfer
python3 -u main.py --train_dataset arxiv --test_dataset cora wikics --k 2 --if_norm --device 0 --epoch 2 >>$LOG_DIR/academic.log 
python3 -u main.py --train_dataset reddit --test_dataset instagram --k 2 --if_norm --device 0 --epoch 1 >>$LOG_DIR/social.log
python3 -u main.py --train_dataset computer --test_dataset photo --k 2 --if_norm --device 0 --epoch 1  >>$LOG_DIR/commerce.log 
