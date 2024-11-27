# Run on 2 A100 devices
WANDB_DISABLED=True python main.py --num_epoch=20 --llm=Mistral-7B  --batch_size=16  --re_split=0 --dataset=cora
WANDB_DISABLED=True python main.py --num_epoch=4 --llm=Mistral-7B  --batch_size=16  --re_split=1  --dataset=cora
