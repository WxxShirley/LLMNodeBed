from gnn_trainer import GNNTrainer
import pandas as pd
from config import cfg, update_cfg
import time


def run(cfg):
    seeds = [cfg.seed] if cfg.seed is not None else range(cfg.runs)

    TRAINER = GNNTrainer

    all_acc = []
    start = time.time()
    for seed in seeds:
        cfg.seed = seed
        trainer = TRAINER(cfg, cfg.gnn.train.feature_type)
        trainer.train()
        _, acc = trainer.eval_and_save()
        all_acc.append(acc)
    end = time.time()

    if len(all_acc) > 1:
        df = pd.DataFrame(all_acc)
        print(f"[{cfg.gnn.model.name} + {cfg.gnn.train.feature_type}] ValACC: {df['val_acc'].mean():.4f} ± {df['val_acc'].std():.4f}, TestAcc: {df['test_acc'].mean():.4f} ± {df['test_acc'].std():.4f}")
    print(f"Running time: {(end-start)/len(seeds):.2f}s")


if __name__ == '__main__':
    cfg = update_cfg(cfg)
    run(cfg)
