from gnn_trainer import GNNTrainer
import pandas as pd
from config import cfg, update_cfg
import time


def run(cfg):
    seeds = [cfg.seed] if cfg.seed is not None else range(cfg.runs)

    TRAINER = GNNTrainer

    all_scores = []
    start = time.time()
    for seed in seeds:
        cfg.seed = seed
        trainer = TRAINER(cfg, cfg.gnn.train.feature_type)
        cur_score, _ = trainer.train()
        all_scores.append(cur_score)

    end = time.time()

    if len(all_scores) > 1:
        df = pd.DataFrame(all_scores)
        print(f"[{cfg.gnn.model.name} + {cfg.gnn.train.feature_type}] Acc {df['test_acc'].mean():.2f} ± {df['test_acc'].std():.2f}, TestF1: {df['test_f1'].mean():.2f} ± {df['test_f1'].std():.2f}, Test weightF1: {df['test_weight_f1'].mean():.2f} ± {df['test_weight_f1'].std():.2f}")
    print(f"Running time: {(end-start)/len(seeds):.2f}s")


if __name__ == '__main__':
    cfg = update_cfg(cfg)
    run(cfg)
