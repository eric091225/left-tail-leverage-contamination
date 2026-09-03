"""
05 统计检验：SWald-TG vs RFE-DD 的 AUC 差异（主分类器 LR）。

修正点：
  - DeLong 用测过的真实实现（src/delong.py，Sun & Xu 快速算法），不再是空壳；
  - 单侧 H_A：AUC_TG > AUC_DD；
  - Bootstrap 加单类保护，避免重采样到只剩一类时 roc_auc_score 报错。
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score
from delong import delong_roc_test   # 同目录 src/delong.py

PRED_IN  = Path("../results/predictions_all.csv")
SEED     = 42
N_BOOT   = 1000


def bootstrap_ci_diff(y, p1, p2, n_boot=N_BOOT, seed=SEED):
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []
    while len(diffs) < n_boot:
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():      # 单类，跳过
            continue
        diffs.append(roc_auc_score(yb, p1[idx]) - roc_auc_score(yb, p2[idx]))
    diffs = np.array(diffs)
    return diffs.mean(), np.percentile(diffs, 2.5), np.percentile(diffs, 97.5)


def main():
    df = pd.read_csv(PRED_IN)
    y = df["true_label"].values.astype(int)
    p_tg = df["TG_lr"].values     # 主分类器 LR 上的 SWald 理论组
    p_dd = df["DD_lr"].values     # 主分类器 LR 上的 RFE 数据驱动组

    auc_tg, auc_dd, z, p = delong_roc_test(y, p_tg, p_dd, alternative="greater")
    m, lo, hi = bootstrap_ci_diff(y, p_tg, p_dd)

    print("==== 主比较：SWald-TG vs RFE-DD（逻辑回归）====")
    print(f"AUC_TG = {auc_tg:.4f}")
    print(f"AUC_DD = {auc_dd:.4f}")
    print(f"ΔAUC   = {auc_tg - auc_dd:+.4f}")
    print(f"DeLong 单侧 (H_A: TG>DD): z = {z:.3f},  p = {p:.4f}")
    print(f"Bootstrap ΔAUC 95% CI = [{lo:+.4f}, {hi:+.4f}]  (均值 {m:+.4f})")
    sig = (p < 0.05) and (lo > 0)
    print(f"结论：{'差异显著' if sig else '差异不显著'}")
    print("→ 对应论文第七章的五路线段；该比较因功效不足，不作为本文结论。")


if __name__ == "__main__":
    main()
