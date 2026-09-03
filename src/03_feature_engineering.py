"""
03 特征工程：构造另外两组特征。

  - RT 朴素基线组（3 维）：rt_mean, rt_std, rt_cv
  - 18 维候选池：供 04 在 LOOCV 内部做 RFE（这里只存池，不做选择）

修正点：本脚本独立重载 pkl（不依赖 02 的内存变量），可单独运行。
全部特征基于「秒」的 RT。
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

PKL_IN     = Path("../data/rt_data.pkl")
RT_OUT     = Path("../results/features/rt_baseline.csv")
POOL_OUT   = Path("../results/features/feature_pool_18d.csv")


def extract_18_features(rt):
    """18 个常见的反应时描述统计量（RFE 的候选原料；按秒）。"""
    f = {}
    f["mean"]   = np.mean(rt)
    f["median"] = np.median(rt)
    f["mode"]   = float(stats.mode(rt, keepdims=True)[0][0])
    f["std"]    = np.std(rt)
    f["iqr"]    = np.percentile(rt, 75) - np.percentile(rt, 25)
    f["cv"]     = np.std(rt) / np.mean(rt)
    f["skew"]   = stats.skew(rt)
    f["kurtosis"] = stats.kurtosis(rt)
    f["p10"]    = np.percentile(rt, 10)
    f["p25"]    = np.percentile(rt, 25)
    f["p75"]    = np.percentile(rt, 75)
    f["p90"]    = np.percentile(rt, 90)
    f["min"]    = np.min(rt)
    f["max"]    = np.max(rt)
    f["range"]  = f["max"] - f["min"]
    f["first_response"]   = rt[0] if len(rt) > 0 else 0.0
    f["sliding_std_mean"] = np.mean([np.std(rt[i:i+5]) for i in range(len(rt)-4)]) \
                            if len(rt) >= 5 else np.std(rt)
    f["interval_mean"]    = np.mean(np.abs(np.diff(rt))) if len(rt) > 1 else 0.0
    return f


def main():
    with open(PKL_IN, "rb") as f:
        data = pickle.load(f)
    rt_data, labels = data["rt_data"], data["labels"]

    rt_rows, pool_rows = {}, {}
    for subj, rt in rt_data.items():
        rt_rows[subj] = {
            "rt_mean": np.mean(rt),
            "rt_std":  np.std(rt),
            "rt_cv":   np.std(rt) / np.mean(rt),
            "label":   labels[subj],
        }
        feat = extract_18_features(rt)
        feat["label"] = labels[subj]
        pool_rows[subj] = feat

    RT_OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame.from_dict(rt_rows, orient="index").to_csv(RT_OUT)
    pd.DataFrame.from_dict(pool_rows, orient="index").to_csv(POOL_OUT)
    print(f"已保存 RT 基线组（3 维）→ {RT_OUT}")
    print(f"已保存 18 维候选池 → {POOL_OUT}（RFE 在 04 的 LOOCV 内部做选择）")


if __name__ == "__main__":
    main()
