"""
02 SWald 特征提取：对每位被试的 RT 序列（秒）做 SWald MLE 拟合。

修正点：
  - import 测过的 fit_swald（src/swald.py，L-BFGS-B + 多重启），不再有未定义函数；
  - 输入是「秒」，于是 T_er≈0.27、RT_cog≈0.4 这种量级，和论文一致；
  - 记录每位被试是否收敛，异常被试的数量与处理方式随结果一并输出。
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from swald import fit_swald   # 同目录 src/swald.py

PKL_IN   = Path("../data/rt_data.pkl")
FEAT_OUT = Path("../results/features/swald_features.csv")


def main():
    with open(PKL_IN, "rb") as f:
        data = pickle.load(f)
    rt_data, labels = data["rt_data"], data["labels"]

    rows = {}
    n_bad = 0
    for subj, rt in rt_data.items():
        fit = fit_swald(rt)                       # rt 已是秒
        if (not fit["converged"]) or (not np.isfinite(fit["v"])) \
           or fit["t_er"] >= np.min(rt) or fit["t_er"] <= 0:
            n_bad += 1
            print(f"[警告] 被试 {subj} 拟合可疑：v={fit['v']:.3g}, "
                  f"a={fit['a']:.3g}, T_er={fit['t_er']:.3g}, 试次={fit['n_trials']}")
        rows[subj] = {
            "v": fit["v"], "a": fit["a"], "ter": fit["t_er"],
            "rt_cog": fit["rt_cog"], "n_trials": fit["n_trials"],
            "converged": fit["converged"], "label": labels[subj],
        }

    df = pd.DataFrame.from_dict(rows, orient="index")
    FEAT_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEAT_OUT)
    print(f"\n完成 SWald 拟合：{len(df)} 位被试，可疑/不收敛 {n_bad} 位。")
    print(f"已保存：{FEAT_OUT}")
    # 把 n_bad 与可疑被试列表记下来，供核查
    if n_bad:
        print("→ 上列被试的 a 估计需谨慎解读；本文的处理是保留并标注。")


if __name__ == "__main__":
    main()
