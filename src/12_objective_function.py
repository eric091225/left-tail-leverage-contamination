#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12 只更换目标函数的受控对照（对应论文 2.2 节）

问题：既然污染对最大似然的偏倚如此严重，反应时领域为何长期无人报告？
假设：原因在于**拟合目标不同**。分位点卡方把一个污染观测的影响限制在它所属的
      那一个区间之内（影响有界）；最大似然则对落在分布支撑边界附近的观测给予
      无界的杠杆——这正是左尾杠杆机制。

检验：以同一个解析 SWald 生成数据并注入污染，对**同一个数组**分别用
      最大似然与分位点卡方拟合。两者共用同一解析密度、均无随机游走模拟，
      唯一的差别就是目标函数。最后在真实数据上并排拟合。

用法：python 12_objective_function.py [--reps 2000]

注：默认 2000 次重复不可调小。0.97% 污染下 175 试次平均只有 1.7 个污染观测，
    是否命中由伯努利决定，而 T_er 塌缩对污染个数高度非线性——200 次重复时
    该均值在 0.050–0.075 间摆动，2000 次时两个不同种子的差异 < 0.001。
"""
import argparse
import numpy as np, pandas as pd
from pathlib import Path
from scipy import stats
from scipy.optimize import minimize
from swald import fit_swald

OUT = Path("../results/tables/table_objective.csv")
RAW = Path("../data/ratcliff_vanunu_2022.csv")
TRUE_V, TRUE_A, TRUE_TER, N_TRIALS = 3.0, 1.2, 0.300, 175
CONTAM_LO, CONTAM_HI = 0.150, 0.250
Q = np.arange(0.05, 1.0, 0.05)          # 19 个分位点 → 20 个区间


def swald_cdf(t, v, a, ter):
    return stats.invgauss.cdf(t, mu=(a / v) / a ** 2, loc=ter, scale=a ** 2)


def quantile_chi2(params, t, qt):
    """分位点卡方：一个污染观测至多把最低分位点推移一个次序统计量，影响有界。"""
    v, a, ter = params
    if v <= 0 or a <= 0 or ter <= 0 or ter >= qt[0]:
        return 1e12
    edges = np.concatenate([[-np.inf], qt, [np.inf]])
    pred = np.diff(np.concatenate([[0.0], swald_cdf(edges[1:-1], v, a, ter), [1.0]]))
    pred = np.clip(pred, 1e-9, None)
    obs = np.full(len(pred), 1.0 / len(pred))     # 由分位点构造，各区间等比例
    return len(t) * np.sum((obs - pred) ** 2 / pred)


def fit_chi2(t, rng):
    qt = np.quantile(t, Q)
    mn = float(np.min(t))
    bounds = [(1e-3, 50.0), (1e-3, 20.0), (1e-4, max(mn - 1e-3, mn * 0.99))]
    best = None
    for x0 in [np.array([1.0, 1.0, 0.5 * mn])] + [
            np.array([rng.uniform(.5, 4), rng.uniform(.5, 3), rng.uniform(.2, .8) * mn])
            for _ in range(5)]:
        x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
        try:
            r = minimize(quantile_chi2, x0, args=(t, qt), method="L-BFGS-B",
                         bounds=bounds, options={"maxiter": 500})
        except Exception:
            continue
        if np.isfinite(r.fun) and (best is None or r.fun < best.fun):
            best = r
    return float(best.x[2]) if best is not None else np.nan


def main(reps, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for p, name in [(0.0, "仿真 · 0（干净）"), (0.0097, "仿真 · 0.97%（老年组实测污染率）"),
                    (0.02, "仿真 · 2%")]:
        mle, ch2 = [], []
        for _ in range(reps):
            t = TRUE_TER + rng.wald(TRUE_A / TRUE_V, TRUE_A ** 2, size=N_TRIALS)
            m = rng.random(N_TRIALS) < p
            if m.sum(): t[m] = rng.uniform(CONTAM_LO, CONTAM_HI, int(m.sum()))
            mle.append(fit_swald(t, seed=int(rng.integers(1 << 30)))["t_er"])
            ch2.append(fit_chi2(t, rng))
        a, b = float(np.nanmean(mle)), float(np.nanmean(ch2))
        rows.append({"条件": name, "最大似然 T_er": round(a, 4),
                     "分位点卡方 T_er": round(b, 4),
                     "最大似然相对偏差": f"{(a-TRUE_TER)/TRUE_TER*100:+.1f}%",
                     "分位点卡方相对偏差": f"{(b-TRUE_TER)/TRUE_TER*100:+.1f}%"})
        print(f"  {name:28} MLE={a:.4f}  卡方={b:.4f}")

    # 真实数据部分固定用 150 ms 口径——论文 2.2 节的论断（d 由 −0.70 变为 +0.10）
    # 正是在该口径下作出的；此处直读原始文件，不依赖 01_preprocess 当前的阈值设定。
    raw = pd.read_csv(RAW)
    raw = raw[(raw.Task == 1) & (raw.RT >= 150) & (raw.RT <= 4000)]
    est = {0: {"mle": [], "c2": []}, 1: {"mle": [], "c2": []}}
    for _sid, g in raw.groupby("Subject"):
        t = g.RT.to_numpy(float) / 1000.0
        grp = 0 if int(g.Age.iloc[0]) == 1 else 1
        est[grp]["mle"].append(fit_swald(t, seed=0)["t_er"])
        est[grp]["c2"].append(fit_chi2(t, rng))
    def cd(x, y):
        n1, n2 = len(x), len(y)
        sp = np.sqrt(((n1-1)*np.var(x, ddof=1)+(n2-1)*np.var(y, ddof=1))/(n1+n2-2))
        return (np.mean(y)-np.mean(x))/sp
    for k, nm in [("mle", "最大似然"), ("c2", "分位点卡方")]:
        yy, oo = np.array(est[0][k]), np.array(est[1][k])
        t_, p_ = stats.ttest_ind(yy, oo)
        rows.append({"条件": f"真实数据(150ms) · {nm}", "最大似然 T_er": "", "分位点卡方 T_er": "",
                     "最大似然相对偏差": f"年轻 {yy.mean():.3f} / 老年 {oo.mean():.3f}",
                     "分位点卡方相对偏差": f"d = {cd(yy,oo):+.2f}, p = {p_:.4f}"})
        print(f"  真实数据(150ms) · {nm:10} 年轻 {yy.mean():.3f} 老年 {oo.mean():.3f}  "
              f"d={cd(yy,oo):+.2f} p={p_:.4f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\n已保存 → {OUT}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260821)
    a = ap.parse_args(); main(a.reps, a.seed)
