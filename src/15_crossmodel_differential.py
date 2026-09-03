#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
15 跨模型验证（续）：三参数 Weibull 上的差异污染

14 号脚本证明「左尾杠杆机制在 Weibull 上同样出现，且由密度在支撑下界处的
行为开关」。本脚本补上第二级——泄漏通道是否也跨模型复现。

做法与 11 号脚本严格同构：构造两组合成样本（30 对 29），令两组的
**真位置参数 gamma 与真尺度参数 lam 完全相同**，唯一的真实差异是形状 k
（构造 Cohen's d = −0.65）。在此基础上比较：

  · 差异污染：两组污染率取自实测（0.23% / 0.97%）
  · 同污染对照：两组污染率相同

若泄漏通道是「密度趋零 + MLE 估计支撑下界」这一组合的性质而非 SWald 特有，
则真值为零的 gamma 与 lam 上应当在差异污染下出现伪效应、在同污染下消失。

用法：
    python 15_crossmodel_differential.py            # 完整 200 次重复
    python 15_crossmodel_differential.py --reps 20  # 快速校准，输出不可用于结论
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from weibull3 import sample_weibull3, fit_weibull3

OUT = Path("../results/tables/table_crossmodel_differential.csv")

TRUE_GAMMA, TRUE_LAM = 0.30, 0.45      # 两组完全相同
K_MEAN, K_SD, K_D = 2.0, 0.5, -0.65    # 唯一的真实差异
N_YOUNG, N_OLD = 30, 29
N_TRIALS = 175
CONTAM_LO, CONTAM_HI = 0.150, 0.250
RATE_YOUNG, RATE_OLD = 0.0023, 0.0097
THRESHOLDS = (0.150, 0.250)


def cohens_d(x, y):
    n1, n2 = len(x), len(y)
    s = np.sqrt(((n1-1)*np.var(x, ddof=1) + (n2-1)*np.var(y, ddof=1)) / (n1+n2-2))
    return (np.mean(y) - np.mean(x)) / s if s > 0 else np.nan


def one_subject(k, p_contam, thresh, rng):
    t = sample_weibull3(TRUE_GAMMA, TRUE_LAM, k, N_TRIALS, rng)
    mask = rng.random(N_TRIALS) < p_contam
    n = int(mask.sum())
    if n:
        t[mask] = rng.uniform(CONTAM_LO, CONTAM_HI, n)
    t = t[(t >= thresh) & (t <= 4.0)]
    if len(t) < 20:
        return None
    r = fit_weibull3(t, seed=int(rng.integers(1 << 30)))
    return r["gamma"], r["lam"], r["k"]


def one_rep(ry, ro, thresh, rng):
    ky = rng.normal(K_MEAN - K_D*0.5*K_SD, K_SD, N_YOUNG)
    ko = rng.normal(K_MEAN + K_D*0.5*K_SD, K_SD, N_OLD)
    ky, ko = np.clip(ky, 1.05, None), np.clip(ko, 1.05, None)   # 保持密度趋零区
    Y = [r for r in (one_subject(k, ry, thresh, rng) for k in ky) if r]
    O = [r for r in (one_subject(k, ro, thresh, rng) for k in ko) if r]
    if len(Y) < 10 or len(O) < 10:
        return None
    Y, O = np.array(Y), np.array(O)
    return [cohens_d(Y[:, i], O[:, i]) for i in range(3)]        # gamma, lam, k


def main(reps, seed):
    rng = np.random.default_rng(seed)
    print(f"真值：两组 gamma={TRUE_GAMMA}、lam={TRUE_LAM} 完全相同；"
          f"k 的构造 d={K_D}\n")
    configs = [("差异污染 0.23%／0.97%（实测比例）", RATE_YOUNG, RATE_OLD),
               ("同污染 0.97%／0.97%（对照）",       RATE_OLD,   RATE_OLD),
               ("同污染 0.23%／0.23%（对照）",       RATE_YOUNG, RATE_YOUNG)]
    rows = []
    for name, ry, ro in configs:
        for th in THRESHOLDS:
            ds = [r for r in (one_rep(ry, ro, th, rng) for _ in range(reps)) if r]
            arr = np.array(ds)
            cell = {"污染配置": name, "阈值": f"{int(th*1000)} ms"}
            for i, pn in enumerate(["gamma", "lam", "k"]):
                m = np.nanmean(arr[:, i])
                lo, hi = np.nanpercentile(arr[:, i], [2.5, 97.5])
                cell[f"{pn} 的 d"] = f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]"
                cell[f"_{pn}_excl0"] = bool(lo > 0 or hi < 0)
            rows.append(cell)
            print(f"  {name:28} {cell['阈值']:>7}  gamma {cell['gamma 的 d']:22} "
                  f"lam {cell['lam 的 d']:22} k {cell['k 的 d']}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    print(f"\n已保存 → {OUT}")

    print("\n判据（真值为零的两个参数 gamma 与 lam，150 ms 口径）：")
    for r in rows:
        if r["阈值"] != "150 ms":
            continue
        flag = r["_gamma_excl0"] or r["_lam_excl0"]
        print(f"  {r['污染配置']:28} → "
              f"{'★ 出现伪效应（区间不含 0）' if flag else '无伪效应（区间均含 0）'}")
    print("\n若仅『差异污染』一行出现伪效应，则泄漏通道在 Weibull 上同样成立，"
          "即它不是 SWald 一族的特性。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260821)
    a = ap.parse_args()
    main(a.reps, a.seed)
