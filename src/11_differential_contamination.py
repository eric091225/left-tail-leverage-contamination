#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
11 差异污染 vs 同污染：伪效应可以被凭空制造（对应论文 4.1 节，表4.1）

10 号脚本证明的是「污染会偏倚特征」。要构成一条**泄漏通道**，还须证明更强的一件事：

    在真值完全没有组间差异时，仅靠污染率与标签的关联，
    就能制造出统计显著的「发现」。

做法：构造两组合成被试（30 对 29），令两组的**真 a 与真 T_er 完全相同**，
唯一的真实差异是 v（构造 Cohen's d = −0.65）。序列长度按真实数据的组内分布抽样。
在此基础上比较两种污染配置：

  · 差异污染：两组污染率取自实测（年轻 0.23%、老年 0.97%）
  · 同污染对照：两组污染率相同（0.97%/0.97%，以及 0.23%/0.23%）

若伪效应来自污染**本身**，则同污染对照下它应当依然出现；
若来自污染率与标签的**关联**，则同污染对照下它必须消失。

用法：
    python 11_differential_contamination.py             # 完整 200 次重复
    python 11_differential_contamination.py --reps 20   # 快速校准
"""
import argparse
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

from swald import fit_swald

OUT = Path("../results/tables/table4_differential.csv")
PKL = Path("../data/rt_data.pkl")

TRUE_A, TRUE_TER = 1.2, 0.30      # 两组完全相同
V_MEAN, V_SD, V_D = 3.0, 1.0, -0.65   # 唯一的真实差异：v 的组间 Cohen's d
N_YOUNG, N_OLD = 30, 29
CONTAM_LO, CONTAM_HI = 0.150, 0.250
RATE_YOUNG, RATE_OLD = 0.0023, 0.0097   # 实测污染率（250 ms 阈值下的剔除比例）
THRESHOLDS = (0.150, 0.250)


def cohens_d(x, y):
    """x 为年轻组、y 为老年组；返回 (老年 − 年轻) 方向的 d，与论文一致。"""
    n1, n2 = len(x), len(y)
    s = np.sqrt(((n1 - 1) * np.var(x, ddof=1) + (n2 - 1) * np.var(y, ddof=1)) / (n1 + n2 - 2))
    return (np.mean(y) - np.mean(x)) / s if s > 0 else np.nan


def load_length_pools():
    """取真实数据的组内试次数分布，供合成被试抽样长度。"""
    with open(PKL, "rb") as f:
        d = pickle.load(f)
    rt, lab = d["rt_data"], d["labels"]
    yl = [len(rt[s]) for s in rt if lab[s] == 0]
    ol = [len(rt[s]) for s in rt if lab[s] == 1]
    return np.array(yl), np.array(ol)


def one_subject(v, n, p_contam, thresh, rng):
    t = TRUE_TER + rng.wald(TRUE_A / v, TRUE_A ** 2, size=n)
    mask = rng.random(n) < p_contam
    k = int(mask.sum())
    if k:
        t[mask] = rng.uniform(CONTAM_LO, CONTAM_HI, k)
    t = t[(t >= thresh) & (t <= 4.0)]
    if len(t) < 20:
        return None
    r = fit_swald(t, seed=int(rng.integers(1 << 30)))
    return r["v"], r["a"], r["t_er"]


def one_rep(rate_y, rate_o, thresh, ylens, olens, rng):
    vy = rng.normal(V_MEAN - V_D * 0.5 * V_SD, V_SD, N_YOUNG)   # 年轻组 v 偏高
    vo = rng.normal(V_MEAN + V_D * 0.5 * V_SD, V_SD, N_OLD)     # 老年组 v 偏低
    vy, vo = np.clip(vy, 0.3, None), np.clip(vo, 0.3, None)
    Y, O = [], []
    for v in vy:
        r = one_subject(v, int(rng.choice(ylens)), rate_y, thresh, rng)
        if r: Y.append(r)
    for v in vo:
        r = one_subject(v, int(rng.choice(olens)), rate_o, thresh, rng)
        if r: O.append(r)
    Y, O = np.array(Y), np.array(O)
    if len(Y) < 10 or len(O) < 10:
        return None
    return [cohens_d(Y[:, k], O[:, k]) for k in range(3)]   # v, a, T_er


def main(reps, seed):
    rng = np.random.default_rng(seed)
    ylens, olens = load_length_pools()
    print(f"长度池：年轻 n={len(ylens)} 均值 {ylens.mean():.1f}，"
          f"老年 n={len(olens)} 均值 {olens.mean():.1f}")
    print(f"真值：两组 a={TRUE_A}、T_er={TRUE_TER} 完全相同；v 的构造 d={V_D}\n")

    configs = [
        ("差异污染 0.23%／0.97%（实测比例）", RATE_YOUNG, RATE_OLD),
        ("同污染 0.97%／0.97%（对照）",       RATE_OLD,   RATE_OLD),
        ("同污染 0.23%／0.23%（对照）",       RATE_YOUNG, RATE_YOUNG),
    ]
    rows = []
    for name, ry, ro in configs:
        for th in THRESHOLDS:
            ds = [r for r in (one_rep(ry, ro, th, ylens, olens, rng) for _ in range(reps)) if r]
            arr = np.array(ds)
            cell = {"污染配置": name, "阈值": f"{int(th*1000)} ms"}
            for k, pname in enumerate(["v", "a", "T_er"]):
                m = np.nanmean(arr[:, k])
                lo, hi = np.nanpercentile(arr[:, k], [2.5, 97.5])
                cell[f"{pname} 的 d"] = f"{m:+.2f} [{lo:+.2f}, {hi:+.2f}]"
                cell[f"_{pname}_excl0"] = bool(lo > 0 or hi < 0)
            rows.append(cell)
            print(f"  {name:28} {cell['阈值']:>7}  "
                  f"v {cell['v 的 d']:22} a {cell['a 的 d']:22} T_er {cell['T_er 的 d']}")

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\n已保存 → {OUT}")

    print("\n判据（真值为零的两个参数 a 与 T_er）：")
    for r in rows:
        if r["阈值"] != "150 ms":
            continue
        flag = r["_a_excl0"] or r["_T_er_excl0"]
        tag = "★ 出现伪效应（区间不含 0）" if flag else "无伪效应（区间均含 0）"
        print(f"  {r['污染配置']:28} → {tag}")
    print("\n若仅『差异污染』一行出现伪效应，则偏倚来源是污染率与标签的关联，"
          "而非污染的绝对水平。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260821)
    a = ap.parse_args()
    main(a.reps, a.seed)
