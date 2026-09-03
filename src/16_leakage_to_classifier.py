#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
16 泄漏在分类器层面的直接演示（对应论文 4.2 节）

11 号脚本证明：污染率与标签的关联能在**参数层面**制造伪效应（真值为零的参数上
出现 d = −1.00）。但「泄漏」一词的定义落在模型性能上——必须证明这个伪效应确实
进入了分类器、把 AUC 抬起来，才谈得上是一条泄漏通道。本脚本补这一步。

设计与 11 号严格同构：两组合成被试（30 对 29），**真 a 与真 T_er 完全相同**，
唯一真实差异是 v（构造 Cohen's d = −0.65）。拟合出的三个参数当特征，跑 LOOCV
逻辑回归（标准化在折内完成），比较四种污染配置下的 AUC。

三条 AUC 分别回答三个问题：

  AUC_full  {v, a, T_er}  —— 现实中会报告的那个数
  AUC_zero  {a, T_er}     —— **判决性对照**：这两个参数的真值差为零，
                             干净与同污染下应在 0.5 附近；差异污染下若显著
                             高于 0.5，则分类器确实吃到了污染率这个变量
  AUC_v     {v}           —— 真实信号本身，用于显示污染对它的衰减

用法：
    python 16_leakage_to_classifier.py            # 完整 200 次重复
    python 16_leakage_to_classifier.py --reps 10  # 快速校准，输出不可用于结论
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from swald import fit_swald

OUT = Path("../results/tables/table_leakage_classifier.csv")

TRUE_A, TRUE_TER = 1.2, 0.30          # 两组完全相同
V_MEAN, V_SD, V_D = 3.0, 1.0, -0.65   # 唯一的真实差异
N_YOUNG, N_OLD = 30, 29
N_TRIALS = 175
CONTAM_LO, CONTAM_HI = 0.150, 0.250
RATE_Y, RATE_O = 0.0023, 0.0097
THRESHOLDS = (0.150, 0.250)

CONFIGS = [("干净（0%／0%）",                 0.0,    0.0),
           ("差异污染 0.23%／0.97%（实测比例）", RATE_Y, RATE_O),
           ("同污染 0.97%／0.97%（对照）",      RATE_O, RATE_O),
           ("同污染 0.23%／0.23%（对照）",      RATE_Y, RATE_Y)]


def one_subject(v, p, thresh, rng):
    t = TRUE_TER + rng.wald(TRUE_A / v, TRUE_A ** 2, size=N_TRIALS)
    m = rng.random(N_TRIALS) < p
    k = int(m.sum())
    if k:
        t[m] = rng.uniform(CONTAM_LO, CONTAM_HI, k)
    t = t[(t >= thresh) & (t <= 4.0)]
    if len(t) < 20:
        return None
    r = fit_swald(t, seed=int(rng.integers(1 << 30)))
    return [r["v"], r["a"], r["t_er"]]


def loocv_auc(X, y):
    """留一交叉验证 AUC；标准化与拟合均在折内完成（论文闸三的要求）。"""
    n = len(y)
    oof = np.empty(n)
    for i in range(n):
        tr = np.ones(n, bool); tr[i] = False
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=1000).fit(sc.transform(X[tr]), y[tr])
        oof[i] = clf.predict_proba(sc.transform(X[i:i+1]))[0, 1]
    return roc_auc_score(y, oof)


def one_rep(py, po, thresh, rng):
    vy = rng.normal(V_MEAN - V_D * 0.5 * V_SD, V_SD, N_YOUNG)
    vo = rng.normal(V_MEAN + V_D * 0.5 * V_SD, V_SD, N_OLD)
    vy, vo = np.clip(vy, 0.3, None), np.clip(vo, 0.3, None)
    Y = [r for r in (one_subject(v, py, thresh, rng) for v in vy) if r]
    O = [r for r in (one_subject(v, po, thresh, rng) for v in vo) if r]
    if len(Y) < 10 or len(O) < 10:
        return None
    X = np.array(Y + O)
    y = np.r_[np.zeros(len(Y), int), np.ones(len(O), int)]
    return (loocv_auc(X, y),                      # {v, a, T_er}
            loocv_auc(X[:, [1, 2]], y),           # {a, T_er}  真值差为零
            loocv_auc(X[:, [0]], y))              # {v}        真实信号


def main(reps, seed):
    rng = np.random.default_rng(seed)
    print(f"真值：两组 a={TRUE_A}、T_er={TRUE_TER} 完全相同；v 的构造 d={V_D}")
    print(f"每格 {reps} 次重复，每次 {N_YOUNG}+{N_OLD} 名合成被试、各 {N_TRIALS} 试次\n")
    rows = []
    for name, py, po in CONFIGS:
        for th in THRESHOLDS:
            res = [r for r in (one_rep(py, po, th, rng) for _ in range(reps)) if r]
            arr = np.array(res)
            cell = {"污染配置": name, "阈值": f"{int(th*1000)} ms"}
            for j, tag in enumerate(["AUC_full{v,a,T_er}", "AUC_zero{a,T_er}", "AUC_v{v}"]):
                m = arr[:, j].mean()
                lo, hi = np.percentile(arr[:, j], [2.5, 97.5])
                cell[tag] = f"{m:.3f} [{lo:.3f}, {hi:.3f}]"
                cell["_" + tag] = float(m)
                if tag == "AUC_zero{a,T_er}":
                    cell["_zero_lo"] = float(lo)
            rows.append(cell)
            print(f"  {name:30} {cell['阈值']:>7}  "
                  f"全部三维 {cell['AUC_full{v,a,T_er}']:22} "
                  f"仅真值为零两维 {cell['AUC_zero{a,T_er}']:22} "
                  f"仅 v {cell['AUC_v{v}']}")

    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in rows])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\n已保存 → {OUT}")

    print("\n判据（150 ms 口径，只看真值差为零的 {a, T_er} 两维）：")
    print("  若该 AUC 的 95% 区间下界高于 0.5，说明分类器从「真值毫无差异」的参数里"
          "读出了类别信息——这就是泄漏在分类器层面的直接证据。\n")
    base = None
    for r in rows:
        if r["阈值"] != "150 ms":
            continue
        z, lo = r["_AUC_zero{a,T_er}"], r["_zero_lo"]
        flag = "★ 高于 0.5（泄漏）" if lo > 0.5 else "未高于 0.5"
        print(f"  {r['污染配置']:30} AUC_zero = {z:.3f}，区间下界 {lo:.3f} → {flag}")
        if r["污染配置"].startswith("干净"):
            base = z
    if base is not None:
        d = [r for r in rows if r["阈值"] == "150 ms"
             and r["污染配置"].startswith("差异")][0]
        print(f"\n  差异污染相对干净基准的抬升：全部三维 "
              f"{d['_AUC_full{v,a,T_er}'] - [r for r in rows if r['阈值']=='150 ms' and r['污染配置'].startswith('干净')][0]['_AUC_full{v,a,T_er}']:+.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260821)
    a = ap.parse_args()
    main(a.reps, a.seed)
