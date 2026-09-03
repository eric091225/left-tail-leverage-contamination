#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
13 参数恢复检验的盲区（对应论文第五章，表5.1）

参数恢复检验是把生成模型参数当作特征时的标准准入手段：用已知真参数生成合成数据，
用同一拟合函数还原，比较真值与估计值。它的通行做法有一处可证实的盲区——
它检验的是「理想条件下估计器是否可用」，而不是「本数据实际所处的条件下是否可用」。

三个区块：
  A  窄网格 · 干净   —— 通行做法：真参数取自文献常见范围，数据不含污染
  B  扩展网格 · 干净 —— 网格扩展到覆盖本数据的真实拟合值，数据仍不含污染
  C  扩展网格 · 含 1% 污染 —— 即本数据实际所处的污染水平

须区分两处变化：A→B 改变的是参数网格，B→C 改变的才是污染。

用法：python 13_recovery_blindspot.py [--reps 200]
"""
import argparse
import numpy as np, pandas as pd
from pathlib import Path
from swald import fit_swald

OUT = Path("../results/tables/table5_recovery.csv")
N_TRIALS = 175
CONTAM_LO, CONTAM_HI = 0.150, 0.250

# 窄网格：论文 2.1 节所述、老化 DDM 文献常见范围
NARROW = dict(v=(1.2, 3.0), a=(0.8, 1.8), ter=(0.20, 0.45))
# 扩展网格：覆盖本数据的真实拟合值（150 ms 下 T_er 低至 0.067、a 高至约 2.4）
EXTEND = dict(v=(1.2, 6.0), a=(0.8, 3.0), ter=(0.05, 0.45))


def block(grid, p_contam, reps, rng):
    true, est = [], []
    for _ in range(reps):
        v = rng.uniform(*grid["v"]); a = rng.uniform(*grid["a"]); ter = rng.uniform(*grid["ter"])
        t = ter + rng.wald(a / v, a ** 2, size=N_TRIALS)
        m = rng.random(N_TRIALS) < p_contam
        if m.sum():
            t[m] = rng.uniform(CONTAM_LO, CONTAM_HI, int(m.sum()))
        r = fit_swald(t, seed=int(rng.integers(1 << 30)))
        if not np.isfinite(r["v"]):
            continue
        true.append((v, a, ter)); est.append((r["v"], r["a"], r["t_er"]))
    T, E = np.array(true), np.array(est)
    out = {}
    for k, nm in enumerate(["v", "a", "ter"]):
        out[f"{nm} 的 r"] = round(float(np.corrcoef(T[:, k], E[:, k])[0, 1]), 2)
    out["T_er 平均相对偏差"] = f"{float(np.mean((E[:,2]-T[:,2])/T[:,2])*100):+.1f}%"
    return out


def main(reps, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for name, grid, p in [("A　窄网格 · 干净（通行做法）", NARROW, 0.0),
                          ("B　扩展网格 · 干净", EXTEND, 0.0),
                          ("C　扩展网格 · 含 1% 污染", EXTEND, 0.01)]:
        r = block(grid, p, reps, rng)
        r["区块"] = name
        r["准入结论"] = "可识别" if r["ter 的 r"] >= 0.5 else "不可识别"
        rows.append(r)
        print(f"  {name:26} v r={r['v 的 r']:.2f}  a r={r['a 的 r']:.2f}  "
              f"T_er r={r['ter 的 r']:.2f}  偏差={r['T_er 平均相对偏差']:>7}  {r['准入结论']}")

    df = pd.DataFrame(rows)[["区块", "v 的 r", "a 的 r", "ter 的 r",
                             "T_er 平均相对偏差", "准入结论"]]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\n已保存 → {OUT}")
    b, c = rows[1]["ter 的 r"], rows[2]["ter 的 r"]
    print(f"\n判据：污染单独的效应是 B→C 的 {b:.2f} → {c:.2f}"
          f"（网格不变、估计器不变、代码不变，仅注入 1% 的实际污染）。")
    print(f"若与通行做法整体相比，准入结论由「{rows[0]['准入结论']}」"
          f"（r = {rows[0]['ter 的 r']:.2f}）翻转为「{rows[2]['准入结论']}」（r = {c:.2f}）。")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260821)
    a = ap.parse_args(); main(a.reps, a.seed)
