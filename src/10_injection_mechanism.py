#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10 受控注入：左尾杠杆机制的判别（对应论文 3.2 节，表3.1、图3.1）

要判别的是两个相互竞争、且给出**可区分预测**的机制：

  H1（约束上界假说）：优化器把 T_er 硬约束在 min(RT) 以下；污染观测压低 min(RT)，
      T_er 的可估上界随之降低，故 T_er 被低估。
      → 预测：污染升高时，**触及上界的比例上升**。

  H2（左尾似然杠杆假说）：SWald 密度在支撑边界处急剧趋零，为容纳一个落在边界附近
      的极快观测，优化器必须把 T_er 推到远低于该观测之处，并由 v 与 a 沿补偿脊
      同向调整以维持分布形状。
      → 预测：**上界触及率始终为零、T_er 反而远离上界**，且 v 与 a 随污染系统变化。

做法：以干净真参数生成序列，按给定比例注入抽自 Uniform(150, 250) ms 的污染观测，
再用与主实验完全相同的 fit_swald() 还原，统计上述各量。

用法：
    python 10_injection_mechanism.py              # 完整 400 次重复
    python 10_injection_mechanism.py --reps 20    # 快速校准
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from swald import fit_swald

OUT = Path("../results/tables/table3_injection.csv")

# 真参数：取论文 3.2 节所述设定（落在老化 DDM 文献常见范围内）
TRUE_V, TRUE_A, TRUE_TER = 3.0, 1.2, 0.30
N_TRIALS = 175                      # 数据集中位试次量
LEVELS = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.08)
CONTAM_LO, CONTAM_HI = 0.150, 0.250  # 污染观测的抽样区间（秒）

# fit_swald 的箱约束：T_er ∈ (1e-4, min_rt - 1e-3)。判定「触及」用相对容差。
LOWER_BOUND = 1e-4
TOL = 1e-3                           # 距边界 1 ms 以内即视为触及


def sample_swald(v, a, t_er, n, rng):
    """SWald 抽样：首次通过时间服从逆高斯 IG(mu=a/v, lambda=a^2)，再平移 T_er。"""
    return t_er + rng.wald(a / v, a ** 2, size=n)


def one_run(p_contam, rng):
    """生成一条含污染的序列并拟合，返回该次重复的各项统计。"""
    t = sample_swald(TRUE_V, TRUE_A, TRUE_TER, N_TRIALS, rng)
    # 逐试次独立的伯努利注入：故低污染率下部分重复实际未注入任何试次
    mask = rng.random(N_TRIALS) < p_contam
    n_inj = int(mask.sum())
    if n_inj:
        t[mask] = rng.uniform(CONTAM_LO, CONTAM_HI, n_inj)

    res = fit_swald(t, seed=int(rng.integers(1 << 30)))
    min_rt = float(np.min(t))
    upper = max(min_rt - 1e-3, min_rt * 0.99)   # 与 fit_swald 内部一致
    ter = res["t_er"]
    return {
        "t_er": ter,
        "dist_upper_ms": (upper - ter) * 1000.0,
        "hit_upper": ter >= upper - TOL,
        "hit_lower": ter <= LOWER_BOUND + TOL,
        "a": res["a"],
        "v": res["v"],
        "n_injected": n_inj,
    }


def main(reps, seed):
    rng = np.random.default_rng(seed)
    rows = []
    for p in LEVELS:
        recs = [one_run(p, rng) for _ in range(reps)]
        d = pd.DataFrame(recs)
        rows.append({
            "注入比例": f"{p*100:g}%" if p else "0（干净）",
            "T_er 估计": round(d.t_er.mean(), 4),
            "距上界距离_ms": round(d.dist_upper_ms.mean(), 1),
            "触及上界": f"{d.hit_upper.mean()*100:.1f}%",
            "触及优化下界": f"{d.hit_lower.mean()*100:.1f}%",
            "a 估计": round(d.a.mean(), 3),
            "v 估计": round(d.v.mean(), 3),
            "实际注入为零的重复占比": f"{(d.n_injected == 0).mean()*100:.0f}%",
        })
        r = rows[-1]
        print(f"  {r['注入比例']:>10}  T_er={r['T_er 估计']:.4f}  "
              f"距上界={r['距上界距离_ms']:6.1f}ms  触上界={r['触及上界']:>6}  "
              f"触下界={r['触及优化下界']:>6}  a={r['a 估计']:.3f}  v={r['v 估计']:.3f}")

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"\n已保存 → {OUT}")

    hits = [float(r["触及上界"].rstrip("%")) for r in rows]
    print("\n判据：")
    print(f"  H1 预测触及上界率随污染上升；实测各水平为 {hits} "
          f"→ {'H1 被否证' if max(hits) == 0 else '★ H1 未被否证，请复核'}")
    lo = [float(r["触及优化下界"].rstrip("%")) for r in rows]
    print(f"  H2 预测 T_er 被推向下界；实测触及下界率 {lo} "
          f"→ {'与 H2 一致' if lo[-1] > lo[0] else '★ 与 H2 不一致'}")
    d0, d1 = rows[0]["距上界距离_ms"], max(r["距上界距离_ms"] for r in rows)
    print(f"  H1 预测 T_er 趋近上界；实测距离由 {d0:.1f} ms 增至 {d1:.1f} ms "
          f"→ {'与 H1 相反' if d1 > d0 else '★ 与 H1 一致，请复核'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260821)
    a = ap.parse_args()
    print(f"受控注入：真参数 v={TRUE_V} a={TRUE_A} T_er={TRUE_TER}，"
          f"{N_TRIALS} 试次，每水平 {a.reps} 次重复")
    main(a.reps, a.seed)
