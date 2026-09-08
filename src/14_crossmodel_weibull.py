#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
14 跨模型验证：三参数 Weibull 上的左尾杠杆污染

本文的一般陈述是「凡以最大似然估计分布支撑下界、且密度在该下界附近急剧趋零的
模型，其参数估计对落在左尾的少量污染具有结构性敏感」。该陈述此前只在 SWald
一族上验证过，实验范围小于结论范围。本脚本补上跨模型证据，并把验证做成两级：

  A 级（复现）：在 Weibull(k=2) 上重跑受控注入，检验是否出现与 SWald 相同的
              三个特征——上界触及率恒为零、位置参数被推向下界、其余参数补偿抬升。

  B 级（机制开关）：Weibull 在支撑下界处的密度行为**由形状参数 k 单独控制**：
              k>1 趋零、k=1 有限、k<1 发散。若 H2 成立，失效强度应随 k 单调，
              且在 k<1 时**方向反转**（似然无界，位置参数被推向上界）。
              这一级检验的是机制的前提条件本身，而非仅仅复现现象。

真参数取 gamma=0.30、lam=0.45，使分布均值 0.70 s 与 SWald 仿真完全一致
（同位置参数、同均值、不同分布族），确保差异只能来自分布形态。

用法：
    python 14_crossmodel_weibull.py            # 完整 400 次重复
    python 14_crossmodel_weibull.py --reps 20  # 快速校准，输出不可用于结论
"""
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

from weibull3 import sample_weibull3, fit_weibull3
from weibull3 import weibull3_nll
from scipy.optimize import minimize


def fit_weibull3_fixed_k(t, k_fixed=1.0, n_restarts=5, seed=0):
    """
    把形状参数钉死、只估 (gamma, lambda) 的对照拟合。
    仅用于论文 3.3 节 k = 1.0 一行的判据五——除此之外全文一律联合估计三个参数。
    箱约束与 fit_weibull3 的前两维完全一致，以保证两者可比。
    """
    t = np.asarray(t, dtype=float)
    t = t[np.isfinite(t)]
    min_t = float(np.min(t))
    upper = max(min_t - 1e-3, min_t * 0.99)
    bounds = [(1e-4, upper), (1e-3, 20.0)]
    rng = np.random.default_rng(seed)
    inits = [np.array([0.5 * min_t, 0.5])]
    for _ in range(n_restarts):
        inits.append(np.array([rng.uniform(0.2, 0.8) * min_t, rng.uniform(0.2, 1.5)]))
    best = None
    for x0 in inits:
        x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
        try:
            res = minimize(lambda q, tt: weibull3_nll(np.array([q[0], q[1], k_fixed]), tt),
                           x0, args=(t,), method="L-BFGS-B", bounds=bounds,
                           options={"maxiter": 500, "ftol": 1e-9})
        except Exception:
            continue
        if np.isfinite(res.fun) and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        return {"gamma": np.nan, "upper": upper}
    return {"gamma": float(best.x[0]), "upper": upper}

OUT_A = Path("../results/tables/table_crossmodel_dose.csv")
OUT_B = Path("../results/tables/table_crossmodel_ksweep.csv")

TRUE_GAMMA, TRUE_LAM = 0.30, 0.45
N_TRIALS = 175
LEVELS = (0.0, 0.0025, 0.005, 0.01, 0.02, 0.08)
K_GRID = (0.8, 1.0, 1.5, 2.0, 3.0)
CONTAM_LO, CONTAM_HI = 0.150, 0.250
LOWER_BOUND, TOL = 1e-4, 1e-3


def one_run(p_contam, k, rng):
    t = sample_weibull3(TRUE_GAMMA, TRUE_LAM, k, N_TRIALS, rng)
    mask = rng.random(N_TRIALS) < p_contam
    n_inj = int(mask.sum())
    if n_inj:
        t[mask] = rng.uniform(CONTAM_LO, CONTAM_HI, n_inj)
    r = fit_weibull3(t, seed=int(rng.integers(1 << 30)))
    g, upper = r["gamma"], r["upper"]
    return {"gamma": g, "dist_ms": (upper - g) * 1000.0,
            "hit_upper": g >= upper - TOL, "hit_lower": g <= LOWER_BOUND + TOL,
            "lam": r["lam"], "k": r["k"], "n_inj": n_inj}


def block(levels, k, reps, rng):
    rows = []
    for p in levels:
        d = pd.DataFrame([one_run(p, k, rng) for _ in range(reps)])
        rows.append({"形状 k": k, "注入比例": f"{p*100:g}%" if p else "0（干净）",
                     "gamma 估计": round(d.gamma.mean(), 4),
                     "距上界距离_ms": round(d.dist_ms.mean(), 1),
                     "触及上界": f"{d.hit_upper.mean()*100:.1f}%",
                     "触及优化下界": f"{d.hit_lower.mean()*100:.1f}%",
                     "lam 估计": round(d.lam.mean(), 3),
                     "k 估计": round(d.k.mean(), 3)})
    return rows


def main(reps, seed):
    rng = np.random.default_rng(seed)

    print(f"【A 级 · 复现】Weibull(k=2.0)，真值 gamma={TRUE_GAMMA} lam={TRUE_LAM}，"
          f"{N_TRIALS} 试次，每水平 {reps} 次重复\n")
    A = block(LEVELS, 2.0, reps, rng)
    for r in A:
        print(f"  {r['注入比例']:>10}  gamma={r['gamma 估计']:.4f}  "
              f"距上界={r['距上界距离_ms']:6.1f}ms  触上界={r['触及上界']:>6}  "
              f"触下界={r['触及优化下界']:>6}  lam={r['lam 估计']:.3f}  k={r['k 估计']:.3f}")
    OUT_A.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(A).to_csv(OUT_A, index=False)

    hits = [float(r["触及上界"].rstrip("%")) for r in A]
    lows = [float(r["触及优化下界"].rstrip("%")) for r in A]
    print(f"\n  判据一（H1 是否同样被否证）：触上界率 {hits} → "
          f"{'H1 在 Weibull 上同样被否证' if max(hits)==0 else '★ 未被否证，请复核'}")
    print(f"  判据二（H2 是否同样成立）  ：触下界率 {lows} → "
          f"{'与 H2 一致' if lows[-1] > lows[0] else '★ 与 H2 不一致'}")
    d0, d1 = A[0]["距上界距离_ms"], max(r["距上界距离_ms"] for r in A)
    print(f"  判据三（距离是否同样拉大）  ：{d0:.1f} → {d1:.1f} ms → "
          f"{'与 SWald 同向' if d1 > d0 else '★ 与 SWald 反向'}")

    print(f"\n【B 级 · 机制开关】固定 1% 污染，扫描形状参数 k\n")
    print("  密度在支撑下界处：k<1 发散 / k=1 有限 / k>1 趋零\n")
    B = []
    for k in K_GRID:
        B += block((0.01,), k, reps, rng)
    for r in B:
        beh = "发散" if r["形状 k"] < 1 else ("有限" if r["形状 k"] == 1 else "趋零")
        print(f"  k={r['形状 k']:<4}（密度{beh}）  gamma={r['gamma 估计']:.4f}  "
              f"触上界={r['触及上界']:>6}  触下界={r['触及优化下界']:>6}")
    pd.DataFrame(B).to_csv(OUT_B, index=False)

    print(f"\n已保存 → {OUT_A}\n         {OUT_B}")
    lo_k = [r for r in B if r["形状 k"] < 1][0]
    hi_k = [r for r in B if r["形状 k"] == 3.0][0]
    up_lo = float(lo_k["触及上界"].rstrip("%")); dn_hi = float(hi_k["触及优化下界"].rstrip("%"))
    print("\n判据四（机制的前提条件）：")
    print(f"  k=0.8（密度发散）触上界 {up_lo:.1f}%、k=3.0（密度趋零）触下界 {dn_hi:.1f}% → "
          f"{'方向随密度行为反转，与 H2 的前提一致' if up_lo>0 and dn_hi>0 else '★ 未见反转，请复核'}")
    print("  若两端方向相反，则失效不是某个分布族的特性，而是"
          "「密度在支撑下界处趋零 + 以 MLE 估计该下界」这一组合的性质。")

    # 判据五：k = 1.0 一行为什么不是 100%——开关由「估计出的 k」拨动，不只由真值 k 拨动。
    # 移位指数的位置参数 MLE 理论上恰等于 min(t)，触上界率应为 100%；联合估计下只有约 45%，
    # 因为 k 自由时会漂到 1 以上，密度在下界处由「有限」转为「趋零」，gamma 随即被释放。
    # 把 k 钉死在 1 重跑同一设定即可复原理论值。对应论文 3.3 节 k = 1.0 段。
    row1 = [r for r in B if r["形状 k"] == 1.0][0]
    rng5 = np.random.default_rng(seed)
    hit_fixed = 0
    for _ in range(reps):
        t = sample_weibull3(TRUE_GAMMA, TRUE_LAM, 1.0, N_TRIALS, rng5)
        m = rng5.random(N_TRIALS) < 0.01
        if m.sum():
            t[m] = rng5.uniform(CONTAM_LO, CONTAM_HI, int(m.sum()))
        f = fit_weibull3_fixed_k(t, 1.0, seed=int(rng5.integers(1 << 30)))
        hit_fixed += int(f["gamma"] >= f["upper"] - TOL)
    print("\n判据五（k = 1.0 一行：开关由估计出的 k 拨动）：")
    print(f"  联合估计 (gamma, lambda, k)：触上界 {row1['触及上界']:>6}，"
          f"k 的估计均值 {row1['k 估计']:.2f}（> 1，密度已转为趋零）")
    print(f"  固定 k = 1（真移位指数）    ：触上界 {100*hit_fixed/reps:5.1f}%"
          f"  ← 理论值 100%")
    print("  → " + ("两者相差悬殊，说明 45% 不是数值伪影，而是 k 自由后密度行为已经改变"
                    if 100*hit_fixed/reps > 90 and float(row1["触及上界"].rstrip("%")) < 70
                    else "★ 未见预期差异，请复核"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--seed", type=int, default=20260821)
    a = ap.parse_args()
    main(a.reps, a.seed)
