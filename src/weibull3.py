"""
三参数 Weibull 模型与最大似然参数估计（跨模型验证用）。

密度（t > gamma）：

    f(t | gamma, lam, k) = (k/lam) * ((t-gamma)/lam)^(k-1)
                           * exp( -((t-gamma)/lam)^k )

参数：
    gamma : 位置参数（分布支撑的下界）—— 与 SWald 的 T_er 同构
    lam   : 尺度参数
    k     : 形状参数

选它做跨模型验证的理由，不只是「另一个分布」，而是它在支撑下界处的
密度行为**由 k 单独控制**，因而可以把左尾杠杆机制开关掉再打开：

    k > 1 : f(t) -> 0      （与 SWald 同构：密度趋零，左尾杠杆应当出现）
    k = 1 : f(t) -> 1/lam  （移位指数分布：密度有限，杠杆应当消失）
    k < 1 : f(t) -> +inf   （似然无界：位置参数应被推向【上】界，方向相反）

这使 H2 的前提条件「密度在支撑下界附近急剧趋零」成为一个可以被
直接检验的开关，而不是一句只能在单个模型上观察到的描述。
"""

import numpy as np
from scipy.optimize import minimize

_EPS = 1e-6


def weibull3_logpdf(t, gamma, lam, k):
    """三参数 Weibull 对数密度（向量化）。"""
    s = np.maximum(t - gamma, _EPS)
    z = s / lam
    return np.log(k) - np.log(lam) + (k - 1.0) * np.log(z) - z ** k


def weibull3_nll(params, t):
    gamma, lam, k = params
    if lam <= 0 or k <= 0 or gamma <= 0:
        return 1e12
    val = -np.sum(weibull3_logpdf(t, gamma, lam, k))
    return val if np.isfinite(val) else 1e12


def sample_weibull3(gamma, lam, k, n, rng):
    """t = gamma + lam * W(k)，W 为标准 Weibull。"""
    return gamma + lam * rng.weibull(k, size=n)


def fit_weibull3(t, n_restarts=5, seed=0):
    """
    三参数 Weibull MLE。箱约束与 fit_swald 完全同构：
    位置参数被约束在 (1e-4, min(t) - 1e-3)，其余参数为正。
    """
    t = np.asarray(t, dtype=float)
    t = t[np.isfinite(t)]
    n = len(t)
    min_t = float(np.min(t))
    upper = max(min_t - 1e-3, min_t * 0.99)
    bounds = [(1e-4, upper), (1e-3, 20.0), (0.1, 20.0)]

    rng = np.random.default_rng(seed)
    inits = [np.array([0.5 * min_t, 0.5, 1.5])]
    for _ in range(n_restarts):
        inits.append(np.array([rng.uniform(0.2, 0.8) * min_t,
                               rng.uniform(0.2, 1.5),
                               rng.uniform(0.5, 3.0)]))
    best = None
    for x0 in inits:
        x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
        try:
            res = minimize(weibull3_nll, x0, args=(t,), method="L-BFGS-B",
                           bounds=bounds, options={"maxiter": 500, "ftol": 1e-9})
        except Exception:
            continue
        if np.isfinite(res.fun) and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        return {"gamma": np.nan, "lam": np.nan, "k": np.nan,
                "nll": np.nan, "n_trials": n, "upper": upper}
    g, l, kk = best.x
    return {"gamma": float(g), "lam": float(l), "k": float(kk),
            "nll": float(best.fun), "n_trials": n, "upper": upper}
