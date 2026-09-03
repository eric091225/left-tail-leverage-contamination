"""
Shifted Wald (SWald) 模型与最大似然参数估计。

SWald 是漂移扩散模型在单吸收边界下的解析退化形式（Anders et al., 2016）。
其反应时（单位：秒）服从三参数逆高斯分布，密度（论文式 (2.1)，扩散系数 sigma=1）：

    f(t | v, a, T_er) = a / sqrt(2*pi*(t-T_er)^3)
                        * exp( -(a - v*(t-T_er))^2 / (2*(t-T_er)) ),  t > T_er

参数物理含义：
    v    : 漂移率（证据积累速率，反映认知效率）
    a    : 边界间距（决策策略 / 阈值）
    T_er : 非决策时间（感觉传导 + 运动执行）

派生特征：
    RT_cog = RT_mean - T_er   （扣除运动成分后的认知耗时）

注意：反应时统一以“秒”为单位拟合（data_loader 会把毫秒转换为秒），
这样 v, a 落在文献常见量级（v~0.5-5，a~0.5-2，T_er~0.1-0.5 s）。
"""

import numpy as np
from scipy.optimize import minimize

_LOG_2PI = np.log(2.0 * np.pi)
_EPS = 1e-6  # (t - T_er) 的数值下界，避免 log 下溢


def swald_logpdf(t, v, a, t_er):
    """SWald 对数密度（向量化）。t 为反应时数组（秒）。"""
    s = np.maximum(t - t_er, _EPS)
    return (np.log(a)
            - 0.5 * _LOG_2PI
            - 1.5 * np.log(s)
            - (a - v * s) ** 2 / (2.0 * s))


def swald_nll(params, t):
    """负对数似然。"""
    v, a, t_er = params
    if v <= 0 or a <= 0 or t_er <= 0:
        return 1e12
    return -np.sum(swald_logpdf(t, v, a, t_er))


def fit_swald(rt_seconds, n_restarts=5, seed=0):
    """
    对单个被试的反应时序列做 SWald MLE 拟合。

    采用 L-BFGS-B（带边界约束的拟牛顿法），多组随机初值重启取最优。
    硬约束：v>0, a>0, 0 < T_er < min(RT)（论文 2.1 节）。

    参数
    ----
    rt_seconds : array-like，反应时（秒）
    n_restarts : 随机初值重启次数
    seed       : 随机种子（保证可复现）

    返回
    ----
    dict: {v, a, t_er, rt_cog, nll, converged, n_trials}
    """
    t = np.asarray(rt_seconds, dtype=float)
    t = t[np.isfinite(t)]
    n = len(t)
    min_rt = float(np.min(t))
    rt_mean = float(np.mean(t))

    # T_er 上界严格小于 min(RT)
    t_er_upper = max(min_rt - 1e-3, min_rt * 0.99)
    bounds = [(1e-3, 50.0), (1e-3, 20.0), (1e-4, t_er_upper)]

    rng = np.random.default_rng(seed)
    # 文献推荐启发式初值 + 随机重启初值
    inits = [np.array([1.0, 1.0, 0.5 * min_rt])]
    for _ in range(n_restarts):
        inits.append(np.array([
            rng.uniform(0.5, 2.0),
            rng.uniform(0.5, 2.0),
            rng.uniform(0.2, 0.8) * min_rt,
        ]))

    best = None
    any_success = False
    for x0 in inits:
        # 保证初值在边界内
        x0 = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
        try:
            res = minimize(swald_nll, x0, args=(t,), method="L-BFGS-B",
                           bounds=bounds,
                           options={"maxiter": 500, "ftol": 1e-9})
        except Exception:
            continue
        any_success = any_success or res.success
        if np.isfinite(res.fun) and (best is None or res.fun < best.fun):
            best = res

    if best is None:
        return {"v": np.nan, "a": np.nan, "t_er": np.nan, "rt_cog": np.nan,
                "nll": np.nan, "converged": False, "n_trials": n}

    v, a, t_er = best.x
    return {
        "v": float(v),
        "a": float(a),
        "t_er": float(t_er),
        "rt_cog": float(rt_mean - t_er),
        "nll": float(best.fun),
        "converged": bool(any_success),
        "n_trials": n,
    }
