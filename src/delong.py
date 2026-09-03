"""
DeLong 检验：比较两条配对 ROC 曲线的 AUC 是否有显著差异。

实现基于 Sun & Xu (2014) 的快速 DeLong 算法，并参考
DeLong et al. (1988) 的非参数协方差估计。两组预测来自同一批被试
（配对设计），因此使用配对版本的 DeLong 检验。

对外主要接口：
    delong_roc_test(y_true, score_a, score_b, alternative="greater")
        返回 (auc_a, auc_b, z, p_value)

约定：y_true 为 0/1，1 表示阳性类（本研究中为“老年组”）。
alternative="greater" 对应单侧备择假设 H_A: AUC_a > AUC_b（论文第七章）。
"""

import numpy as np
from scipy import stats


def _compute_midrank(x):
    """计算中位秩（处理并列值），返回 1-based 中位秩。"""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1.0
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T
    return T2


def _fast_delong(predictions_sorted_transposed, label_1_count):
    """
    快速 DeLong 协方差估计。

    参数
    ----
    predictions_sorted_transposed : np.ndarray, shape [k, n]
        k 个预测器在 n 个样本上的得分；样本已重排为“阳性在前”。
    label_1_count : int
        阳性样本数 m。

    返回
    ----
    aucs : np.ndarray, shape [k]
    delongcov : np.ndarray, shape [k, k]
    """
    m = label_1_count
    n_total = predictions_sorted_transposed.shape[1]
    n = n_total - m
    k = predictions_sorted_transposed.shape[0]

    positive = predictions_sorted_transposed[:, :m]
    negative = predictions_sorted_transposed[:, m:]

    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, n_total], dtype=float)
    for r in range(k):
        tx[r, :] = _compute_midrank(positive[r, :])
        ty[r, :] = _compute_midrank(negative[r, :])
        tz[r, :] = _compute_midrank(predictions_sorted_transposed[r, :])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n
    v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    # np.cov 对 [k, m] 返回 [k, k]，需要 k>=2；k=1 时单独处理
    if k == 1:
        sx = np.array([[np.var(v01[0], ddof=1)]])
        sy = np.array([[np.var(v10[0], ddof=1)]])
    else:
        sx = np.cov(v01)
        sy = np.cov(v10)
    delongcov = sx / m + sy / n
    return aucs, delongcov


def _prepare(y_true, scores):
    """把样本重排为阳性在前，返回重排后的得分矩阵与阳性数。"""
    y_true = np.asarray(y_true).astype(int)
    order = np.argsort(-y_true, kind="mergesort")  # 1 在前，稳定排序
    label_1_count = int(np.sum(y_true == 1))
    scores = np.asarray(scores, dtype=float)
    if scores.ndim == 1:
        scores = scores[np.newaxis, :]
    return scores[:, order], label_1_count


def delong_auc_var(y_true, score):
    """单条 ROC 的 AUC 及其 DeLong 方差。"""
    sorted_scores, m = _prepare(y_true, score)
    aucs, cov = _fast_delong(sorted_scores, m)
    return float(aucs[0]), float(cov[0, 0])


def delong_roc_test(y_true, score_a, score_b, alternative="greater"):
    """
    配对 DeLong 检验：检验 AUC_a 与 AUC_b 的差异。

    alternative:
        "greater"   -> H_A: AUC_a > AUC_b（单侧，论文默认）
        "less"      -> H_A: AUC_a < AUC_b
        "two-sided" -> H_A: AUC_a != AUC_b

    返回 (auc_a, auc_b, z, p_value)
    """
    scores = np.vstack([np.asarray(score_a, float), np.asarray(score_b, float)])
    sorted_scores, m = _prepare(y_true, scores)
    aucs, cov = _fast_delong(sorted_scores, m)
    auc_a, auc_b = float(aucs[0]), float(aucs[1])

    var_diff = cov[0, 0] + cov[1, 1] - 2.0 * cov[0, 1]
    if var_diff <= 0:
        # 两预测器几乎完全一致，差异方差退化
        z = 0.0
        if alternative == "two-sided":
            p = 1.0
        else:
            p = 0.5
        return auc_a, auc_b, z, p

    z = (auc_a - auc_b) / np.sqrt(var_diff)
    if alternative == "greater":
        p = stats.norm.sf(z)
    elif alternative == "less":
        p = stats.norm.cdf(z)
    elif alternative == "two-sided":
        p = 2.0 * stats.norm.sf(abs(z))
    else:
        raise ValueError("alternative 必须是 greater / less / two-sided")
    return auc_a, auc_b, float(z), float(p)
