"""
07 共线性诊断（回应老师建议 5a：处理 SWald 四特征的多重共线性）。

背景：RT_cog = RT_mean − T_er，所以 T_er 与 RT_cog 天然相关。若把四维特征
一起喂给逻辑回归，共线会让权重方差变大，“看 LR 权重解读各特征贡献”就不稳。

本脚本：
  1) 报告 {v, a, T_er, RT_cog} 的相关矩阵与 VIF（方差膨胀因子）；
  2) 给出结论口径：单特征的判别贡献**以消融实验（表4）为准**，不依赖联合 LR 的权重——
     消融逐个特征单独评估，天然绕开共线性。

诚实细节（答辩可用）：严格线性依赖是 RT_cog = RT_mean − T_er，但 RT_mean 不在这四个
特征里，所以四特征之间**并非**严格线性相关（VIF 不会是无穷），只是 T_er 与 RT_cog 相关较强。
下面用实测 VIF 说话。
"""
import numpy as np
import pandas as pd
from pathlib import Path

FEAT_SW = Path("../results/features/swald_features.csv")
OUT     = Path("../results/tables/collinearity.csv")
COLS    = ["v", "a", "ter", "rt_cog"]


def vif(X):
    """各列 VIF_j = 1/(1-R_j^2)，R_j^2 为用其余列回归该列的判定系数。"""
    from numpy.linalg import lstsq
    n, p = X.shape
    out = []
    for j in range(p):
        y = X[:, j]
        Xo = np.column_stack([np.ones(n), np.delete(X, j, axis=1)])
        beta, *_ = lstsq(Xo, y, rcond=None)
        ss_res = np.sum((y - Xo @ beta) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        out.append(1.0 / (1.0 - r2) if r2 < 1 else np.inf)
    return out


def main():
    sw = pd.read_csv(FEAT_SW, index_col=0)
    good = sw[COLS].notna().all(axis=1)
    if "converged" in sw.columns:
        good &= sw["converged"].astype(bool)
    X = sw.loc[good, COLS].values

    print("相关矩阵（v, a, ter, rt_cog）：")
    corr = np.corrcoef(X, rowvar=False)
    print(pd.DataFrame(corr, index=COLS, columns=COLS).round(3).to_string())

    vifs = vif(X)
    print("\nVIF（>5 中度共线，>10 强共线）：")
    for c, vf in zip(COLS, vifs):
        print(f"  {c:8s} VIF = {vf:.2f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"feature": COLS, "VIF": [round(v, 2) for v in vifs]}).to_csv(OUT, index=False)
    print(f"\n已保存 → {OUT}")
    print("结论口径（本版论文未引用本脚本任何结果，详见 README）：")
    print("  · T_er / RT_cog 若 VIF 偏高，说明二者共线，联合 LR 权重不宜直接解读；")
    print("  · 各特征判别贡献以**消融实验（表4）**为准——逐特征单独评估，不受共线影响；")
    print("  · LR 仍作主分类器用于整体 AUC 比较，只是不把它的系数当特征重要性。")


if __name__ == "__main__":
    main()
