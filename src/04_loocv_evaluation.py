"""
04 LOOCV 评估：三组特征 × 三种分类器，留一交叉验证。

修正点：
  - 补齐 SVM-RBF 与随机森林作稳健性对照，逻辑回归为预先指定的主分类器；
  - RFE 仍在每折训练集内部做（防泄漏，这点初版就对）；
  - 三组特征对齐到同一批被试（剔除 SWald 不收敛/NaN 的被试，保持三组可比并报告数量）；
  - 输出 9 组 out-of-fold 预测到 csv，供 05/06 复用；打印表2。

组：SWald 理论组(TG, 4维) / 原始RT基线(RT, 3维) / RFE数据驱动(DD, 从18维选4维)
分类器：lr(主) / svm(RBF) / rf
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

SEED = 42
RFE_SELECTOR_C = 1.0   # RFE 内部逻辑回归选择器的正则强度，实验前固定，不随结果调整
FEAT_DIR = Path("../results/features")
PRED_OUT = Path("../results/predictions_all.csv")
TABLE2_OUT = Path("../results/tables/table2.csv")


def make_clf(name):
    if name == "lr":
        return LogisticRegression(max_iter=1000)
    if name == "svm":
        return SVC(kernel="rbf", probability=True, random_state=SEED)
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, random_state=SEED)
    raise ValueError(name)


def loocv_fixed(X, y, clf_name):
    """固定特征组（SWald / RT）的 LOOCV，返回 out-of-fold 概率。"""
    n = len(y)
    oof = np.zeros(n)
    for i in range(n):
        tr = np.delete(np.arange(n), i)
        sc = StandardScaler().fit(X[tr])
        clf = make_clf(clf_name).fit(sc.transform(X[tr]), y[tr])
        oof[i] = clf.predict_proba(sc.transform(X[i:i+1]))[0, 1]
    return oof


def loocv_rfe(Xpool, y, clf_name, k=4):
    """数据驱动组：每折训练集内部 RFE 选 k 维（LR 为选择器），再分类。防泄漏。"""
    n = len(y)
    oof = np.zeros(n)
    for i in range(n):
        tr = np.delete(np.arange(n), i)
        sc = StandardScaler().fit(Xpool[tr])
        Xtr, Xte = sc.transform(Xpool[tr]), sc.transform(Xpool[i:i+1])
        # 【已钉死】RFE 选择器的正则强度必须显式给出，不可依赖 sklearn 默认值。
        # 实测该参数对 DD 组结果的影响极大：C=0.1 时 DD=0.735、C=1.0 时 0.646、C=10 时 0.639，
        # 且 C=0.1 会使 DD 组由 0.646 升至 0.735。本研究在实验前固定 C=1.0（见论文 6.1 节闸三与第七章）。
        sel = RFE(LogisticRegression(max_iter=1000, C=RFE_SELECTOR_C),
                  n_features_to_select=k).fit(Xtr, y[tr])
        clf = make_clf(clf_name).fit(sel.transform(Xtr), y[tr])
        oof[i] = clf.predict_proba(sel.transform(Xte))[0, 1]
    return oof


def main():
    sw = pd.read_csv(FEAT_DIR / "swald_features.csv", index_col=0)
    rt = pd.read_csv(FEAT_DIR / "rt_baseline.csv", index_col=0)
    pool = pd.read_csv(FEAT_DIR / "feature_pool_18d.csv", index_col=0)

    # 对齐被试：剔除 SWald 不收敛/NaN，保持三组同一批被试
    good = sw[["v", "a", "ter", "rt_cog"]].notna().all(axis=1)
    if "converged" in sw.columns:
        good &= sw["converged"].astype(bool)
    subjects = sw.index[good]
    dropped = (~good).sum()
    if dropped:
        print(f"[提示] 剔除 SWald 不收敛/NaN 被试 {dropped} 位，"
              f"三组均在剩余 {len(subjects)} 位上比较。")
    sw, rt, pool = sw.loc[subjects], rt.loc[subjects], pool.loc[subjects]

    y = sw["label"].astype(int).values
    X_tg = sw[["v", "a", "ter", "rt_cog"]].values
    X_rt = rt[["rt_mean", "rt_std", "rt_cv"]].values
    X_pool = pool.drop(columns=["label"]).values
    print(f"参与评估 N={len(y)}（年轻 {sum(y==0)} / 老年 {sum(y==1)}），18维候选池 {X_pool.shape[1]} 维")

    groups = {"TG": ("fixed", X_tg), "RT": ("fixed", X_rt), "DD": ("rfe", X_pool)}
    preds = {"true_label": y}
    table = []
    for cname in ["lr", "svm", "rf"]:
        row = {"classifier": cname}
        for gkey, (kind, X) in groups.items():
            oof = loocv_fixed(X, y, cname) if kind == "fixed" else loocv_rfe(X, y, cname)
            preds[f"{gkey}_{cname}"] = oof
            row[gkey] = round(roc_auc_score(y, oof), 4)
        table.append(row)
        print(f"  {cname:4s}  TG={row['TG']:.4f}  RT={row['RT']:.4f}  DD={row['DD']:.4f}")

    PRED_OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(preds).to_csv(PRED_OUT, index=False)
    TABLE2_OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(table).to_csv(TABLE2_OUT, index=False)
    print(f"\n已保存预测 → {PRED_OUT}")
    print(f"已保存表2（三组×三分类器 AUC）→ {TABLE2_OUT}")
    print("注：主结论以 lr 行为准；svm/rf 为稳健性。DeLong p 与 Bootstrap CI 见 05。")


if __name__ == "__main__":
    main()
