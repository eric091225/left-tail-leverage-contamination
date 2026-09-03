"""
09 预期反应剔除的敏感性分析（对应论文 2.2 节，表2.1 及其注）

背景：在允许抢跑的任务范式中，预期反应会产生极短 RT，使 SWald 的最大似然拟合给出
与经典 DDM 文献相反的 T_er 年龄效应，并给衍生特征 RT_cog 注入伪信号。
【机制须注意】原因不是「污染压低了 T_er 的可估上界」——该解释（论文中的 H1）已被
受控注入实验否证（0–8% 全部污染水平下触及上界的比例恒为 0.0%）。正确机制是左尾似然
杠杆：密度在支撑边界处趋零，极快观测迫使优化器把 T_er 推到远低于它之处直至顶到优化
下界，同时 a 与 v 沿补偿脊被抬高。详见论文 2.2 节。

本脚本以四种剔除条件重跑「预处理 → SWald 拟合 → 特征 → LOOCV」，检验该假设：
  150 ms（原研究沿用） / 250 ms（本研究主分析） / 300 ms / 个体化剔最快 5%

判据：
  ① 剂量效应   —— 提高阈值后 T_er 的年龄效应是否随之消失
  ② 生理合理性 —— T_er 绝对值是否回到文献常见的 200–400 ms 区间
  ③ 选择性     —— 影响应只打在 T_er 上（v、a 不受 min(RT) 约束，应基本不动）

注意：本脚本直接从原始 CSV 出发，不依赖 01 的输出，以便独立复现敏感性分析。
列名配置与 01_preprocess.py 保持一致。
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

from swald import fit_swald

# ---- 列名配置：直接从 01_preprocess.py 导入，避免两处副本失步 ----
# （原先此处是一份手抄副本，改了 01 不会同步到这里，已改为单一事实来源）
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("_prep", str(Path(__file__).with_name("01_preprocess.py")))
_prep = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_prep)
COL_SUBJECT, COL_RT, RT_UNIT = _prep.COL_SUBJECT, _prep.COL_RT, _prep.RT_UNIT
COL_AGE, VAL_OLD = _prep.COL_AGE, _prep.VAL_OLD
COL_CONDITION, VAL_ONECHOICE = _prep.COL_CONDITION, _prep.VAL_ONECHOICE

DATA_IN = Path("../data/ratcliff_vanunu_2022.csv")
OUT_T7  = Path("../results/tables/table7_sensitivity.csv")
OUT_T8  = Path("../results/tables/table8_param_effects.csv")
RT_MAX_S = 4.0

CONDITIONS = [("150 ms（原研究沿用）", "fixed", 0.150),
              ("250 ms（本研究主分析）", "fixed", 0.250),
              ("300 ms", "fixed", 0.300),
              ("个体化（剔最快5%）", "pct", 5.0)]

F18 = ["mean","median","mode","sd","iqr","cv","skew","kurt",
       "p10","p25","p75","p90","min","max","range","first_rt","roll_sd","diff_mean"]


def cohen_d(a, b):
    """d = (mean(a) - mean(b)) / pooled_sd；此处 a=老年、b=年轻。"""
    na, nb = len(a), len(b)
    s = np.sqrt(((na-1)*a.var(ddof=1) + (nb-1)*b.var(ddof=1)) / (na+nb-2))
    return (a.mean() - b.mean()) / s


def feats18(rt):
    f = {}
    f["mean"]=np.mean(rt); f["median"]=np.median(rt)
    f["mode"]=float(stats.mode(rt, keepdims=True)[0][0])
    f["sd"]=np.std(rt); f["iqr"]=np.percentile(rt,75)-np.percentile(rt,25)
    f["cv"]=np.std(rt)/np.mean(rt); f["skew"]=stats.skew(rt); f["kurt"]=stats.kurtosis(rt)
    for q in (10,25,75,90): f[f"p{q}"]=np.percentile(rt,q)
    f["min"]=np.min(rt); f["max"]=np.max(rt); f["range"]=f["max"]-f["min"]
    f["first_rt"]=rt[0] if len(rt) else 0.0
    f["roll_sd"]=np.mean([np.std(rt[i:i+5]) for i in range(len(rt)-4)]) if len(rt)>=5 else np.std(rt)
    f["diff_mean"]=np.mean(np.abs(np.diff(rt))) if len(rt)>1 else 0.0
    return f


def loocv_lr(X, y, rfe=False, k=4):
    n=len(y); oof=np.zeros(n)
    for i in range(n):
        tr=np.delete(np.arange(n),i)
        sc=StandardScaler().fit(X[tr]); Xtr=sc.transform(X[tr]); Xte=sc.transform(X[i:i+1])
        if rfe:
            sel=RFE(LogisticRegression(max_iter=5000), n_features_to_select=k).fit(Xtr, y[tr])
            Xtr, Xte = sel.transform(Xtr), sel.transform(Xte)
        oof[i]=LogisticRegression(max_iter=5000).fit(Xtr,y[tr]).predict_proba(Xte)[0,1]
    return roc_auc_score(y, oof)


def main():
    raw = pd.read_csv(DATA_IN)
    if COL_CONDITION in raw.columns:
        raw = raw[raw[COL_CONDITION] == VAL_ONECHOICE].copy()
    raw["rt_sec"] = raw[COL_RT].astype(float) / (1000.0 if RT_UNIT=="ms" else 1.0)
    raw["label"]  = (raw[COL_AGE] == VAL_OLD).astype(int)

    t7, t8 = [], []
    for name, mode, par in CONDITIONS:
        if mode == "fixed":
            keep = (raw["rt_sec"] >= par) & (raw["rt_sec"] <= RT_MAX_S)
        else:
            keep = pd.Series(False, index=raw.index)
            for s, g in raw.groupby(COL_SUBJECT):
                thr = g["rt_sec"].quantile(par/100.0)
                keep.loc[g.index] = (g["rt_sec"] > thr) & (g["rt_sec"] <= RT_MAX_S)
        d = raw[keep]
        rm_pct = 100*(1-keep.mean())
        # 分组剔除数（论文报告极快反应是否集中在老年组）
        rm_y = int((~keep & (raw["label"]==0)).sum())
        rm_o = int((~keep & (raw["label"]==1)).sum())

        rows=[]
        for subj, g in d.groupby(COL_SUBJECT):
            rt = g["rt_sec"].values
            fit = fit_swald(rt)
            f18 = feats18(rt)
            rows.append(dict(subj=subj, label=int(g["label"].iloc[0]), n=len(rt),
                             v=fit["v"], a=fit["a"], ter=fit["t_er"],
                             rt_cog=f18["mean"]-fit["t_er"],
                             ok=fit["converged"], **f18))
        F = pd.DataFrame(rows)
        y = F["label"].values

        # 参数年龄效应
        eff={}
        for p in ["v","a","ter","rt_cog"]:
            yo, ol = F[y==0][p].values, F[y==1][p].values
            eff[p] = dict(young=yo.mean(), older=ol.mean(),
                          d=cohen_d(ol, yo), p=stats.ttest_ind(ol, yo).pvalue)
            t8.append({"条件":name,"参数":p,"年轻组":round(yo.mean(),3),"老年组":round(ol.mean(),3),
                       "Cohen_d":round(eff[p]["d"],2),"p":round(eff[p]["p"],4)})

        mins = F.groupby("label")["min"].mean()
        d_min = cohen_d(F[y==1]["min"].values, F[y==0]["min"].values)

        auc_tg = loocv_lr(F[["v","a","ter","rt_cog"]].values, y)
        auc_rt = loocv_lr(F[["mean","sd","cv"]].values, y)
        auc_dd = loocv_lr(F[F18].values, y, rfe=True)

        t7.append({"剔除条件":name, "剔除比例":f"{rm_pct:.2f}%",
                   "剔除数(年轻/老年)":f"{rm_y}/{rm_o}",
                   "T_er老年组":round(eff["ter"]["older"],3),
                   "T_er的d":round(eff["ter"]["d"],2), "p":round(eff["ter"]["p"],4),
                   "min(RT)的d":round(d_min,2),
                   "SWald_TG":round(auc_tg,3), "RT朴素":round(auc_rt,3), "RFE_DD":round(auc_dd,3),
                   "拟合失败":int((~F["ok"]).sum())})
        arrow = "↑" if eff["ter"]["d"]>0 else "↓"
        print(f"{name:<20} 剔除{rm_pct:5.2f}% | T_er 老年{arrow} d={eff['ter']['d']:+.2f} "
              f"p={eff['ter']['p']:.4f} 绝对值={eff['ter']['older']:.3f}s | "
              f"min(RT) d={d_min:+.2f} | TG={auc_tg:.3f} RT={auc_rt:.3f} DD={auc_dd:.3f}")

    OUT_T7.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(t7).to_csv(OUT_T7, index=False)
    pd.DataFrame(t8).to_csv(OUT_T8, index=False)
    print(f"\n已保存 → {OUT_T7}（论文表2.1）")
    print(f"已保存 → {OUT_T8}（论文表2.1 的注）")
    print("\n判读要点：若假设成立，应看到 ①T_er 的 d 随阈值提高而趋零；"
          "②T_er 绝对值回到 0.2–0.4 s。注意：不要据此断言「v、a 不动、影响是选择性的」——受控注入显示污染沿补偿脊同时抬高 a 与 v，整个参数向量被推离（论文 2.2 节）。")


if __name__ == "__main__":
    main()
