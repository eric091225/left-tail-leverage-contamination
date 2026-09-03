"""
01 预处理：筛任务、剔异常、转秒、构标签、存中间结果。

修正点（相对初版）：
  - 列名全部抽到顶部 COLUMN CONFIG，拿到真实数据先 `head` 一下，改这里一处即可；
  - 反应时统一转成「秒」再下传（SWald 拟合、参数恢复、论文里 T_er≈0.27s 都按秒）；
  - 显式保存被试级 RT 序列与标签到 pkl，供 02/03/06 独立调用。
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# ============================================================
# COLUMN CONFIG —— 拿到真实数据后先运行：head data/ratcliff_vanunu_2022.csv
# 然后把下面 7 个值改成真实列名 / 取值。改完这里，全流程都对。
# ============================================================
# 以下取值已对照 OSF 原始文件 Data.csv 核实（列：Subject, Age, Task, RT,
# Choice, Brightness, Accuracy；Task==1 为单选项任务，9,996 个观测；
# Age==1 为年轻组 30 人、Age==2 为老年组 29 人）。
COL_SUBJECT   = "Subject"       # 被试 ID 列
COL_RT        = "RT"            # 反应时列
RT_UNIT       = "ms"            # 原始反应时单位："ms" 或 "s"
COL_AGE       = "Age"           # 年龄组列
VAL_OLD       = 2               # 年龄组列里「老年」对应的取值（阳性类=1）
COL_CONDITION = "Task"          # 任务条件列；若数据里没有这一列，设为 None 跳过筛选
VAL_ONECHOICE = 1               # 单选项任务对应的取值
# ============================================================

DATA_IN  = Path("../data/ratcliff_vanunu_2022.csv")
DATA_OUT = Path("../data/ratcliff_vanunu_2022_clean.csv")
PKL_OUT  = Path("../data/rt_data.pkl")

# RT 区间（秒）：150ms–4000ms，与论文 2.2 节一致
# RT 区间（秒）。
# 【重要·与论文一致】主分析下限为 0.25 s（250 ms），不是常规的 150 ms。
# 原因见论文 2.2 节：150 ms 不足以滤除预期反应（抢跑）。机制是「左尾杠杆」——
# SWald 密度在支撑边界处急剧趋零，为容纳一个极快观测，最大似然必须把 T_er 推到
# 远低于它之处直至顶到优化下界，同时 a 与 v 沿补偿脊被抬高，整个参数向量被推离真值。
# 注意：不是「污染压低了 T_er 的可估上界」——该解释（论文中的 H1）已被受控注入
# 实验否证：0–8% 全部污染水平下触及上界的比例恒为 0.0%，且 T_er 距上界反而更远。
# 250 ms 是消除该反转所需的最小干预（本数据上仅剔除 0.59% 试次）。
# 做敏感性分析时改这里即可：0.15 / 0.25 / 0.30，或把 TRIM_FASTEST_PCT 设为 5.0。
RT_MIN_S, RT_MAX_S = 0.25, 4.0

# 个体化剔除：剔掉每位被试最快的百分之多少（0 表示不启用，走上面的固定阈值）
TRIM_FASTEST_PCT = 0.0


def main():
    df = pd.read_csv(DATA_IN)
    print("原始列名：", list(df.columns))   # 方便你核对 COLUMN CONFIG 是否填对

    # 1) 只保留单选项任务（若有该列）
    if COL_CONDITION is not None and COL_CONDITION in df.columns:
        df = df[df[COL_CONDITION] == VAL_ONECHOICE].copy()
    else:
        print(f"[提示] 未按条件筛选（COL_CONDITION={COL_CONDITION}）。"
              f"若数据含多种任务，请设置正确的条件列。")

    # 2) 反应时 → 秒
    rt_s = df[COL_RT].astype(float)
    if RT_UNIT == "ms":
        rt_s = rt_s / 1000.0
    elif RT_UNIT != "s":
        raise ValueError("RT_UNIT 必须是 'ms' 或 's'")
    df["rt_sec"] = rt_s

    # 3) 剔除极短/极长反应（按秒）
    before = len(df)
    if TRIM_FASTEST_PCT > 0:
        # 个体化：每位被试剔掉最快的 TRIM_FASTEST_PCT%
        keep = pd.Series(False, index=df.index)
        for s, g in df.groupby(COL_SUBJECT):
            thr = g["rt_sec"].quantile(TRIM_FASTEST_PCT / 100.0)
            keep.loc[g.index] = (g["rt_sec"] > thr) & (g["rt_sec"] <= RT_MAX_S)
        df = df[keep].copy()
        rule = f"个体化剔最快{TRIM_FASTEST_PCT}%"
    else:
        df = df[(df["rt_sec"] >= RT_MIN_S) & (df["rt_sec"] <= RT_MAX_S)].copy()
        rule = f"{RT_MIN_S*1000:.0f}–{RT_MAX_S*1000:.0f} ms"
    dropped = before - len(df)
    print(f"剔除规则：{rule}")
    print(f"剔除试次：{dropped} / {before}  ({100*dropped/before:.2f}%)")
    # 分组剔除比例：极快反应是否集中在老年组（论文 2.2 节）
    if COL_AGE in df.columns:
        for grp in sorted(set(pd.read_csv(DATA_IN)[COL_AGE].dropna())):
            tot = (pd.read_csv(DATA_IN)[COL_AGE] == grp).sum()
            print(f"  组 {grp}: 保留 {(df[COL_AGE]==grp).sum()} / 原始 {tot}")

    # 4) 每被试试次数分布（论文 2.1 节：核对“中位数约175、范围45–256”）
    n_trials = df.groupby(COL_SUBJECT)["rt_sec"].count()
    print("\n每被试试次数分布：")
    print(n_trials.describe())

    # 5) 标签：老年=1，年轻=0
    df["label"] = (df[COL_AGE] == VAL_OLD).astype(int)

    # 6) 保存清洗后的长表
    DATA_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_OUT, index=False)

    # 7) 被试级 RT 序列（秒，保持试次顺序）+ 标签 → pkl
    rt_data, labels = {}, {}
    for subj, g in df.groupby(COL_SUBJECT):
        rt_data[subj] = g["rt_sec"].values        # 已是秒、按行序
        labels[subj]  = int(g["label"].iloc[0])
    with open(PKL_OUT, "wb") as f:
        pickle.dump({"rt_data": rt_data, "labels": labels}, f)

    n_old = sum(labels.values()); n_young = len(labels) - n_old
    print(f"\n被试数 N={len(labels)}（年轻 {n_young} / 老年 {n_old}）")
    print(f"已保存：{DATA_OUT}  和  {PKL_OUT}")


if __name__ == "__main__":
    main()
