#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
论文配图生成脚本

本脚本生成**不依赖原始数据**的三张图（另加一张未收入论文的参考图）：

  fig2_mechanism   → 论文图3.1  左尾杠杆污染的机制判别   数值取自论文表3.1
  fig5_ksweep      → 论文图3.2  机制的前提条件可开关     数值取自论文表3.3
  fig3_pre_checks  → 论文图6.1  四道前置检查（结构示意）
  fig4_routes      → 未收入论文，保留作参考             数值取自第七章五路线段

论文图6.2（不可达路径泄漏诊断）需要 OSF 原始数据，由 fig1_leakage_diagnostic.py
独立生成——该图须复现被撤回的那个带宽 DTW 距离矩阵，不宜与本脚本混在一起。

用法：
    python make_figures.py
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

CJK = "PingFang SC"   # Windows 改 "Microsoft YaHei"；Linux 改 "Noto Sans CJK SC"
plt.rcParams.update({
    "font.family": [CJK, "DejaVu Sans"], "axes.unicode_minus": False,
    "font.size": 9, "axes.linewidth": 0.8, "axes.edgecolor": "#444444",
    "figure.dpi": 200, "savefig.dpi": 300, "savefig.bbox": "tight",
})
INK, GREY, LIGHT = "#1a1a1a", "#8a8a8a", "#d9d9d9"
BLUE, RED, AMBER, GREEN = "#3b6ea5", "#c1443f", "#d38b2c", "#4a7a54"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "figures")


def save(fig, name):
    os.makedirs(OUT, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/{name}.{ext}")
    plt.close(fig)
    print(f"  已输出 {OUT}/{name}.png / .pdf")


# ===================== 图3  四道前置检查 =====================
def fig3():
    fig, ax = plt.subplots(figsize=(7.0, 8.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 12.4); ax.axis("off")

    def box(x, y, w, h, text="", fc="#fff", ec=INK, lw=1.0, fs=9, tc=INK):
        ax.add_patch(FancyBboxPatch((x - w/2, y - h/2), w, h,
                                    boxstyle="round,pad=0.06,rounding_size=0.10",
                                    fc=fc, ec=ec, lw=lw, zorder=2))
        if text:
            ax.text(x, y, text, ha="center", va="center", fontsize=fs,
                    color=tc, zorder=3, linespacing=1.5)

    def arrow(x1, y1, x2, y2, color=INK, lw=1.1, ls="-"):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=11, lw=lw, color=color,
                                     linestyle=ls, shrinkA=0, shrinkB=0, zorder=1))

    CX, FX, BW = 3.55, 8.05, 5.05
    ax.text(CX, 12.05, "四道前置检查", ha="center", va="center",
            fontsize=13, fontweight="bold", color=INK)
    ax.text(CX, 11.60, "全部通过后，特征间的比较才被视为有意义",
            ha="center", va="center", fontsize=8.8, color=GREY)
    box(CX, 10.85, BW, 0.62, "输入：两组待比较的特征（理论选出 vs 算法选出）",
        fc="#f2f2f2", ec=GREY, fs=9)

    gates = [
        (9.55, "闸一 · 可识别性前置检验",
               "参数恢复须在数据实际所处的污染水平下运行\n配套判据：边界解率、量纲合理性",
               "参数不可解释\n比较无意义", True),
        (7.62, "闸二 · 维度严格匹配对照",
               "把两组特征锁死在相同维度 k\n差异只能来自「理论选的还是算法选的」",
               "差异可归因于\n特征维度", False),
        (5.69, "闸三 · 主分类器预注册",
               "事前固定唯一的可解释主分类器\n特征选择与标准化须在折内完成",
               "差异可归因于\n事后挑选分类器", False),
        (3.76, "闸四 · 以下游同款模型探测混淆",
               "对每个元数据变量单跑一次泄漏基线\n并核查距离矩阵中的 inf 元素",
               "差异可归因于\n混淆变量泄漏", True),
    ]
    for i, (y, title, detail, fail, newone) in enumerate(gates):
        col = AMBER if newone else BLUE
        box(CX, y, BW, 1.24, fc="#fdf4e6" if newone else "#eef3f8",
            ec=col, lw=1.6 if newone else 1.0)
        ax.text(CX, y + 0.34, title, ha="center", va="center",
                fontsize=9.4, fontweight="bold", color=col, zorder=4)
        ax.text(CX, y - 0.20, detail, ha="center", va="center",
                fontsize=8.3, color=INK, zorder=4, linespacing=1.55)
        if newone:
            ax.text(CX - BW/2 + 0.10, y + 0.72, "本研究增量", ha="left", va="center",
                    fontsize=7.6, color="white", fontweight="bold", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.22", fc=AMBER, ec="none"))
        arrow(CX + BW/2, y - 0.20, FX - 0.98, y - 0.20, RED, 0.9, (0, (4, 2)))
        ax.text((CX + BW/2 + FX - 0.98)/2, y - 0.06, "不通过",
                ha="center", va="bottom", fontsize=7.4, color=RED)
        box(FX + 0.32, y - 0.20, 2.10, 0.80, fail, fc="#fdeeee", ec=RED, lw=0.9,
            fs=7.8, tc=RED)
        y_next = gates[i+1][0] + 0.62 if i + 1 < len(gates) else 2.70
        arrow(CX, y - 0.62, CX, y_next, INK)
        ax.text(CX + 0.13, (y - 0.62 + y_next)/2, "通过", ha="left", va="center",
                fontsize=7.4, color=GREEN)

    box(CX, 2.36, BW, 0.68, "四道全部通过 → 特征间的比较方为有意义",
        fc="#eef6ef", ec=GREEN, lw=1.2, fs=9.4)
    ax.text(CX, 1.62,
            "四道都不把标签回灌进特征流程，都在比较开始之前运行；任一不通过，\n"
            "问题即归因于右侧对应的一栏，而不进入比较。",
            ha="center", va="center", fontsize=8.3, color=INK, linespacing=1.6)
    ax.text(CX, 0.62,
            "闸一与闸四为本研究增量：前者要求可识别性检验在数据实际所处的污染水平下运行；\n"
            "后者要求以下游同款模型探测混淆变量。",
            ha="center", va="center", fontsize=7.8, color=GREY,
            style="italic", linespacing=1.6)
    save(fig, "fig3_pre_checks")


# ============ 图2  左尾杠杆污染的机制判别（数据取自表3） ============
def fig2():
    lab    = ["0\n(干净)", "0.25%", "0.5%", "1%", "2%", "8%"]
    hit_up = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]                # 触及上界 %
    hit_lo = [0.0, 27.8, 47.0, 75.2, 94.2, 100.0]          # 触及优化下界 %
    d_up   = [98.5, 132.1, 145.1, 165.8, 170.6, 156.1]     # 距上界距离 ms
    ter    = [302, 191, 138, 57, 10, 0.1]                   # T_er 估计 ms
    a_est  = [1.200, 1.787, 2.064, 2.424, 2.477, 1.814]
    v_est  = [2.996, 3.379, 3.559, 3.745, 3.625, 2.757]
    x = np.arange(len(lab))
    i1 = 3                                                  # 1% —— 实测污染率所在行

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.3))

    # (A) H1 的判决性预测：触及上界率
    ax = axes[0]
    ax.plot(x, [0, 18, 34, 55, 75, 92], ls=(0, (5, 3)), color=GREY, lw=1.6, zorder=2)
    ax.annotate("H1 的预测：上界被压低\n→ 触及率应当上升", xy=(2.0, 34),
                xytext=(0.08, 88), fontsize=7.8, color=GREY, ha="left",
                va="center", linespacing=1.45,
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8))
    ax.plot(x, hit_lo, "-o", color=BLUE, lw=2.0, ms=5.5, zorder=4,
            label="触及优化下界")
    ax.plot(x, hit_up, "-o", color=RED, lw=2.8, ms=7.0, zorder=5,
            label="触及上界（实测）")
    for xi, yv in zip(x, hit_up):
        ax.text(xi, yv + 4.5, "0.0%", ha="center", fontsize=7.4,
                color=RED, fontweight="bold", zorder=6)
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=7.8)
    ax.set_ylim(-8, 110); ax.set_ylabel("比例（%）", fontsize=9.4)
    ax.set_xlabel("注入的污染比例", fontsize=9)
    ax.set_title("(A) H1 被否证：上界触及率恒为 0.0%",
                 fontsize=9.8, fontweight="bold", pad=8)
    ax.legend(fontsize=7.8, loc="center left", frameon=False,
              bbox_to_anchor=(0.01, 0.32))
    ax.spines[["top", "right"]].set_visible(False)

    # (B) T_er 被压低，但离上界更远
    ax = axes[1]
    ax.plot(x, ter, "-o", color=RED, lw=2.2, ms=5.5, label="$T_{er}$ 估计值")
    ax.plot(x, d_up, "-s", color=BLUE, lw=2.2, ms=5.0, label="与上界的距离")
    ax.annotate("与上界的距离反而拉大\n98.5 → 165.8 ms —— 与 H1 相反",
                xy=(3.0, 166.5), xytext=(2.05, 268), fontsize=7.8, color=BLUE,
                ha="left", va="center", linespacing=1.45,
                arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=1.2,
                                connectionstyle="arc3,rad=0.20"))
    ax.annotate("$T_{er}$ 被一路推向下界", xy=(4.15, 12), xytext=(2.30, 78),
                fontsize=7.8, color=RED, ha="left", va="center",
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.2,
                                connectionstyle="arc3,rad=-0.18"))
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=7.8)
    ax.set_ylabel("毫秒", fontsize=9.4); ax.set_xlabel("注入的污染比例", fontsize=9)
    ax.set_ylim(-20, 330)
    ax.set_title("(B) 被压低的同时远离上界 → 不是上界在起作用",
                 fontsize=9.8, fontweight="bold", pad=8)
    ax.legend(fontsize=7.8, loc="upper right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    # (C) 补偿脊 + 与真实数据的定量吻合
    ax = axes[2]
    ax.axhline(1.2, color=AMBER, lw=0.8, ls=":", zorder=1)
    ax.axhline(3.0, color=GREEN, lw=0.8, ls=":", zorder=1)
    ax.text(-0.42, 1.24, "真值 1.2", fontsize=7.2, color=AMBER, va="bottom")
    ax.text(-0.42, 3.04, "真值 3.0", fontsize=7.2, color=GREEN, va="bottom")
    ax.axvspan(i1 - 0.26, i1 + 0.26, color=BLUE, alpha=0.10, zorder=0)
    ax.plot(x, a_est, "-o", color=AMBER, lw=2.2, ms=5.5, label="$a$ 估计", zorder=4)
    ax.plot(x, v_est, "-o", color=GREEN, lw=2.2, ms=5.5, label="$v$ 估计", zorder=4)
    ax.scatter([i1], [2.319], marker="D", s=66, facecolor="white",
               edgecolor=AMBER, lw=1.8, zorder=6)
    ax.scatter([i1], [3.815], marker="D", s=66, facecolor="white",
               edgecolor=GREEN, lw=1.8, zorder=6)
    ax.annotate("◇ 老年组实测（污染率 0.97%）\n$a$ = 2.32　$v$ = 3.82",
                xy=(i1 + 0.16, 2.319), xytext=(3.50, 1.52), fontsize=7.6,
                color=INK, linespacing=1.45, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=GREY, lw=0.8))
    # 百分比按数据现算，避免改了数组却忘了改标签
    ax.text(i1 - 0.14, 2.46, f"+{(a_est[i1]/1.2-1)*100:.0f}%", ha="right",
            fontsize=8.2, color=AMBER, fontweight="bold", zorder=6)
    ax.text(i1 - 0.14, 3.72, f"+{(v_est[i1]/3.0-1)*100:.0f}%", ha="right",
            fontsize=8.2, color=GREEN, fontweight="bold", zorder=6)
    ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=7.8)
    ax.set_ylabel("参数估计值", fontsize=9.4); ax.set_xlabel("注入的污染比例", fontsize=9)
    ax.set_xlim(-0.55, 5.5); ax.set_ylim(0.7, 4.30)
    ax.set_title("(C) 补偿脊：整个参数向量被推离",
                 fontsize=9.8, fontweight="bold", pad=8)
    ax.legend(fontsize=7.8, loc="lower left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    fig.subplots_adjust(wspace=0.32)
    save(fig, "fig2_mechanism")


# ============ 图4  五条技术路线（数据取自表6） ============
def fig4():
    fig, ax = plt.subplots(figsize=(7.6, 4.5))
    routes = ["SWald-TG\n领域先验", "原始 RT\n描述统计", "RFE-DD\n算法选择",
              "1D-CNN\n端到端", "DTW-kNN\n基于距离"]
    auc = [0.666, 0.669, 0.646, 0.680, 0.624]
    x = np.arange(len(routes))

    ax.axhspan(0.624, 0.680, color=BLUE, alpha=0.10, zorder=0)
    ax.axhline(0.5, color=GREY, lw=0.9, ls="--", zorder=1)
    ax.text(4.66, 0.505, "随机基线 0.50", fontsize=7.4, color=GREY,
            va="bottom", ha="left")
    ax.axhline(0.766, color=RED, lw=1.3, ls=(0,(5,3)), zorder=2)
    ax.text(4.66, 0.822, "泄漏基线 0.766\n（仅用试次数）\n高于全部路线", fontsize=7.4,
            color=RED, va="center", ha="left", linespacing=1.4)
    ax.text(4.66, 0.648, "五条路线\n0.624–0.680\n四项比较 p ≥ 0.302", fontsize=7.4,
            color=BLUE, va="center", ha="left", linespacing=1.4)

    bars = ax.bar(x, [v - 0.45 for v in auc], bottom=0.45, width=0.58,
                  color=BLUE, edgecolor="white", lw=0.6, zorder=3)
    bars[3].set_color("#5c86b4")
    for i, v in enumerate(auc):
        ax.text(i, v + 0.008, f"{v:.3f}", ha="center", va="bottom",
                fontsize=9.0, fontweight="bold", color=INK, zorder=5)

    ax.annotate("带宽实现曾给出 0.950\n（150 ms 口径，已撤回）",
                xy=(4.30, 0.634), xytext=(2.95, 0.868), fontsize=7.6,
                color=RED, ha="center", linespacing=1.45,
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.1,
                                connectionstyle="arc3,rad=-0.16"))
    ax.annotate("150 ms 口径下曾为 0.713\n其中约 0.03 来自被污染的 $T_{er}$",
                xy=(-0.31, 0.664), xytext=(0.78, 0.800), fontsize=7.6,
                color=RED, ha="center", linespacing=1.45,
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.1,
                                connectionstyle="arc3,rad=0.22"))

    ax.set_xticks(x); ax.set_xticklabels(routes, fontsize=8.4)
    ax.set_xlim(-0.62, 6.15); ax.set_ylim(0.45, 0.93)
    ax.set_ylabel("LOOCV AUC", fontsize=9.5)
    ax.set_title("五条技术路线在同一协议、同一口径（250 ms）下的对照",
                 fontsize=10, fontweight="bold", pad=10)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, "fig4_routes")


# ============ 图5  跨模型：机制的前提条件（数据取自表3.3） ============
def fig5():
    """
    三栏叙事：成因 → 开关 → 梯度。

    (A) 三参数 Weibull 密度在支撑下界附近的行为，由形状参数 k 单独控制；
    (B) 该行为如何开关掉位置参数的失效方向（触上界 / 触下界）；
    (C) 估计值与上界的距离随 k 的连续梯度，跨三个数量级。

    数值取自 results/tables/table_crossmodel_ksweep.csv（每格 400 次重复，1% 污染）。
    """
    K   = [0.8, 1.0, 1.5, 2.0, 3.0]
    HIU = [97.5, 45.0, 0.2, 0.0, 0.0]     # 触及上界 %
    HIL = [0.0, 0.0, 0.0, 0.0, 38.5]      # 触及优化下界 %
    DIS = [0.1, 1.3, 13.4, 41.3, 130.7]   # 距上界距离 ms

    GAMMA, LAM = 0.30, 0.45
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9))

    # ---------- (A) 密度形态：机制的成因 ----------
    ax = axes[0]
    t = np.linspace(GAMMA + 1e-4, GAMMA + 0.32, 1200)
    z = (t - GAMMA) / LAM
    for k, col, lab, ls in [(0.8, RED, "k = 0.8　密度发散", "-"),
                            (1.0, AMBER, "k = 1.0　密度有限", "-"),
                            (2.0, BLUE, "k = 2.0　密度趋零", "-"),
                            (3.0, GREEN, "k = 3.0　密度趋零", "--")]:
        f = (k / LAM) * z ** (k - 1) * np.exp(-z ** k)
        ax.plot(t, f, color=col, lw=1.9, ls=ls, label=lab, clip_on=True)
    ax.axvline(GAMMA, color=INK, lw=1.1, ls=":")
    ax.text(GAMMA - 0.010, 3.0, "支撑下界 γ", fontsize=8.5, color=INK, rotation=90,
            va="center", ha="center")
    ax.set_xlim(GAMMA - 0.020, GAMMA + 0.32)
    ax.set_ylim(0, 6.0)
    ax.set_xlabel("t（秒）")
    ax.set_ylabel("密度 f(t)")
    ax.set_title("(A) 成因：密度在支撑下界处的行为\n由形状参数 k 单独控制", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    # ---------- (B) 触界率：机制的开关 ----------
    ax = axes[1]
    ax.axvspan(0.65, 1.0, color=RED, alpha=0.07)
    ax.axvspan(1.0, 3.2, color=BLUE, alpha=0.07)
    ax.plot(K, HIU, "o-", color=RED, lw=2.0, ms=6, label="触及【上】界")
    ax.plot(K, HIL, "s--", color=BLUE, lw=2.0, ms=5.5, label="触及【下】界")
    ax.axvline(1.0, color=INK, lw=1.2, ls=":")
    ax.annotate("97.5%", (0.8, 97.5), textcoords="offset points", xytext=(6, -4),
                fontsize=9, color=RED, fontweight="bold")
    ax.annotate("38.5%", (3.0, 38.5), textcoords="offset points", xytext=(-34, 4),
                fontsize=9, color=BLUE, fontweight="bold")
    ax.text(0.88, 60, "密度发散\n似然无界\n→ 推向上界", fontsize=8.2, color=RED,
            ha="left", va="center", linespacing=1.5)
    ax.text(2.15, 78, "密度趋零\n左尾杠杆\n→ 推离上界", fontsize=8.2, color=BLUE,
            ha="center", va="center", linespacing=1.5)
    ax.set_xlim(0.65, 3.2); ax.set_ylim(-6, 108)
    ax.set_xlabel("形状参数 k")
    ax.set_ylabel("占 400 次重复的比例（%）")
    ax.set_title("(B) 开关：方向在 k = 1 处反转", fontsize=9.5)
    ax.legend(frameon=False, fontsize=8, loc="center right")
    ax.spines[["top", "right"]].set_visible(False)

    # ---------- (C) 距离梯度 ----------
    ax = axes[2]
    ax.plot(K, DIS, "o-", color=INK, lw=2.0, ms=6)
    for k, d in zip(K, DIS):
        off = (0, 10) if k == 0.8 else ((-6, 10) if k == 3.0 else (0, 10))
        ha = "right" if k == 3.0 else "center"
        ax.annotate(f"{d:g}", (k, d), textcoords="offset points",
                    xytext=off, ha=ha, fontsize=8.5, color=INK)
    ax.axvline(1.0, color=INK, lw=1.2, ls=":")
    ax.set_yscale("log")
    ax.set_xlim(0.65, 3.2); ax.set_ylim(0.05, 400)
    ax.set_xlabel("形状参数 k")
    ax.set_ylabel("估计值与上界的距离（ms，对数轴）")
    ax.set_title("(C) 梯度：0.1 → 130.7 ms，跨三个数量级", fontsize=9.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.80, 0.062, "贴死上界", fontsize=8.2, color=RED, ha="center")
    ax.text(2.55, 45, "远离上界", fontsize=8.2, color=BLUE, ha="center")

    fig.text(0.5, -0.035,
             "失效不属于任何单个分布族，而属于「密度在支撑下界处趋零 + 以最大似然估计该下界」这一组合"
             "——撤掉组合的前一半（k ≤ 1），失效消失甚至反向",
             ha="center", fontsize=9, color=INK)
    fig.tight_layout()
    save(fig, "fig5_ksweep")


if __name__ == "__main__":
    print("生成 fig2_mechanism  → 论文图3.1（机制判别）…"); fig2()
    print("生成 fig3_pre_checks → 论文图6.1（四道前置检查）…"); fig3()
    print("生成 fig4_routes     → 未收入论文，参考图…"); fig4()
    print("生成 fig5_ksweep     → 论文图3.2（前提条件可开关）…"); fig5()
    print()
    print("论文图6.2（不可达路径泄漏诊断）不在本脚本内——它需要 OSF 原始数据，")
    print("由 fig1_leakage_diagnostic.py 独立生成：")
    print("    python fig1_leakage_diagnostic.py            # 接好数据后出正式图")
    print("    python fig1_leakage_diagnostic.py --preview  # 仅看布局")
