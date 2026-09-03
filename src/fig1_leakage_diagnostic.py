#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
图1　不可达路径泄漏的诊断图（两联）

(A) 每被试试次数的分组点图：两组均值接近而标准差相差约四倍，
    试次数 < 130 的 8 名被试全部属于老年组 —— 混淆在方差与局部聚类，不在均值。
(B) 59x59 的带宽 DTW 距离矩阵热力图（被试按试次数排序）：
    长度差 > 带宽的被试对不存在可达路径，返回 inf（单独着色），
    可见其呈明显的对角带状结构 —— 距离矩阵实际编码的是序列长度。

用法
----
    python fig1_leakage_diagnostic.py              # 用真实数据出图（需先接好 load_sequences）
    python fig1_leakage_diagnostic.py --preview    # 用重建的合成试次数出布局预览

注意：本图刻意复现的是「被撤回的那个实现」——Sakoe-Chiba 带宽 DTW，半径 20，
不可达对原样返回 inf。这正是当时给出 AUC = 0.950 的那个距离矩阵。
"""

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

# ---------------------------------------------------------------- 样式
def _resolve_cjk_font():
    """按平台挑一个实际存在的中文字体；都没有则显式报警，避免静默渲染成方块。"""
    import matplotlib.font_manager as fm
    have = {f.name for f in fm.fontManager.ttflist}
    for name in ("PingFang SC", "Songti SC", "Heiti TC",          # macOS
                 "Noto Sans CJK SC", "WenQuanYi Zen Hei",          # Linux
                 "Microsoft YaHei", "SimHei",                      # Windows
                 "Arial Unicode MS"):
        if name in have:
            return name
    import warnings
    warnings.warn("未找到任何中文字体，图中汉字将显示为方块。"
                  "请安装 Noto Sans CJK SC 或在 _resolve_cjk_font() 中补入本机字体名。")
    return "DejaVu Sans"


CJK_FONT = _resolve_cjk_font()

plt.rcParams.update({
    "font.sans-serif": [CJK_FONT, "DejaVu Sans"],
    "font.family": "sans-serif",
    "axes.unicode_minus": False,
    "figure.dpi": 110,
    "savefig.dpi": 300,
    "axes.linewidth": 0.9,
    "axes.edgecolor": "#4a4a4a",
})

C_YOUNG = "#2F6CA8"     # 年轻组
C_OLD = "#C0392B"       # 老年组
C_INF = "#E8B4AD"       # 不可达（inf）单元
C_TEXT = "#333333"
C_NOTE = "#B03A2E"

BAND_RADIUS = 20        # Sakoe-Chiba 半径，与被撤回的实现一致


# ---------------------------------------------------------------- 数据接口
def load_sequences(csv_path="../data/ratcliff_vanunu_2022.csv"):
    """接入 Ratcliff & Vanunu (2022) OSF 公开数据（https://osf.io/exz4d/，ratcliff_vanunu_2022.csv）。

    返回三元组 (counts, groups, seqs)：
        counts : (59,) int    每名被试的试次数
        groups : (59,) int    0 = 年轻组(Age==1), 1 = 老年组(Age==2)
        seqs   : list[np.ndarray]  每名被试的反应时序列（秒）

    口径说明：只取 Task == 1（单选项任务，无 Choice/Accuracy，合计 9,996 个观测），
    并沿用被撤回那次实现的 150 ms 剔除下限——该数据 RT 实际范围为 152–3897 ms，
    故此阈值不剔除任何试次，(B) 复现的即是当时那个 AUC = 0.950 的距离矩阵。
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    df = df[(df.Task == 1) & (df.RT >= 150) & (df.RT <= 4000)]
    counts, groups, seqs = [], [], []
    for sid, g in df.groupby("Subject", sort=True):
        seqs.append(g.RT.to_numpy(float) / 1000.0)      # 转为秒
        counts.append(len(g))
        groups.append(0 if int(g.Age.iloc[0]) == 1 else 1)
    return np.asarray(counts), np.asarray(groups), seqs


def synth_sequences(seed=20260821):
    """按论文报告的统计量重建试次数，用于布局预览。合成数据，不可用于论文。"""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(4000):
        y = rng.normal(172.0, 14.5, 30)
        low = rng.uniform(45, 129, 8)
        up = rng.normal(rng.uniform(180, 200), rng.uniform(18, 34), 20)
        c = np.round(np.concatenate([y, low, up, [256.0]])).astype(int)
        c = np.clip(c, 45, 256)
        g = np.array([0] * 30 + [1] * 29)
        c[g == 0] += int(round(172.0 - c[g == 0].mean()))
        c[g == 1] += int(round(166.7 - c[g == 1].mean()))
        c = np.clip(c, 45, 256)
        d = np.abs(c[:, None] - c[None, :])[np.triu_indices(len(c), 1)]
        m = d > BAND_RADIUS
        if not (g[c < 130] == 1).all() or (c < 130).sum() != 8:
            continue
        err = (abs(c[g == 0].mean() - 172.0) + abs(c[g == 1].mean() - 166.7)
               + abs(c[g == 0].std(ddof=1) - 14.5) + abs(c[g == 1].std(ddof=1) - 56.4)
               + abs(m.mean() - 0.622) * 300 + abs(np.median(d[m]) - 50) / 2)
        if best is None or err < best[0]:
            best = (err, c.copy(), g.copy())
    _, counts, groups = best
    seqs = [rng.lognormal(-0.55, 0.32, n) for n in counts]   # 仅为让 (B) 有可视的有限值
    return counts, groups, seqs


# ---------------------------------------------------------------- 带宽 DTW
def dtw_banded(x, y, radius=BAND_RADIUS):
    """Sakoe-Chiba 带宽约束的 DTW。带内无可达路径时返回 inf（多数实现的默认行为）。"""
    n, m = len(x), len(y)
    if abs(n - m) > radius:          # 带内不存在从 (0,0) 到 (n,m) 的路径
        return np.inf
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0
    for i in range(1, n + 1):
        lo = max(1, i - radius)
        hi = min(m, i + radius)
        cost = np.abs(x[i - 1] - y[lo - 1:hi])
        prev, cur = D[i - 1], D[i]
        for k, j in enumerate(range(lo, hi + 1)):
            cur[j] = cost[k] + min(prev[j], cur[j - 1], prev[j - 1])
    return D[n, m]


def distance_matrix(seqs, radius=BAND_RADIUS):
    N = len(seqs)
    D = np.full((N, N), np.nan)
    for i in range(N):
        D[i, i] = 0.0
        for j in range(i + 1, N):
            d = dtw_banded(seqs[i], seqs[j], radius)
            D[i, j] = D[j, i] = d
    return D


# ---------------------------------------------------------------- 绘图
def make_figure(counts, groups, D, preview=False, out="fig1"):
    N = len(counts)
    order = np.argsort(counts)
    Ds, gs, cs = D[np.ix_(order, order)], groups[order], counts[order]

    iu = np.triu_indices(N, 1)
    gaps = np.abs(counts[:, None] - counts[None, :])[iu]
    inf_pairs = gaps > BAND_RADIUS
    frac = inf_pairs.mean()

    # 自检：实际距离矩阵中 inf 的位置，必须与「长度差 > 带宽」这一判据完全吻合。
    # 这一步把图中报告的 62.2% 从「由长度推断」升级为「由距离矩阵实测」。
    order_iu = np.triu_indices(N, 1)
    observed_inf = (~np.isfinite(Ds))[order_iu]
    predicted_inf = (np.abs(cs[:, None] - cs[None, :]) > BAND_RADIUS)[order_iu]
    assert np.array_equal(observed_inf, predicted_inf), \
        "距离矩阵的 inf 模式与长度判据不符——请检查 dtw_banded 的带宽实现"
    assert abs(observed_inf.mean() - frac) < 1e-12

    fig = plt.figure(figsize=(14.2, 6.8))
    gs_ = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.34], wspace=0.24,
                           left=0.055, right=0.955,
                           bottom=0.155, top=0.815 if not preview else 0.775)

    # ---------- (A) 试次数分组点图 ----------
    ax = fig.add_subplot(gs_[0])
    rng = np.random.default_rng(7)
    means = {}
    for k, col in enumerate([C_YOUNG, C_OLD]):
        v = counts[groups == k]
        x = k + rng.uniform(-0.115, 0.115, len(v))
        ax.scatter(x, v, s=46, color=col, alpha=0.78, edgecolor="white",
                   linewidth=0.8, zorder=3)
        mu, sd = v.mean(), v.std(ddof=1)
        means[k] = (mu, sd)
        ax.hlines(mu, k - 0.25, k + 0.32, color=col, lw=2.4, zorder=4)
        ax.add_patch(Rectangle((k + 0.235, mu - sd), 0.085, 2 * sd,
                               facecolor=col, alpha=0.28, edgecolor=col, lw=1.0, zorder=2))
        ax.text(k + 0.345, mu + 12, f"均值 {mu:.1f}", color=col, fontsize=11,
                va="bottom", ha="left")
        ax.text(k + 0.345, mu - 12, f"SD {sd:.1f}", color=col, fontsize=11,
                va="top", ha="left", fontweight="bold")

    low = counts < 130
    ax.scatter(np.where(groups[low] == 0, 0, 1) + rng.uniform(-0.115, 0.115, low.sum()),
               counts[low], s=175, facecolor="none", edgecolor=C_NOTE,
               linewidth=1.7, zorder=5)
    ax.axhline(130, color=C_NOTE, ls="--", lw=1.1, alpha=0.75, zorder=1)
    ax.text(-0.52, 132, "130 试次", color=C_NOTE, fontsize=9.8, va="bottom")
    ax.annotate(f"试次数 < 130 的 {int(low.sum())} 名被试\n全部属于老年组",
                xy=(0.84, np.sort(counts[low])[-2]), xytext=(-0.50, 78),
                color=C_NOTE, fontsize=11, fontweight="bold", ha="left", va="center",
                arrowprops=dict(arrowstyle="->", color=C_NOTE, lw=1.5,
                                connectionstyle="arc3,rad=-0.28"))

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["年轻组\n(n = 30)", "老年组\n(n = 29)"], fontsize=12)
    ax.set_xlim(-0.58, 1.60)
    ax.set_ylim(28, 278)
    ax.set_ylabel("每被试试次数", fontsize=12)
    ax.set_title("(A) 混淆不在均值上，而在方差与局部聚类上",
                 fontsize=13.5, fontweight="bold", pad=26, color=C_TEXT)
    _md = abs(means[0][0] - means[1][0])
    _rt = max(means[0][1], means[1][1]) / max(min(means[0][1], means[1][1]), 1e-9)
    ax.text(0.5, 1.035, f"均值仅差 {_md:.1f}，标准差差约 {_rt:.1f} 倍",
            transform=ax.transAxes, ha="center", fontsize=10.6, color="#6a6a6a")
    ax.grid(axis="y", color="#dcdcdc", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # ---------- (B) 距离矩阵热力图 ----------
    axb = fig.add_subplot(gs_[1])
    finite = Ds[np.isfinite(Ds) & (Ds > 0)]
    vmax = np.percentile(finite, 97) if finite.size else 1.0
    base = plt.get_cmap("Blues")
    cmap = ListedColormap(base(np.linspace(0.10, 0.92, 256)))
    cmap.set_bad(C_INF)

    im = axb.imshow(np.ma.masked_invalid(Ds), cmap=cmap, vmin=0, vmax=vmax,
                    interpolation="nearest", origin="upper")

    # 组别色条（左侧与顶部）：低试次一端全部为老年组
    for i, gg in enumerate(gs):
        col = C_YOUNG if gg == 0 else C_OLD
        axb.add_patch(Rectangle((-2.6, i - 0.5), 1.9, 1, facecolor=col,
                                edgecolor="none", clip_on=False))
        axb.add_patch(Rectangle((i - 0.5, -2.6), 1, 1.9, facecolor=col,
                                edgecolor="none", clip_on=False))

    axb.set_xlim(-3.4, N - 0.5)
    axb.set_ylim(N - 0.5, -3.4)
    axb.set_xlabel("被试（按试次数升序排列）→", fontsize=11.5)
    axb.set_ylabel("← 被试（按试次数升序排列）", fontsize=11.5)
    axb.set_xticks([]); axb.set_yticks([])
    axb.set_title("(B) 距离矩阵实际编码的是序列长度，不是内容",
                  fontsize=13.5, fontweight="bold", pad=26, color=C_TEXT)
    axb.text(0.5, 1.035, f"带宽半径 20 下 {frac*100:.1f}% 的被试对不存在可达路径",
             transform=axb.transAxes, ha="center", fontsize=10.6, color="#6a6a6a")

    cb = fig.colorbar(im, ax=axb, fraction=0.040, pad=0.035)
    cb.set_label("DTW 距离（可达对）", fontsize=10.5)
    cb.outline.set_linewidth(0.7)

    # 两个区域直接就地标注，不用引线
    axb.text(N * 0.78, N * 0.15,
             f"不可达 → inf\n（{frac*100:.1f}% 的被试对）\n被 kNN 排在最后",
             fontsize=11, color=C_NOTE, fontweight="bold", ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.42", fc="white", ec=C_NOTE, lw=1.0, alpha=0.94))
    axb.text(N * 0.20, N * 0.86,
             "可达对\n（长度差 ≤ 20）",
             fontsize=11, color="#1F4E79", fontweight="bold", ha="center", va="center",
             bbox=dict(boxstyle="round,pad=0.42", fc="white", ec="#1F4E79", lw=1.0, alpha=0.94))
    axb.annotate("", xy=(N * 0.47, N * 0.53), xytext=(N * 0.27, N * 0.79),
                 arrowprops=dict(arrowstyle="->", color="#1F4E79", lw=1.5,
                                 connectionstyle="arc3,rad=-0.25"))

    # 不可达图例块
    axb.add_patch(Rectangle((0.0, -0.115), 0.030, 0.042, transform=axb.transAxes,
                            facecolor=C_INF, edgecolor="#8a8a8a", lw=0.7, clip_on=False))
    axb.text(0.038, -0.094, "不可达 = inf", transform=axb.transAxes,
             fontsize=10, va="center", color=C_NOTE)
    axb.add_patch(Rectangle((0.275, -0.115), 0.030, 0.042, transform=axb.transAxes,
                            facecolor=C_YOUNG, edgecolor="none", clip_on=False))
    axb.text(0.313, -0.094, "年轻组", transform=axb.transAxes, fontsize=10,
             va="center", color="#666666")
    axb.add_patch(Rectangle((0.415, -0.115), 0.030, 0.042, transform=axb.transAxes,
                            facecolor=C_OLD, edgecolor="none", clip_on=False))
    axb.text(0.453, -0.094, "老年组", transform=axb.transAxes, fontsize=10,
             va="center", color="#666666")

    med = int(np.median(gaps[inf_pairs])) if inf_pairs.any() else 0
    fig.text(0.5, 0.048,
             f"不可达对的长度差：最小 {int(gaps[inf_pairs].min())}、中位 {med}、"
             f"最大 {int(gaps[inf_pairs].max())}",
             ha="center", fontsize=10.6, color="#555555")
    fig.text(0.5, 0.013,
             "带宽是标准做法，kNN 是标准分类器，长度不等是数据的客观性质 —— "
             "三者单独都不构成错误，耦合起来生成一个 AUC = 0.950 的伪信号",
             ha="center", fontsize=10.6, color=C_TEXT, style="italic")

    if preview:
        fig.patches.append(Rectangle((0.0, 0.945), 1.0, 0.055, transform=fig.transFigure,
                                     facecolor="#FDECEA", edgecolor=C_NOTE, lw=1.1, zorder=10))
        fig.text(0.5, 0.9715,
                 "布局预览　·　试次数为按论文报告统计量重建的合成值，距离为模拟值　·　"
                 "不可用于论文，请接入真实数据后重跑本脚本",
                 ha="center", va="center", fontsize=11, color=C_NOTE,
                 fontweight="bold", zorder=11)

    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return frac


# ---------------------------------------------------------------- main
_FIGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       "..", "results", "figures")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", action="store_true", help="用重建的合成试次数出布局预览")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    if a.preview:
        counts, groups, seqs = synth_sequences()
        out = a.out or os.path.join(_FIGDIR, "fig1_preview")
    else:
        counts, groups, seqs = load_sequences()
        out = a.out or os.path.join(_FIGDIR, "fig1")

    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    D = distance_matrix(seqs)
    frac = make_figure(counts, groups, D, preview=a.preview, out=out)

    print(f"已输出 {out}.png / {out}.pdf")
    print(f"  被试数            {len(counts)}（年轻 {int((groups==0).sum())}，"
          f"老年 {int((groups==1).sum())}）")
    print(f"  试次数            均值 {counts[groups==0].mean():.1f} / "
          f"{counts[groups==1].mean():.1f}，SD {counts[groups==0].std(ddof=1):.1f} / "
          f"{counts[groups==1].std(ddof=1):.1f}")
    print(f"  不可达对占比      {frac*100:.1f}%")
    print(f"  < 130 试次        {int((counts<130).sum())} 名，"
          f"全部老年组：{bool((groups[counts<130]==1).all())}")

    # 与论文正文报告的数值逐项比对——不符即说明预处理口径或带宽设置有出入
    gp = np.abs(counts[:, None] - counts[None, :])[np.triu_indices(len(counts), 1)]
    gp = gp[gp > BAND_RADIUS]
    print(f"  不可达对长度差    最小 {int(gp.min())}、中位 {int(np.median(gp))}、"
          f"最大 {int(gp.max())}")
    if not a.preview:
        print()
        print("  与论文 3.1 节报告值比对（不符请检查 150 ms 预处理口径与带宽半径）：")
        for label, got, want in (
            ("两组均值",   f"{counts[groups==0].mean():.1f}/{counts[groups==1].mean():.1f}", "172.0/166.7"),
            ("两组标准差", f"{counts[groups==0].std(ddof=1):.1f}/{counts[groups==1].std(ddof=1):.1f}", "14.5/56.4"),
            ("不可达占比", f"{frac*100:.1f}%", "62.2%"),
            ("长度差",     f"{int(gp.min())}/{int(np.median(gp))}/{int(gp.max())}", "21/50/211"),
        ):
            print(f"    {label:10} 实测 {got:16} 论文 {want:12}"
                  f"{'  ✓' if got.replace(' ','') == want else '  ← 请核对'}")
