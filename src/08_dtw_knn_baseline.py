"""
08 经典时序基线：DTW + kNN（修正版，对应论文 6.2 节）

【本文件曾出过一个重要错误，修正过程已写进论文，请勿改回】
初版用「变长序列 + Sakoe-Chiba 带宽 20」得到 AUC = 0.950，一度被当成全文唯一显著结果。
复核确认该结果不成立，成因是两层叠加的序列长度混淆：

  (1) 数据集本身：两组试次数均值接近（年轻 172.0、老年 166.7），但老年组标准差约为年轻组
      四倍（56.4 vs 14.5），且试次数 < 130 的 8 名被试全部属于老年组。
      仅用试次数做 kNN，AUC 即达 0.69–0.79 —— 变体 C 就是把这个泄漏显式量化出来。
  (2) 实现层面：带宽受限的 DTW 在长度差 > 带宽的序列对上无法形成可达路径，返回 inf。
      实测 62.2% 的被试对落入此情形（长度差最小 21、中位 50、最大 211）。
      下游 kNN 的 argsort 把 inf 排到末位，等价于「只有长度差 ≤ 带宽者才可能成为邻居」。

因此本文件：① 一律全窗 DP（不设带宽），杜绝 inf；② 提供等长截断与正确归一化两个变体；
③ 显式跑「仅用试次数」的泄漏基线，把混淆强度报出来。

【归一化的正确写法 —— 另一个容易踩的坑】
DTW 累积路径上的平方代价，总量正比于路径长度 L。
  错误：sqrt(总代价) / L   → 正比于 1/sqrt(L)，长度依赖没消除，只是方向反了
  正确：sqrt(总代价 / L)   → 路径均方根，与长度基本无关
本文件用后者，并在 main() 开头做长度不变性自检。

【不做逐序列 z-标准化】
UCR 上 DTW-kNN 通常先 z-normalize 每条序列，但本任务的类别信号正体现在 RT 的量级上
（老年组整体更慢），逐序列标准化会把它抹掉。故使用原始 RT（秒）。

三个变体（对应论文 6.2 节）：
  A  统一截断至数据集最小试次数 → 全部等长，长度混淆彻底消除（最公平，主报告值）
  B  变长 + 全窗 + 路径长度归一化 → 保留全部数据，但归一化削弱幅度信息（对 DTW 偏严格）
  C  仅用试次数的 kNN → 泄漏基线，是诊断而非方法
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

PKL_IN    = Path("../data/rt_data.pkl")
PRED_IN   = Path("../results/predictions_all.csv")
FEAT_SW   = Path("../results/features/swald_features.csv")
TABLE_OUT = Path("../results/tables/table_dtw_knn.csv")

K_LIST = (1, 3, 5, 7, 9, 11)


def dtw_distance(a, b, normalize=False):
    """全窗 DP（无带宽，不会出现 inf）。

    normalize=True 返回 sqrt(总代价 / max(na, nb))。

    【须如实说明】这里除的是 max(na, nb)，**不是真实的翘曲路径长度**。真实路径长
    L 满足 max(na,nb) ≤ L ≤ na+nb−1，且二者之差在等长对上最大、在极不等长对上
    最小（实测：(200,200) 真实 265 对 max 200；(200,45) 真实 202 对 max 200）。
    因此两种归一化并不等价：本数据上变体 B 的 AUC 为 0.542（max）对 0.678（真实
    路径长），相差 0.136。本研究沿用 max(na, nb)——它是常见且计算廉价的长度归一
    化——但该选择必须显式声明，不可当作「路径长度归一化」的唯一实现。
    切勿写成 sqrt(总代价)/L，那会引入反向的长度偏差。
    """
    na, nb = len(a), len(b)
    prev = np.full(nb + 1, np.inf)
    prev[0] = 0.0
    for i in range(1, na + 1):
        cur = np.full(nb + 1, np.inf)
        ai = a[i - 1]
        for j in range(1, nb + 1):
            cur[j] = (ai - b[j - 1]) ** 2 + min(prev[j], cur[j - 1], prev[j - 1])
        prev = cur
    total = prev[nb]
    return np.sqrt(total / max(na, nb)) if normalize else np.sqrt(total)


def build_matrix(seqs, normalize=False, tag=""):
    n = len(seqs)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            D[i, j] = D[j, i] = dtw_distance(seqs[i], seqs[j], normalize)
        if (i + 1) % 15 == 0:
            print(f"    {tag} 距离矩阵 {i+1}/{n}")
    off = ~np.eye(n, dtype=bool)
    assert np.isfinite(D[off]).all(), "距离矩阵出现 inf —— 检查是否误用了带宽"
    return D


def nested_knn(D, y, K=K_LIST):
    """外层 LOOCV；内层再 LOOCV 选 k（超参数不在测试样本上调优）。"""
    n = len(y)
    D = D.copy()
    np.fill_diagonal(D, np.inf)          # 不把自己当邻居
    oof = np.zeros(n)
    for i in range(n):
        tr = np.setdiff1d(np.arange(n), [i])
        best_k, best_auc = K[0], -1.0
        for k in K:
            if k >= len(tr):
                continue
            sc = np.empty(len(tr))
            for m, a in enumerate(tr):
                tr2 = np.setdiff1d(tr, [a])
                nn = tr2[np.argsort(D[a][tr2])[:k]]
                sc[m] = y[nn].mean()
            au = roc_auc_score(y[tr], sc)
            # 【显式声明】内层 AUC 并列时取**较小**的 k（K_LIST 靠前者）。
            # 这不是无关紧要的细节：改为 `>=`（并列取较大 k）会使变体 A 的
            # AUC 由 0.624 变为 0.637。属实现选择的敏感性，见 README。
            if au > best_auc:
                best_auc, best_k = au, k
        nn = tr[np.argsort(D[i][tr])[:best_k]]
        oof[i] = y[nn].mean()
    return roc_auc_score(y, oof), oof


def selfcheck_normalization():
    """长度不变性自检。

    【修正记录】此前的版本只比较**等长**序列对（45,45）(120,120)(200,200)，
    而变体 B 的全部意义在于消除**不等长**比较中的长度偏差——原自检因此不可能
    检出它要防的那个问题。现补上不等长对，并分别报告两类离散。
    """
    rng = np.random.default_rng(0)

    eq = [dtw_distance(rng.normal(0.6, 0.15, L), rng.normal(0.6, 0.15, L),
                       normalize=True) for L in (45, 120, 200)]
    spread_eq = (max(eq) - min(eq)) / float(np.mean(eq))
    print(f"  等长对 (45,45)/(120,120)/(200,200) = "
          f"{eq[0]:.4f} / {eq[1]:.4f} / {eq[2]:.4f}，相对离散 {spread_eq:.1%}")

    # 关键：同分布但不等长。理想的长度不变量在此应保持平坦。
    ne = [float(np.mean([dtw_distance(rng.normal(0.6, 0.15, 200),
                                      rng.normal(0.6, 0.15, L2), normalize=True)
                         for _ in range(20)])) for L2 in (45, 120, 200)]
    spread_ne = (max(ne) - min(ne)) / float(np.mean(ne))
    print(f"  不等长对 (200,45)/(200,120)/(200,200) = "
          f"{ne[0]:.4f} / {ne[1]:.4f} / {ne[2]:.4f}，相对离散 {spread_ne:.1%}")

    assert spread_eq < 0.25, "等长对上归一化即随长度漂移——检查是否写成了 sqrt(cost)/L"
    assert spread_ne < 0.25, "不等长对上仍有长度偏差——max(na,nb) 归一化不足以消除"
    print(f"  自检通过。注意不等长对的离散（{spread_ne:.1%}）显著大于等长对"
          f"（{spread_eq:.1%}）：max(na,nb) 只是近似的长度不变量。")


def main():
    print("=== 归一化正确性自检 ===")
    selfcheck_normalization()

    with open(PKL_IN, "rb") as f:
        data = pickle.load(f)
    rt_data, labels = data["rt_data"], data["labels"]

    sw = pd.read_csv(FEAT_SW, index_col=0)
    good = sw[["v", "a", "ter", "rt_cog"]].notna().all(axis=1)
    if "converged" in sw.columns:
        good &= sw["converged"].astype(bool)
    subjects = list(sw.index[good])
    y = np.array([labels[s] for s in subjects], dtype=int)
    seqs = [np.asarray(rt_data[s], float) for s in subjects]
    lens = np.array([len(s) for s in seqs])
    n = len(y)

    print(f"\n参与评估 N={n}（年轻 {int((y==0).sum())} / 老年 {int((y==1).sum())}）")
    print(f"试次数：年轻 {lens[y==0].mean():.1f}±{lens[y==0].std(ddof=1):.1f}，"
          f"老年 {lens[y==1].mean():.1f}±{lens[y==1].std(ddof=1):.1f}")

    rows = []

    print("\n=== 变体 C：泄漏基线（仅用试次数）===")
    Dc = np.abs(lens[:, None] - lens[None, :]).astype(float)
    auc_c, _ = nested_knn(Dc, y)
    print(f"  嵌套 LOOCV AUC = {auc_c:.3f}   ← 序列长度本身携带的标签信息")
    rows.append({"variant": "C 泄漏基线（仅试次数）", "AUC": round(auc_c, 4)})

    MIN = int(lens.min())
    print(f"\n=== 变体 A：统一截断至 {MIN} 试次（全部等长）===")
    Da = build_matrix([s[:MIN] for s in seqs], normalize=False, tag="A")
    auc_a, oof_a = nested_knn(Da, y)
    print(f"  嵌套 LOOCV AUC = {auc_a:.3f}")
    rows.append({"variant": f"A 等长截断至{MIN}", "AUC": round(auc_a, 4)})

    print("\n=== 变体 B：变长 + 全窗 + 路径长度归一化 ===")
    Db = build_matrix(seqs, normalize=True, tag="B")
    auc_b, oof_b = nested_knn(Db, y)
    print(f"  嵌套 LOOCV AUC = {auc_b:.3f}")
    rows.append({"variant": "B 变长+归一化", "AUC": round(auc_b, 4)})

    if PRED_IN.exists():
        pred = pd.read_csv(PRED_IN)
        y_ref = pred["true_label"].values.astype(int)
        if len(y_ref) == n:
            tg = pred["TG_lr"].values
            print(f"\n=== 与 SWald-TG（{roc_auc_score(y_ref, tg):.3f}）比较 ===")
            try:
                from delong import delong_roc_test
                for lab, oof in (("A", oof_a), ("B", oof_b)):
                    a1, a2, z, p = delong_roc_test(y_ref, oof, tg, alternative="greater")
                    print(f"  变体{lab}: ΔAUC {a1-a2:+.3f} | DeLong p(DTW>TG) = {p:.4f}")
            except Exception as e:
                print(f"  [跳过 DeLong] {e}")

    TABLE_OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(TABLE_OUT, index=False)
    print(f"\n已保存 → {TABLE_OUT}")
    print("对应论文 6.2 节：变体 A 为主报告；本版论文不使用变体 B 的结果。")


if __name__ == "__main__":
    main()
