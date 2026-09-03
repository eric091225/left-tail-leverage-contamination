"""
06 端到端参照：1D-CNN 直接消费原始 RT 序列（秒），LOOCV。

【重要·CNN 必须与论文一致】
本文件的 CNN 是 1→8→16 通道，参数量 = 737，与论文第七章写的「约 737 个参数」、
参数量与逻辑回归的 5 个参数相比约为 147 倍。
你初版用的是 1→16→32（2721 个参数）。请勿混用——若你坚持用 1→16→32，
则论文中引用的参数量与倍数须同步更新——代码与论文务必同一个数。

修正点：架构对齐论文；输入为秒；标准化只在训练折 fit；与 SWald-TG(LR) 做 DeLong 单侧对照。
"""
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from delong import delong_roc_test

PKL_IN   = Path("../data/rt_data.pkl")
PRED_IN  = Path("../results/predictions_all.csv")
TABLE3_OUT = Path("../results/tables/table3.csv")

SEQ_LEN = 150
EPOCHS  = 60
LR      = 1e-3
WD      = 1e-3
DROPOUT = 0.5
SEED    = 42


class CNN1D(nn.Module):
    """与论文一致的小容量 1D-CNN：1→8→16，约 737 参数。"""
    def __init__(self, dropout=DROPOUT):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(8, 16, kernel_size=5, padding=2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(16, 1))

    def forward(self, x):              # x: [B, L]
        h = self.net(x.unsqueeze(1)).squeeze(-1)   # [B, 16]
        return self.head(h).squeeze(-1)


def to_matrix(rt_data, subjects, seq_len=SEQ_LEN):
    """每被试 RT 序列（秒）截断/补零到固定长度。"""
    M = np.zeros((len(subjects), seq_len), dtype=float)
    for r, s in enumerate(subjects):
        seq = np.asarray(rt_data[s], dtype=float)[:seq_len]
        M[r, :len(seq)] = seq
    return M


def train_predict(Xtr, ytr, Xte, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    model = CNN1D()
    opt = optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    crit = nn.BCEWithLogitsLoss()
    Xtr_t = torch.tensor(Xtr, dtype=torch.float32)
    ytr_t = torch.tensor(ytr, dtype=torch.float32)
    model.train()
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = crit(model(Xtr_t), ytr_t)
        loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(Xte, dtype=torch.float32))).numpy()


def main():
    print("CNN 参数量 =", sum(p.numel() for p in CNN1D().parameters()), "（应为 737，与论文一致）")
    with open(PKL_IN, "rb") as f:
        data = pickle.load(f)
    rt_data, labels = data["rt_data"], data["labels"]

    # 与 04 对齐到同一批被试（用 predictions_all 的顺序与标签）
    pred_df = pd.read_csv(PRED_IN)
    y = pred_df["true_label"].values.astype(int)
    # predictions_all 未存被试ID；这里按 swald_features 的被试顺序重建
    sw = pd.read_csv("../results/features/swald_features.csv", index_col=0)
    good = sw[["v", "a", "ter", "rt_cog"]].notna().all(axis=1)
    if "converged" in sw.columns:
        good &= sw["converged"].astype(bool)
    subjects = list(sw.index[good])
    assert len(subjects) == len(y), "被试数与 04 不一致，请确认 04 已先运行"

    M = to_matrix(rt_data, subjects)
    n = len(y)
    oof = np.zeros(n)
    for i in range(n):
        tr = np.delete(np.arange(n), i)
        sc = StandardScaler().fit(M[tr])
        Xtr, Xte = sc.transform(M[tr]), sc.transform(M[i:i+1])
        oof[i] = train_predict(Xtr, y[tr], Xte, seed=SEED + i)[0]
        if (i + 1) % 10 == 0:
            print(f"  CNN LOOCV 完成 {i+1}/{n}")

    auc_cnn = roc_auc_score(y, oof)
    p_tg = pred_df["TG_lr"].values
    auc_tg, _, z, p = delong_roc_test(y, p_tg, oof, alternative="greater")

    print(f"\n1D-CNN AUC = {auc_cnn:.4f}")
    print(f"SWald-TG(LR) AUC = {auc_tg:.4f}")
    print(f"DeLong 单侧 (H_A: TG>CNN): z={z:.3f}, p={p:.4f}")
    verdict = "未超过" if auc_cnn <= auc_tg else "超过"
    print(f"→ 端到端参照{verdict}领域先验特征（对应论文第七章五路线段的 1D-CNN 行）")

    TABLE3_OUT.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{
        "model": "1D-CNN", "AUC": round(auc_cnn, 4),
        "SWald_TG_LR": round(auc_tg, 4),
        "DeLong_p_TG_gt_CNN": round(p, 4),
    }]).to_csv(TABLE3_OUT, index=False)

    # 存档 out-of-fold 预测，供后续计算 Bootstrap 置信区间等复用
    pd.DataFrame({"true_label": y, "CNN": oof}).to_csv(
        TABLE3_OUT.parent.parent / "predictions_cnn.csv", index=False)
    print(f"已保存表3 → {TABLE3_OUT}")


if __name__ == "__main__":
    main()
