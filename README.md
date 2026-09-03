# 越错越显著：一条从污染率到分类器 AUC 的泄漏通道

本仓库是该论文的完整复现代码。

**先说清主体在哪。** 论文的核心论证是**受控实验**（脚本 `10`–`16`），其证据强度不依赖任何单个
数据集的样本量；Ratcliff & Vanunu (2022) 的 OSF 公开数据只提供一个独立测得的污染率锚点
（老年组 0.97%、年轻组 0.23%）与一组供比对的实测参数。**因此本仓库有两条使用路径**：

- **路径 A（论文主体，无需下载数据）**：跑 `10`、`13`、`14`、`15`、`16` 与 `make_figures`，
  可复现表3.1／3.2／3.3／4.3／5.1 与图3.1／3.2／6.1
- **路径 B（完整复现）**：先下载 OSF 数据，再依序跑全部脚本

仓库中同时保留了两处**已被撤回或已被否证**的实现，加注说明，以便他人复核溯源过程。

---

## 快速开始

```bash
pip install -r requirements.txt
cd src

# ── 路径 A：论文主体，不需要任何外部数据 ──
python 10_injection_mechanism.py         # 表3.1、图3.1 的数据
python 13_recovery_blindspot.py          # 表5.1
python 14_crossmodel_weibull.py          # 表3.2、表3.3、图3.2 的数据
python 15_crossmodel_differential.py     # 表4.2
python 16_leakage_to_classifier.py       # 表4.3（唯一需要 scikit-learn 的受控实验）
python make_figures.py                   # 图3.1、图3.2、图6.1

# ── 路径 B：完整复现，先取数据 ──
cd .. && curl -L -o data/ratcliff_vanunu_2022.csv \
  "https://osf.io/download/61f043899f195e075a2e22b5/"      # md5 见 data/README.md
cd src
python 01_preprocess.py && python 02_swald_fit.py && python 03_feature_engineering.py
python 04_loocv_evaluation.py && python 05_statistical_tests.py
python 06_cnn_baseline.py && python 08_dtw_knn_baseline.py
python 09_sensitivity_anticipation.py    # 表2.1 及其注、第七章 v 的年龄效应
python 11_differential_contamination.py  # 表4.1（合成数据，但需 01 的长度池）
python 12_objective_function.py          # 2.2 节（真实数据部分直读原始 CSV）
python fig1_leakage_diagnostic.py        # 图6.2
```

数据无需任何手工配置：`01_preprocess.py` 顶部的列名已按 OSF 原始文件核实并写死
（`Subject / Age / Task / RT`；`Task == 1` 为单选项任务，`Age == 2` 为老年组）。

---

## 脚本 → 论文对应

### 受控实验（论文主体）

| 脚本 | 产出 | 论文位置 | 需要原始数据？ |
|---|---|---|---|
| `10_injection_mechanism.py` | `table3_injection.csv` | **表3.1、图3.1** | 否 |
| `11_differential_contamination.py` | `table4_differential.csv` | **表4.1** | 是（读 `01` 的长度池） |
| `12_objective_function.py` | `table_objective.csv` | **2.2 节**目标函数对照 | 是（真实数据部分） |
| `13_recovery_blindspot.py` | `table5_recovery.csv` | **表5.1** | 否 |
| `14_crossmodel_weibull.py` | `table_crossmodel_dose.csv`<br>`table_crossmodel_ksweep.csv` | **表3.2、表3.3、图3.2** | 否 |
| `15_crossmodel_differential.py` | `table_crossmodel_differential.csv` | **表4.2** | 否 |
| `16_leakage_to_classifier.py` | `table_leakage_classifier.csv` | **表4.3** | 否 |

### 真实数据流水线

| 脚本 | 产出 | 论文位置 |
|---|---|---|
| `01_preprocess.py` | `data/rt_data.pkl` | — |
| `02_swald_fit.py` | `swald_features.csv` | — |
| `03_feature_engineering.py` | `rt_baseline.csv`、`feature_pool_18d.csv` | — |
| `04_loocv_evaluation.py` | `table2.csv`、`predictions_all.csv` | 第七章五路线段 |
| `05_statistical_tests.py` | 终端输出 | 第七章五路线段（p ≥ 0.302） |
| `06_cnn_baseline.py` | `table3.csv`、`predictions_cnn.csv` | 第七章五路线段（1D-CNN） |
| `08_dtw_knn_baseline.py` | `table_dtw_knn.csv` | **6.2 节**（撤回值 0.950、修正值 0.686） |
| `09_sensitivity_anticipation.py` | `table7_sensitivity.csv`<br>`table8_param_effects.csv` | **表2.1** 及其注、第七章 v 的年龄效应 |

### 绘图与公用模块

| 文件 | 说明 |
|---|---|
| `make_figures.py` | 产出 `fig2_mechanism`（**图3.1**）、`fig5_ksweep`（**图3.2**）、`fig3_pre_checks`（**图6.1**）；不需原始数据 |
| `fig1_leakage_diagnostic.py` | 产出 `fig1.*`（**图6.2**）；**需原始数据**，且刻意复现被撤回的带宽实现 |
| `swald.py` | SWald 密度与 MLE，被 `02`、`09`、`10`、`11`、`12`、`13`、`16` import |
| `weibull3.py` | 三参数 Weibull 密度与 MLE，被 `14`、`15` import；箱约束与 `swald.py` 严格同构 |
| `delong.py` | DeLong 检验（Sun & Xu 快速算法），被 `05`、`06`、`08` import |

---

## ⚠️ 两项产出不对应论文任何结果

这两项**保留在库中但论文未使用**，列出以免复现者困惑：

| 文件 | 情况 |
|---|---|
| `07_collinearity_check.py` → `collinearity.csv` | SWald 四特征的 VIF 共线性诊断。研究过程中做过，**本版论文未引用任何结果**；保留作为研究记录 |
| `make_figures.py` 的 `fig4_routes` | 五条技术路线的柱状图。论文第七章保留了该比较的文字披露，但**未收该图**；保留作参考 |

---

## ⚠️ 代码里的编号是内部旧编号

代码中的文件名、变量名与注释沿用开发期的编号，**与论文编号不一致**，未做重命名以免与已产出的
结果文件脱节。读代码注释时请以下表为准：

| 代码中的称呼 | 产出文件 | 论文编号 |
|---|---|---|
| 表3（注入） | `table3_injection.csv` | **表3.1** |
| 表4（差异污染） | `table4_differential.csv` | **表4.1** |
| 表5（恢复） | `table5_recovery.csv` | **表5.1** |
| 表7 | `table7_sensitivity.csv` | **表2.1** |
| 表8 | `table8_param_effects.csv` | 表2.1 的注、第七章 v 的年龄效应 |
| 表2 | `table2.csv` | 第七章五路线段 |
| 表3（CNN） | `table3.csv` | 第七章五路线段的 1D-CNN 行 |
| 表6 | `table_dtw_knn.csv` | 6.2 节 |
| — | `table_objective.csv` | 2.2 节 |
| — | `table_crossmodel_dose.csv` | **表3.2** |
| — | `table_crossmodel_ksweep.csv` | **表3.3** |
| — | `table_crossmodel_differential.csv` | **表4.2** |
| — | `table_leakage_classifier.csv` | **表4.3** |
| 图1 | `fig1.*` | **图6.2** |
| 图2 | `fig2_mechanism.*` | **图3.1** |
| 图3 | `fig3_pre_checks.*` | **图6.1** |
| 图4 | `fig4_routes.*` | 未收入论文 |
| 图5 | `fig5_ksweep.*` | **图3.2** |

---

## 五个脚本内置自检

跑完直接看终端结论，不必自己比对论文：

- **`10_injection_mechanism.py`** —— 打印 H1 是否被否证（上界触及率是否恒为 0）、H2 是否被确认
- **`14_crossmodel_weibull.py`** —— 打印 H1 在 Weibull 上是否同样被否证，以及 B 级扫描中
  失效方向是否随密度行为反转（这是论文 3.3 节的核心判据）
- **`16_leakage_to_classifier.py`** —— 打印「仅用真值差为零的两维」的 AUC 区间下界是否高于 0.5
- **`13_recovery_blindspot.py`** —— 打印准入结论是否由「可识别」翻转为「不可识别」
- **`fig1_leakage_diagnostic.py`** —— 把实测的均值、标准差、不可达占比、长度差同论文 6.2 节
  报告值逐项打勾（应全部 ✓：`172.0/166.7`、`14.5/56.4`、`62.2%`、`21/50/211`）

---

## 与论文一致的关键设定（改动前务必先读）

### 1. RT 剔除下限取 250 ms

依据是**先验的**：反应时研究长期把短于约 200–250 ms 的响应视为预期反应（抢跑）而非真实决策
（Ratcliff & Tuerlinckx, 2002）。**该阈值不是因为「它能让某个反转消失」才选的**——论文 2.2 节
对此有专门说明。

做敏感性分析请跑 `09`，它并列输出 150 / 250 / 300 ms 与「个体化剔除最快 5%」四种条件；
不要直接改主流程阈值。

### 2. 污染扭曲参数的机制是「左尾杠杆」，不是「上界被压低」

**一个直观的错误解释已被本研究否证**：

> **✗ 已否证（H1，约束上界假说）**：污染压低了最快观测，使 T_er 的可估上界随之降低，
> 故 T_er 被低估。受控注入显示，注入 0–8% 的**全部水平上触及上界的比例恒为 0.0%**，
> 且 T_er 与上界的距离不降反升（98.5 → 165.8 ms）——与该解释的预测正好相反。

> **✓ 正确机制（H2，左尾似然杠杆）**：密度在支撑边界处急剧趋零，为容纳一个落在边界附近的
> 极快观测，优化器必须把位置参数推到远低于它之处直至顶到优化下界，同时其余参数沿补偿脊
> 同向抬高——**被扭曲的不是一个参数，是整个参数向量**。

跑 `10` 与 `14` 可分别在 SWald 与 Weibull 上复核这一判别。

### 3. 跨模型验证为什么选三参数 Weibull

不只因为它是另一个分布族，而是**它在支撑下界处的密度行为由形状参数 k 单独控制**：
k < 1 发散（似然无界）、k = 1 有限、k > 1 趋零（与 SWald 同构）。这使机制的**前提条件本身**
成为可开关的自变量——`14` 的 B 级扫描检验的不是「现象是否复现」，而是「该前提是否真的是
它的前提」。

真值取 `gamma=0.30`、`lam=0.45`，使分布均值 0.699 s 与 SWald 仿真的 0.700 s 相当
（同位置参数、同均值、同试次数、同污染区间、同重复次数、同种子），改动会破坏该对照。

### 4. DTW 一律用全窗动态规划，不设 Sakoe-Chiba 带宽

带宽受限时，长度差超过带宽的序列对无法形成可达路径而返回 `inf`；本数据中 **62.2%** 的被试对
落入此情形，而下游 kNN 在 argsort 时把 `inf` 静默当作「最不相似」——**这正是初版 AUC = 0.950
的来源，该结果已撤回**；改用全窗后同口径真实值为 0.686。`08` 用 `assert` 保证距离矩阵不含 `inf`。

**`08` 中变体B 的长度归一化除的是 `max(n_a, n_b)`，不是真实翘曲路径长度。** 二者并不等价
（250 ms 口径下 AUC 分别为 0.542 与 0.678）。本版论文不使用变体B 的任何结果，此处列出是为了
提醒直接调用该脚本的人。

### 5. 实现选择须原样保留

- **RFE 选择器：逻辑回归，`C = 1.0`（`RFE_SELECTOR_C`），`k = 4`**——改为 `C = 0.1` 会使
  DD 组由 0.646 升至 0.735（论文第七章据此说明该比较不具备区分力）
- 主分类器：逻辑回归，不调参（论文 6.1 节闸三的预注册要求）
- **防泄漏**：特征选择与标准化只在每折训练集上拟合（同上）

以下两项**论文未作规定**，是本仓库为保证可复现而固定的实现细节：

- 1D-CNN：`1→8→16`，共 737 参数（论文第七章只提到「1D-CNN」这一路线，未给架构与参数量）
- 随机种子：`20260821`（全部受控实验共用）

### 6. 全流程按「秒」计算

`T_er ≈ 0.20` 才对得上论文。若看到 ~200，是没转秒。

### 7. 6.2 节的 0.686 是 150 ms 口径，须改阈值后重跑

`08_dtw_knn_baseline.py` **没有阈值开关**，只读 `01` 产出的 `rt_data.pkl`，而 `01` 硬编码
250 ms。直接跑 `01`→`08` 得到的是 250 ms 数值（变体A 0.624）；论文 6.2 节引用的 **0.686 是
150 ms 口径**。要复现它：

```bash
# 1) 把 01_preprocess.py 第 44 行改为：RT_MIN_S, RT_MAX_S = 0.150, 4.0
python 01_preprocess.py && python 08_dtw_knn_baseline.py   # → 0.686
# 2) 改回 0.25 并重跑 01，否则后续脚本口径会错
```

这是本仓库唯一需要手工改常量的地方。

---

## 重复次数不可随意调小

`10`–`16` 都接受 `--reps`，但**论文值必须用默认重复次数**：

| 脚本 | 默认（论文所用） | 调小的后果 |
|---|---|---|
| `10_injection_mechanism.py` | 400 | 各污染水平的比率估计不稳 |
| `11_differential_contamination.py` | 200 | **实测：`--reps 3` 时同污染对照组假阳性，结论完全反转** |
| `12_objective_function.py` | **2000** | **实测：200 次时 0.97% 行的均值在 0.050–0.075 间摆动；2000 次时两个种子差异 < 0.001** |
| `13_recovery_blindspot.py` | 200 | **实测：25 次得 r = 0.15，200 次得 r = 0.08** |
| `14_crossmodel_weibull.py` | 400 | k 扫描的触界率在低重复次数下不稳 |
| `15_crossmodel_differential.py` | 200 | 与 `11` 同理 |
| `16_leakage_to_classifier.py` | 200 | AUC 的百分位区间在低重复次数下极不稳，判据（区间下界是否高于 0.5）会失真 |

`--reps` 存在只是为了让人先用小值验证通路是否跑得通，**其输出不可用于任何结论**。

---

## 保留的错误实现

以下两处**故意不修**，用于复核论文的溯源过程：

| 位置 | 内容 |
|---|---|
| `fig1_leakage_diagnostic.py` | 完整复现被撤回的带宽 DTW（Sakoe-Chiba 半径 20，不可达对原样返回 `inf`），即当年给出 AUC = 0.950 的那个距离矩阵。以本脚本重跑得 0.941，0.009 的差异源自原实现中已不可考的细节 |
| `10_injection_mechanism.py` | 保留 H1 的判决性预测，用于展示它如何被数据否证 |

---

## 目录结构

```
├── data/
│   └── README.md          原始数据的下载指令与 md5 校验值（数据本身不入库）
├── src/                   21 个 .py
├── results/
│   ├── tables/            论文各表的直接来源
│   ├── features/          特征矩阵
│   └── figures/           绘图产出
├── requirements.txt
└── UPLOAD_MANIFEST.md     入库清单与逐项复现说明
```

`results/` 下的文件是**运行产出**而非输入，随库上传是为了让读者不跑代码也能核对论文数值。

---

## 环境

Python 3.10+，依赖见 `requirements.txt`（numpy / pandas / scipy / scikit-learn / torch / matplotlib）。

- **绘图脚本需要 matplotlib**；若只想复现表格，可在没有 matplotlib 的环境里跑完 `01`–`16`
- 受控实验中**只有 `16` 需要 scikit-learn**（它要跑 LOOCV 逻辑回归）；`10`–`15` 只依赖
  numpy / pandas / scipy。真实数据流水线的 `04`／`05`／`06`／`08`／`09` 本来就需要 scikit-learn
- **torch 只被 `06_cnn_baseline.py` 使用**；不跑第七章五路线段可以不装
- **路径 A 的五个脚本（`10`／`13`／`14`／`15`／`16`）既不读原始 CSV 也不读 `rt_data.pkl`**，
  这一点已逐个核实；`11` 与 `12` 虽属受控实验，但分别需要 `01` 的长度池与原始 CSV

中文字体在绘图脚本顶部配置：macOS 用 `PingFang SC`，Windows 改 `Microsoft YaHei`，
Linux 改 `Noto Sans CJK SC`。若一个都找不到，脚本会显式 warning 而非静默输出方块图。
