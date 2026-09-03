# GitHub 上传清单（对应论文《越错越显著》V9）

验证方式：建立干净目录，**只放本清单所列文件 + 从 OSF 下载的原始数据**，从零依序运行全部脚本，
产出数值与论文逐项一致。

## 一、必须上传

### 脚本 21 个 —— `src/`

**受控实验（论文主体，证据强度不依赖样本量）**

| 文件 | 论文位置 | 需要原始数据？ |
|---|---|---|
| `10_injection_mechanism.py` | 表3.1、图3.1 | 否 |
| `11_differential_contamination.py` | 表4.1 | 是（`01` 的长度池） |
| `12_objective_function.py` | 2.2 节 | 是（真实数据部分） |
| `13_recovery_blindspot.py` | 表5.1 | 否 |
| `14_crossmodel_weibull.py` | 表3.2、表3.3、图3.2 | 否 |
| `15_crossmodel_differential.py` | 表4.2 | 否 |
| `16_leakage_to_classifier.py` | 表4.3 | 否 |

**真实数据流水线**

| 文件 | 论文位置 |
|---|---|
| `01_preprocess.py` → `02_swald_fit.py` → `03_feature_engineering.py` | 前置，无直接产出 |
| `04_loocv_evaluation.py`、`05_statistical_tests.py`、`06_cnn_baseline.py` | 第七章五路线段 |
| `08_dtw_knn_baseline.py` | 6.2 节（撤回值 0.950、修正值 0.686） |
| `09_sensitivity_anticipation.py` | 表2.1 及其注、第七章 v 的年龄效应 |

**绘图与模块**

| 文件 | 说明 |
|---|---|
| `make_figures.py` | 图3.1、图3.2、图6.1（不需数据） |
| `fig1_leakage_diagnostic.py` | 图6.2（需数据；刻意复现被撤回的带宽实现） |
| `swald.py` | 被 `02`、`09`、`10`、`11`、`12`、`13`、`16` 共 7 个脚本 import |
| `weibull3.py` | 被 `14`、`15` import；箱约束与 `swald.py` 严格同构 |
| `delong.py` | 被 `05`、`06`、`08` 共 3 个脚本 import |

**不对应论文任何结果、但保留**

| 文件 | 理由 |
|---|---|
| `07_collinearity_check.py` | VIF 共线性诊断，研究过程中做过，本版论文未引用；保留作研究记录，README 已明确标注 |

### 结果 19 个 —— `results/`

论文 8 张表的直接来源，另含逐样本预测与特征矩阵。**它们是产出而非输入**，可由脚本重新生成；
上传的价值在于读者不跑代码也能核对论文每一个数。其中 `collinearity.csv` 同样是未被论文引用的
研究记录。

### 说明文件 4 个

`README.md`、`requirements.txt`、`.gitignore`、`data/README.md`

## 二、不上传

| 文件 | 理由 |
|---|---|
| `data/ratcliff_vanunu_2022.csv` | **他人数据，不转发布**；`data/README.md` 给出 OSF 下载指令与 md5 校验值 |
| `data/ratcliff_vanunu_2022_clean.csv`、`data/rt_data.pkl` | `01` 的中间产物，可重新生成 |
| `results/figures/*.png`、`*.pdf` | 绘图脚本产出，可重新生成 |
| `__pycache__/` | 字节码 |

均已写入 `.gitignore`。

## 三、两条复现路径

论文主体是受控实验，**不依赖任何单个数据集的样本量**，故复现可分两级：

```bash
pip install -r requirements.txt
cd src

# 路径 A —— 论文主体，无需下载任何数据
python 10_injection_mechanism.py && python 13_recovery_blindspot.py
python 14_crossmodel_weibull.py && python 15_crossmodel_differential.py
python 16_leakage_to_classifier.py && python make_figures.py

# 路径 B —— 完整复现，先按 data/README.md 取数并校验 md5
python 01_preprocess.py && python 02_swald_fit.py && python 03_feature_engineering.py
python 04_loocv_evaluation.py && python 05_statistical_tests.py && python 06_cnn_baseline.py
python 08_dtw_knn_baseline.py && python 09_sensitivity_anticipation.py
python 11_differential_contamination.py && python 12_objective_function.py
python fig1_leakage_diagnostic.py
```

**注意**：`10`–`16` 的 `--reps` 若调小，结果会偏离论文值（实测：`13` 用 25 次得 r = 0.15、
论文的 200 次得 r = 0.08；`11` 用 3 次时同污染对照组假阳性、结论完全反转；`12` 用 200 次时
均值在 0.050–0.075 间摆动、2000 次时两种子差异 < 0.001）。详见 README。

## 四、五个脚本内置自检

跑完直接看终端结论：

- `10` —— H1 是否被否证（上界触及率是否恒为 0）、H2 是否被确认
- `14` —— H1 在 Weibull 上是否同样被否证；B 级扫描中失效方向是否随密度行为反转
- `16` —— 「仅用真值差为零的两维」的 AUC 区间下界是否高于 0.5
- `13` —— 准入结论是否由「可识别」翻转为「不可识别」
- `fig1_leakage_diagnostic.py` —— 均值/标准差/不可达占比/长度差四项与论文 6.2 节逐项打勾

## 五、编号说明

代码内部沿用开发期编号，与论文编号不一致（如 `table7_sensitivity.csv` 对应论文**表2.1**）。
README 载有完整对照表，读代码注释时以该表为准。
