# 数据获取

本仓库**不再分发原始数据**——它属于 Ratcliff & Vanunu (2022)，
已由原作者在 OSF 公开发布，请自行下载以确保取得权威版本。

## 下载

```bash
curl -L -o data/ratcliff_vanunu_2022.csv \
  "https://osf.io/download/61f043899f195e075a2e22b5/"
```

来源：OSF 项目 `exz4d`（https://osf.io/exz4d/），文件名 `Data.csv`

## 校验

```bash
md5 data/ratcliff_vanunu_2022.csv     # macOS
md5sum data/ratcliff_vanunu_2022.csv  # Linux
```

应得：`3c054348799140297ad6bea6ebc4a004`（804 KB，42,444 行）

校验不符请勿继续——论文全部数值均以该文件为准。

## 列结构

`Subject, Age, Task, RT, Choice, Brightness, Accuracy`

- `Task == 1` 为单选项任务（本文唯一使用的任务），9,996 个观测
- `Age == 1` 年轻组 30 人；`Age == 2` 老年组 29 人
- `RT` 单位为毫秒，实际范围 152–3897 ms
