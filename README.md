# 多模态大模型评测指标协作项目

## 内容介绍

本仓库提供指标 3.1 的测试数据、已训练模型和评测程序，并预留指标 3.9。

| 指标大类 | 内容 |
| --- | --- |
| [3.1 跨模态道德判断评测](3.1/README.md) | 同一场景的图片 / 文字道德判断一致性与正确性 |
| [3.9 指标](3.9/) | 预留 |

主要运行文件：

```text
.
├── 3.1/
│   ├── run.py               # 统一运行入口
│   ├── run.sh               # 可选的 shell 入口
│   ├── benchmark/           # 空白预留
│   ├── data/                # 数据、来源与版权
│   ├── models/              # 已训练 LoRA
│   ├── results/reference/   # 已有预测、训练记录和指标
│   ├── moral_llava/         # 训练、推理与评分源码
│   ├── docs/                # 方法、历史文档
│   └── tests/               # 测试
├── 3.9/                     # 空白预留
├── environment.yml          # Conda 环境
├── requirements.txt         # 环境依赖
└── NOTICE.md                # 版权说明
```

## 最短运行步骤

真实模型评测需要 Linux、Python 3.11 和 NVIDIA CUDA GPU，建议显存 24 GB 或以上，默认 batch size 为 1。首次需下载约 14.1 GB 的 LLaVA 基模，建议预留 25 GB 磁盘空间。

```bash
git lfs install
git clone https://github.com/Muc-26302077/moral-llava-clip.git
cd moral-llava-clip
conda env create -f environment.yml
conda activate moral_llava
python 3.1/run.py evaluate
```

程序自动下载或复用固定版本基模，加载已训练的 MoralLLaVA，完成真实图像 / 文本推理并输出指标。结果保存在 `3.1/results/runs/evaluate-*/predictions/metrics_*.json`。

## 运行测试

### 指标 3.1：跨模态道德判断

如果只核对已有结果，无需 GPU、基模或额外 Python 依赖，Python 3.11+ 即可：

```bash
python 3.1/run.py score
```

结果位于 `3.1/results/runs/score-*/metrics_*.json`。完整文件与模型来源校验使用 `python 3.1/run.py verify`，需要先获取 Git LFS 权重。

也可以使用 shell：`sh 3.1/run.sh evaluate`，参数与 Python 入口相同。脚本使用自身位置定位项目，支持 Linux、macOS 的 POSIX shell 和 WSL；原生 Windows 可直接运行 Python 入口。实际 GPU 评测环境以 Linux CUDA 为准。

复用本地基模、离线运行及方法说明见 [3.1 README](3.1/README.md)。

### 指标 3.9

`3.9/` 保留为空，仅含 `.gitkeep`，待补充指标与实现。`3.1/benchmark/` 同样保留为空。

## 数据与版权

数据来自 [SMID](https://doi.org/10.1371/journal.pone.0190954) 的项目衍生包，126 张测试图完整提供。公开副本按版权记录排除 2 张训练图片，逐图署名和许可见 [数据版权说明](3.1/data/COPYRIGHT.md)。模型提供 LoRA 权重，基模和第三方材料分别遵守其原条款，见 [NOTICE.md](NOTICE.md)。
