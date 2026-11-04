# 多模态大模型评测指标协作项目

## 内容介绍

本仓库提供指标 3.1 的测试数据、已训练模型和评测程序，并提供指标 3.9 的隐私保护图文检索实验代码包。当前 R@1 与极性 TICR 正式协议位于 `codex/ticr-polarity-v1` 分支。

| 指标大类 | 内容 |
| --- | --- |
| [3.1 跨模态道德判断评测](3.1/README.md) | MoralCLIP 道德标签 R@1；Moral-LLaVA 极性 TICR 与准确率 |
| [3.9 隐私保护图文检索评测](3.9/) | LDP-Slicing / CluSanT 隐私扰动与 MoralCLIP 检索实验代码包 |

主要运行文件：

```text
.
├── 3.1/
│   ├── run_retrieval.py     # 原始 MoralCLIP 道德标签 R@1
│   ├── run_polarity.py      # 正式极性问答与 TICR
│   ├── run.py               # 历史三状态任务入口
│   ├── run.sh               # 可选的 shell 入口
│   ├── benchmark/           # 空白预留
│   ├── data/                # 数据、来源与版权
│   ├── models/              # 已训练 LoRA
│   ├── results/reference/   # 已有预测、训练记录和指标
│   ├── moral_llava/         # 训练、推理与评分源码
│   ├── docs/                # 方法、历史文档
│   └── tests/               # 测试
├── 3.9/                     # 隐私保护图文检索实验代码包
├── environment.yml          # Conda 环境
├── requirements.txt         # 环境依赖
└── NOTICE.md                # 版权说明
```

## 运行环境与入口

建议使用 Linux、Python 3.11 和 NVIDIA CUDA GPU。MoralCLIP 原始权重约 599 MB；Moral-LLaVA 推理另需约 14.1 GB 固定 LLaVA 基模，建议显存 24 GB 或以上并预留 25 GB 磁盘空间。

```bash
git lfs install
git clone --branch codex/ticr-polarity-v1 https://github.com/Muc-26302077/moral-llava-clip.git
cd moral-llava-clip
conda env create -f environment.yml
conda activate moral_llava
python 3.1/run_retrieval.py --model /path/to/moralclip-base \
  --expected-model-sha256 4ace10e40a63343dc270afe776a0cc851d28dc36fd13417de07dccb12e4bac2a \
  --output 3.1/results/runs/original-r1/test.json
```

上例测试原始 MoralCLIP 的项目 **R@1**（原道德标签 Hit@1），需先提供完整模型目录；固定权重和下载方式见 [R@1 协议](3.1/docs/MORALCLIP_R1_PROTOCOL.md)。原始模型实测为 **126/126，100.00%**，完整结果和诊断见[基准报告](3.1/results/reference/moralclip-r1-original-20261104/test.report.md)。

Moral-LLaVA 正式极性任务使用 `python 3.1/run_polarity.py ...`，参见 [TICR 协议](3.1/docs/TICR_POLARITY_PROTOCOL.md)。`python 3.1/run.py evaluate` 保留用于历史三状态模型复现，不代表正式极性 TICR。

## 运行测试

### 指标 3.1：跨模态道德判断

MoralCLIP R@1 可不加载模型、在 CPU 上独立核对归档矩阵和逐样本结果（需要 NumPy）：

```bash
python 3.1/scripts/score_retrieval.py --evaluation 3.1/results/reference/moralclip-r1-original-20261104/test.json
```

历史三状态任务的已有结果可仅用 Python 3.11+ 标准库复算：

```bash
python 3.1/run.py score
```

结果位于 `3.1/results/runs/score-*/metrics_*.json`。完整文件与模型来源校验使用 `python 3.1/run.py verify`，需要先获取 Git LFS 权重。

也可以使用 shell：`sh 3.1/run.sh evaluate`，参数与 Python 入口相同。脚本使用自身位置定位项目，支持 Linux、macOS 的 POSIX shell 和 WSL；原生 Windows 可直接运行 Python 入口。实际 GPU 评测环境以 Linux CUDA 为准。

复用本地基模、离线运行及方法说明见 [3.1 README](3.1/README.md)。

### 指标 3.9

实验代码包：[moralclip_3.9_eps20_20261103.zip](3.9/moralclip_3.9_eps20_20261103.zip)。解压后请阅读 `3.9/README.md` 中的参数、依赖与运行说明；模型权重、数据和外部实现需按包内说明单独准备。`3.1/benchmark/` 仍为空白预留。

## 数据与版权

数据来自 [SMID](https://doi.org/10.1371/journal.pone.0190954) 的项目衍生包，126 张测试图完整提供。公开副本按版权记录排除 2 张训练图片，逐图署名和许可见 [数据版权说明](3.1/data/COPYRIGHT.md)。模型提供 LoRA 权重，基模和第三方材料分别遵守其原条款，见 [NOTICE.md](NOTICE.md)。
