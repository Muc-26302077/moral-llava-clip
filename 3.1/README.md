# 3.1 跨模态道德判断评测

输入同一场景的图片或纯文字描述，分别判断 `care`、`fairness`、`loyalty`、`authority`、`sanctity` 五个道德维度，评估两路完整一致性及正确性。

## 新增：六项是/否问答实验（尚无真实模型新成绩）

2026-11-03 新增独立的 `run_binary.py`：保留原 LLaVA 视觉塔、投影器、语言解码器和生成头，将任务改为五个道德维度的标注有无问答，另报 non-moral 辅助项。Care/Harm 任一有标注均表示 care 维度存在，不再区分正负极性；这与旧五维三状态任务分开评分。

```bash
python 3.1/run_binary.py prepare --help
python 3.1/run_binary.py train --help
python 3.1/run_binary.py infer --help
python 3.1/run_binary.py score --help
```

数据生成到新目录 `3.1/data/binary-v1/`，每项另有拆分文件；不覆盖旧数据。正式测试复用原 126 场景和 A/B 分组，以五项图文匹配率宏平均各组 >=60% 为阶段要求，同时检查正确率，常量回答不能作为有效成功。仅完成短 CPU 小样例验证，未启动正式训练或 7B CPU/GPU 推理。具体定义、未来命令、视觉结构核查、第二阶段架构及版本恢复见[新实验方案](docs/BINARY_MORAL_QA_PLAN.md)。原 `run.py` 命令及下文历史结果继续保留。

## 运行方式

在仓库根目录创建并激活 [统一环境](../environment.yml) 后运行：

```bash
python 3.1/run.py evaluate
```

程序自动下载 / 复用指定基模，核验三份基模权重的 SHA-256，加载 `models/moralllava/` 中验证集选中的 LoRA，生成当前机器可用的图片路径协议，然后完成两种模态、两种解码的基模 / 新模型评测。结果写入新的 `results/runs/evaluate-*` 目录，已有记录不会覆盖。

| 命令 | 功能 | 需要 GPU |
| --- | --- | --- |
| `python 3.1/run.py score` | 从已有原始回答重新评分 | 否，Python 3.11+ 标准库即可 |
| `python 3.1/run.py verify` | 核验文件、模型来源、数据与原始结果 | 否，需要 Git LFS 权重 |
| `python 3.1/run.py prepare` | 下载 / 复用基模并核验权重 | 否，需要完整环境与联网或本地缓存 |
| `python 3.1/run.py evaluate` | 真实图片 / 文本推理及评分 | 默认使用 CUDA |

指定输出目录或复用本地快照：

```bash
python 3.1/run.py score --output /tmp/moralllava-score
python 3.1/run.py evaluate --base-model /缓存路径/snapshots/b234b804b114d9e37bb655e11cbbb5f5e971b7a9 --offline
# 只做两场景的短检查，不生成正式指标
python 3.1/run.py evaluate --limit 2 --decoding constrained
```

`--output` 的相对路径从调用时的工作目录解析，默认输出始终在本指标目录下。`--batch-size` 默认为 1，可按显存增大。基模遵循 Hugging Face 的 `HF_HOME` 和 `HF_ENDPOINT`；下载完成后可使用 `--offline`。

## Shell 的迁移方式

`run.sh` 仅将参数原样转发给同目录的 `run.py`，不激活本机虚拟环境、不写死服务器目录，也不要求从特定工作目录启动。以下命令等价：

```bash
python 3.1/run.py score
sh 3.1/run.sh score
PYTHON=/path/to/python sh 3.1/run.sh score
```

迁移时复制仓库并重新创建 Conda 环境；已有基模可通过 `--base-model` 复用，不能直接复制旧机器的虚拟环境。POSIX shell 可在 Linux、macOS 和 WSL 使用；原生 Windows 直接调用 Python。CUDA GPU 运行仍需要匹配的驱动与 PyTorch。

## 数据与模型

原始衍生包共有 2,400 个场景，划分为 2,140 训练、134 验证、126 测试。测试 A/B 各 63 个，固定使用第 0 条描述；它们来自同一数据集。公开副本有 2,398 张图片，排除了两张版权保留的训练图，测试集完整。来源、逐图归属和排除清单见 [data/README.md](data/README.md) 与 [COPYRIGHT.md](data/COPYRIGHT.md)。

| 模型 | 目录 | 说明 |
| --- | --- | --- |
| 图文联合 MoralLLaVA | `models/moralllava/` | 验证集选中的第 2 轮，主要交付模型 |
| 旧图像标签 LoRA | `models/image-lora/` | 继续训练初始化和对照 |
| 基模 | Hugging Face 缓存 | `llava-hf/llava-1.5-7b-hf`，revision `b234b804b114d9e37bb655e11cbbb5f5e971b7a9` |

权重使用 Git LFS。模型训练与条款见 [models/README.md](models/README.md)。

## 已有结果

| 测试组 | 指标 | 旧图像 LoRA | 图文联合 MoralLLaVA |
| --- | --- | ---: | ---: |
| A（63 场景） | 五维完整一致率 | 57.14% | 68.25% |
| B（63 场景） | 五维完整一致率 | 57.14% | 60.32% |
| 全部（126 场景） | 五维完整一致率 | 57.14% | 64.29% |
| 全部（126 场景） | 两种输入均正确率 | 42.86% | 42.06% |

主分析采用合法 JSON token 约束，无效回答计失败；自由生成单独评分。`not_annotated` 代表未标注，不表示已确认道德中性。一致率上升伴随共同错误从 18 增到 28；基模的 100% 约束一致率来自全 `not_annotated` 的退化输出。

原始预测、训练配置和指标按原内容归档在 `results/reference/`，历史文件里的绝对路径不用于新入口。完整方法、统计边界和历史记录见 [docs/README.md](docs/README.md)。

## 目录约定

`data/` 放数据与版权，`models/` 放正式 LoRA，`results/reference/` 放历史证据，新实验写入已被 Git 忽略的 `results/runs/`。代码在 `moral_llava/`，历史文档与旧命令在 `docs/archive/`。`benchmark/` 保持空白，后续再补。
