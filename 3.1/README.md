# 3.1 跨模态道德判断评测

本指标分别提供原始 MoralCLIP 的道德标签相关检索基准，以及 Moral-LLaVA 的图像／文字极性判断一致性与正确性评测。

## MoralCLIP 检索基准：R@1

项目将原“道德标签 Hit@1”正式命名为 **R@1**：图像查询文本、文本查询图像，首位结果与查询场景共享至少一个道德标签即命中，两个方向分别统计。126 张测试图像、1260 条候选描述，保留本图描述和 non-moral 标签。原始模型双向实测：图到文 **126/126，100.00%**，文到图 **1176/1260，93.33%**；同图配对与跨场景检索仅作单独诊断，不替代本指标。

```bash
python 3.1/run_retrieval.py --model /path/to/moralclip-base \
  --expected-model-sha256 4ace10e40a63343dc270afe776a0cc851d28dc36fd13417de07dccb12e4bac2a \
  --output 3.1/results/runs/original-r1/test.json
```

原始权重版本、下载方式、候选池和独立复算见 [R@1 正式协议](docs/MORALCLIP_R1_PROTOCOL.md)，结果见[原始模型实测表](results/reference/moralclip-r1-bidirectional-20261104/test.report.md)。当前 Moral-LLaVA 极性模型仍保留原 LLaVA 视觉塔，未接入这份 MoralCLIP 权重。

## 正式目标：TICR 五维极性二分类

图文一致率正式命名为 **TICR（Text-Image Consistency Rate）**，不再使用 C@1。正式任务为在相应道德基础已标注的场景中判断正负极性，答案为关怀/伤害、公平/欺骗、忠诚/背叛、权威/颠覆、纯洁/污秽。某维度未标注时不进入该维度任务；不把未标注当作负向标签，也不再训练 non_moral 辅助项。

完整存在性实验已先完成并独立复核：图片准确率 **82.54%**、文本准确率 **79.84%**、TICR **83.33%**、共同正确率 **72.86%**。这是旧“是否涉及”任务的对照结果，不代表极性任务达标。详见[本轮微调分析](docs/PRESENCE_V1_ANALYSIS.md)。随后建立独立极性分支，保留旧数据、模型与运行目录。

公开数据仍按原 2138/134/126 场景划分筛选。筛选后参与极性任务的场景为 1381/79/87 个，训练/验证/测试问答为 **9806/580/576** 条。测试为 288 个图文判断对，五维样本数 72/42/63/55/56；原 39 个 non-moral 测试场景经过检查后不进入极性任务。

```bash
python 3.1/run_polarity.py prepare
# 运行前按仓库环境安装依赖、准备固定基模并 git lfs pull。
screen -L -Logfile ticr-polarity.screen.log -dmS ticr-polarity \
  python -u 3.1/scripts/run_polarity_experiment.py \
  --output 3.1/results/runs/ticr-polarity-v1
# 固定基模已缓存时可追加 --local-files-only；恢复时须确认原进程退出后再加 --resume。
```

流程依次执行专项测试、GPU 两步 smoke、adapter 重载、众数对照、原 LoRA 新提示基线、3 epochs 正式训练、验证选模、完整测试、独立复算和报告。新训练从原图像 LoRA 初始化，不继续存在性权重。TICR 按五维宏平均，A/B 各自达到 60%，同时两路整体宏平均 balanced accuracy >0.5；各维度分母不同，不能沿用旧 189/315 的整数判定。完整协议与限制见 [TICR 极性方案](docs/TICR_POLARITY_PROTOCOL.md)。

## 历史对照：六项是/否存在性问答（完整实验已结束）

2026-11-03 新增独立的 `run_binary.py`：保留原 LLaVA 视觉塔、投影器、语言解码器和生成头，将任务改为五个道德维度的标注有无问答，另报 non-moral 辅助项。Care/Harm 任一有标注均表示 care 维度存在，不再区分正负极性；这与旧五维三状态任务分开评分。

```bash
python 3.1/run_binary.py prepare --help
python 3.1/run_binary.py train --help
python 3.1/run_binary.py infer --help
python 3.1/run_binary.py score --help
```

数据生成到新目录 `3.1/data/binary-v1/`，每项另有拆分文件；不覆盖旧数据。正式测试复用原 126 场景和 A/B 分组，以五项图文匹配率宏平均各组 >=60% 为阶段要求，并要求两路 macro balanced accuracy >0.5。常量回答不能作为有效成功。

真实 7B 小样本训练／重载已完成：训练 32、验证 8、测试 12 场景，图片准确率 71.67%、文字 56.67%、一致率 68.33%、共同正确率 48.33%，不属于完整达标结论。见[小样本报告](docs/BINARY_SMALL_SAMPLE_REPORT.md)。[远端独立复现核验](docs/REMOTE_REPRODUCIBILITY_AUDIT.md)另证实旧主模型 504 条全量回答与旧参考逐条相同；旧指标和新二分类指标分别解释。小样本 adapter 未打包，归档包含固定抽样 ID、原始回答、指标及日志；脚本可按原 ID 重新训练。

完整第一阶段流程使用公开训练 2138、验证 134、测试 126 场景；从旧图像 LoRA 初始化，3 epochs，BF16，batch 4、梯度累积 2、LR 2e-5、seed 42，仅验证 loss 选模。自动顺序执行准备、专项测试、GPU 两步烟雾与重载、众数对照、旧 LoRA 新提示基线、正式训练、选中权重全量测试、评分与报告。它不自动启动第二阶段视觉适配。

```bash
# 先安装统一环境和 git lfs pull；从仓库根目录执行。
# screen 会话独立于 SSH；把 python 替换为已激活环境中的解释器即可。
screen -DmS qav1-full python -u 3.1/scripts/run_binary_full_experiment.py \
  --output 3.1/results/runs/binary-full-v1
# 已缓存固定基模时，可追加 --local-files-only。
screen -ls
screen -r qav1-full
```

每个子阶段单独写日志，`status.json` 每 15 秒记录心跳、子进程 PID 和训练进度；完成后生成 `report.md`、`summary.json` 和各模型完整评分。仅在原 runner 和子进程确认退出后，才使用同一 `--output` 加 `--resume` 从已保存 checkpoint 恢复，避免重复启动。选参和协议在 `frozen_config.json` 中固定，测试结果不用于追加调参。

具体任务定义、视觉结构、第二阶段架构及版本恢复见[新实验方案](docs/BINARY_MORAL_QA_PLAN.md)。原 `run.py` 命令及下文历史结果继续保留。

## 历史三状态模型的运行方式

以下 `run.py` 复现历史 JSON 三状态任务。当前 R@1 使用 `run_retrieval.py`，正式极性 TICR 使用 `run_polarity.py`。

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
| 图文联合 MoralLLaVA | `models/moralllava/` | 验证集选中的第 2 轮，历史三状态模型 |
| 旧图像标签 LoRA | `models/image-lora/` | 继续训练初始化和对照 |
| 基模 | Hugging Face 缓存 | `llava-hf/llava-1.5-7b-hf`，revision `b234b804b114d9e37bb655e11cbbb5f5e971b7a9` |

权重使用 Git LFS。模型训练与条款见 [models/README.md](models/README.md)。

## 历史三状态模型的已有结果

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

### 原始 MoralCLIP 视觉塔接入的极性问答基线

分支 `codex/moralclip-polarity-baseline` 新增真实 MoralCLIP ViT-B/16 视觉权重替换、768→4096→4096 投影器适配，以及投影器＋语言 LoRA 极性问答训练。完整方案、运行、检查点恢复和评测见 [MORALCLIP_LLAVA_BASELINE.md](docs/MORALCLIP_LLAVA_BASELINE.md)。当前训练状态以对应运行目录的 `status.json` 为准；本项独立于历史原 LLaVA 视觉塔的极性结果及原始 MoralCLIP 双向检索 R@1。
