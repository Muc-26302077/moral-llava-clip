# MoralCLIP 数据迁移到 LLaVA-1.5

2026-11-02：下一轮已设计为“保留图像/文本真实标签监督，使用可靠文本到图片的条件蒸馏”，目标为共同完整正确率 >=80%、完整一致率 >=90%。[实验方案及备份恢复说明](CROSSMODAL_CONDITIONAL_DISTILLATION_PLAN.md)与三组设计配置已保存；新预测头和训练器尚未实施，也未启动正式训练。改动前源码、数据、当前主线权重及评测记录已归档备份。

最新阶段目标为**同场景图像／纯文本道德判断一致性**。已从旧图像 LoRA 继续进行成对监督和状态一致性训练，固定测试组 A 达 43/63（68.25%）、B 达 38/63（60.32%），均达到当前 60% 点估计门槛；共同正确场景 54→53、一致但错误 18→28，存在准确率取舍。完整方法、原始基模对照及边界见 [最新实测总结](CROSSMODAL_IMPROVEMENT_RESULTS.md)，整体历史见 [进度记录](PROGRESS_CROSSMODAL.md)。

以下保留历史 **图文匹配 + 道德约束检索** 实验。双路编码、对比损失、道德损失、训练和检索评估见 [ALIGNMENT.md](ALIGNMENT.md)，逐行命令见 [alignment_commands.txt](alignment_commands.txt)。历史 λ=0 与 0.4 的正式对照已完成，两组均训练10轮；126张测试图、1260条候选描述上的结果如下。这些检索指标不等同于图像／文本独立判断的一致率。

| 新版方案 | 配对 R@1 | 道德 MAP | 跨图道德 MAP |
| --- | ---: | ---: | ---: |
| CLIP式图文对比基线（λ=0） | 63.49% | 43.48% | 40.64% |
| MoralCLIP式道德约束迁移到 LLaVA（λ=0.4） | 55.56% | 44.05% | 41.36% |

λ=0.4 组使用论文 MoralCLIP 的道德相似度约束思路并迁移到 LLaVA，不是论文原始 MoralCLIP 模型或完整复现。这次道德 MAP 提高 0.57 个百分点，配对 R@1 降低 7.94 个百分点；不能说加约束后全面优于基线。两组均按验证道德 MAP 选择权重，单次 seed 的小幅提升尚未做显著性检验。详见[新版实验报告](outputs/alignment_experiment_report.md)、[结果图](outputs/alignment_experiment_results.png)和[仅推理命令](alignment_inference_commands.txt)。

用户上传图片的[单图检索图](outputs/alignment_user_retrieval.png)也已生成，但当前候选库里的前五条描述与画面不贴切；它展示了实际失败结果，不能当作准确率或成功演示。


下面保留的是旧版五维标签问答基线记录。旧目标：输入一张图片，输出五个道德维度的结构化判断，以及对应的中文标签摘要。A800 上的 3 轮 BF16 LoRA 训练及分类测试已完成：126 条输出均有效，micro-F1 为 0.7796，exact match 为 55.56%。详细实验结果见 [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md)，旧服务器目录和日志见 [AUTODL.md](AUTODL.md)。这些分数不代表新版图文检索效果。

## 采用的方法

这里实现的是 **LLaVA-1.5-7B 的标签监督指令微调基线**。使用 Hugging Face 转换版 `llava-hf/llava-1.5-7b-hf`，本轮固定 revision `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`，采用普通 BF16 LoRA。代码也支持 4-bit QLoRA。该加载方式需要 HF 转换格式，不应直接替换为原仓库的 `liuhaotian/llava-v1.5-7b` 文件。

MoralCLIP 在 CLIP 的对比学习中加入基于道德标签重合度的对齐约束。论文使用的 ViT-B/16 与 LLaVA-1.5 的 CLIP ViT-L/14@336 不同，不能直接互换视觉权重。本项目把已有道德标签转成图像问答监督，未复现 MoralCLIP 的对比损失。

训练时冻结视觉编码器、视觉投影器和基础语言模型权重，只训练语言模型注意力与 MLP 线性层的 LoRA。训练与推理使用相同问题及 `USER: <image>\n… ASSISTANT:` 格式；只有答案和结束符参与损失计算，图片 token、问题和 padding 均被屏蔽。

## 数据与标签含义

已处理根目录的 `SMID_moralclip.zip`，保留原始图片级划分：

| 划分 | 图片数 | 转换文件 |
| --- | ---: | --- |
| 训练 | 2,140 | `data/processed/train.json` |
| 验证 | 134 | `data/processed/validation.json` |
| 测试 | 126 | `data/processed/test.json` |
| 合计 | 2,400 | |

主清单 `smid_manifest.json` 包含训练和验证的 2,274 条记录；测试的 126 条位于独立的 `test/smid_manifest.json`，转换时已一起读取。每张图片有 10 条生成英文描述，均未写入训练问题或答案，推理也不使用这些描述。

已检查所有图片存在且可解码、标签合法、ID 与路径不重复，以及文件 SHA-256 没有跨划分重复。该检查不等同于近似图像去重。压缩包中的 `missing_images.json` 是历史记录，所列 ID 不在当前清单中；当前 2,400 条样本没有缺图。详细结果见 `data/processed/audit.json`。

五个字段与原始标签的对应关系：

| 字段 | `virtue` | `vice` |
| --- | --- | --- |
| `care` | Care／关怀 | Harm／伤害 |
| `fairness` | Fairness／公平 | Cheating／欺骗 |
| `loyalty` | Loyalty／忠诚 | Betrayal／背叛 |
| `authority` | Respect／尊重权威 | Subversion／颠覆权威 |
| `sanctity` | Sanctity／纯洁 | Degradation／污秽 |

某个维度两极标签都不存在时，输出 `not_annotated`，含义是“在该数据集中未标注”，**不能解释为已确认中性**。按照本数据的标签约束，五维都未标注时，`non_moral` 为 `true`；其余情况为 `false`。

下面只是输出格式示例，不代表对某张图片的真实预测：

```json
{
  "care": "virtue",
  "fairness": "not_annotated",
  "loyalty": "not_annotated",
  "authority": "not_annotated",
  "sanctity": "not_annotated",
  "non_moral": false
}
```

数据没有人工编写的视觉道德解释，因此模型的训练目标只包含上述判断。推理结果的 `answer` 是程序根据预测生成的中文标签摘要，不是经过监督训练的视觉推理理由。若要输出结合画面证据的自由文本解释，需要另行构建和审核解释标注。

## 文件结构

```text
moral_llava/
  schema.py       统一标签、问题、JSON 校验与中文摘要
  prepare.py      解压、图片与标签审计、转换原始划分
  train.py        LoRA／QLoRA 训练、验证集选择 checkpoint
  infer.py        原模型或 adapter 的批量／单图推理
  evaluate.py     严格覆盖全部样本的生成结果评估
tests/            无需模型权重的单元测试
requirements-data.txt
requirements-train.txt
requirements-autodl.txt  复用当前 AutoDL 镜像的 PyTorch
scripts/                AutoDL 环境激活与依赖／模型检查
data/raw/SMID_moralclip/
data/processed/
runs/             checkpoint、预测、指标、日志与运行状态
```

## 本地准备与检查

要求 Python **3.11 或更新版本**。下面命令均在项目根目录执行；数据准备只依赖 Pillow，无需 PyTorch 或 GPU。当前工作区已经生成数据，重复运行会检查原始解压文件是否与压缩包一致，不覆盖内容不同的原始文件。

```bash
python -m pip install -r requirements-data.txt
python -m moral_llava.prepare
python -m unittest discover -s tests -v
```

自定义路径可使用 `prepare --archive /path/SMID_moralclip.zip --raw-dir data/raw --output-dir data/processed`。转换后的图片路径相对于 `data/raw/SMID_moralclip`，搬到服务器时保持这个目录关系即可。原始数据、转换数据、模型权重及运行结果已配置为 Git 忽略项。

## GPU 服务器运行

本次正式实验在 `/root/autodl-tmp/moral-llava` 完成。`runs/experiment_runner.py` 已串行完成训练、微调后的测试集推理和评估；阶段状态见 `runs/experiment_status.json`，日志见 `runs/logs/experiment.log`，GPU 采样见 `runs/logs/experiment_gpu.csv`，固定版本及推理配置见 `runs/experiment_manifest.json`。这些路径均相对于服务器项目目录。

当前 AutoDL A800 实例请按 [AUTODL.md](AUTODL.md) 操作，复用其 PyTorch 2.8。另行搭建环境时，建议使用 Linux、NVIDIA CUDA 和独立的 Python 3.11／3.12 虚拟环境，并按服务器驱动选择匹配的 PyTorch CUDA wheel；下面的 `requirements-train.txt` 用于这种新环境。

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-train.txt
```

默认基模约 7B 参数，已完整缓存并核验权重。A800 上的两步 BF16 短训练和正式训练均已完成。正式运行中，`nvidia-smi` 每 2 秒采样的最大显存占用为 73,073 MiB（约 71.36 GiB），这是采样最大值，并非 PyTorch 精确峰值。`--load-in-4bit` 使用 CUDA bitsandbytes NF4 QLoRA；不加该参数则为 LoRA。`--local-files-only` 使用本地缓存；也可用 `--model-id` 指定本地 HF 模型目录。

### 训练

本轮训练已完成，下面保留对应的离线命令。重新训练时必须将 `--output-dir` 改为新的空目录，例如 `runs/moral-llava-1.5-rerun`，以保留已有结果：

```bash
cd /root/autodl-tmp/moral-llava
source scripts/activate_autodl.sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m moral_llava.train \
  --revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9 \
  --local-files-only \
  --train-data data/processed/train.json \
  --validation-data data/processed/validation.json \
  --image-root data/raw/SMID_moralclip \
  --output-dir runs/moral-llava-1.5 \
  --epochs 3 \
  --precision bf16 \
  --batch-size 8 \
  --gradient-accumulation-steps 1 \
  --eval-batch-size 8 \
  --no-gradient-checkpointing \
  --num-workers 4 \
  --seed 42
```

本轮为 3 epochs、每设备 batch 8、梯度累积 1（有效 batch 8）、验证 batch 8、BF16，关闭梯度检查点。其余采用默认值：学习率 `2e-4`、最大长度 2048、LoRA rank 16／alpha 32／dropout 0.05、seed 42。代码校验预处理后的 576 个图片 token；超长样本会报错，不截断图片或答案。

每个 epoch 计算验证损失并保存 checkpoint，最终导出验证损失最低的 adapter、processor 和 `training_metadata.json`。本次完成 804 steps，训练耗时 1,350.5993 秒（约 22.5 分钟）；三轮验证 loss 依次为 0.026465144008398056、0.026325300335884094、0.02562384493649006，最佳为 `checkpoint-804`。元数据记录基模版本、运行参数、依赖版本和最佳 checkpoint。测试集不参与模型选择。仅在继续已有训练时使用 `--resume-from-checkpoint`。

正式训练前，已在独立目录 `runs/smoke-bf16` 完成 `--max-steps 2` 短训练、验证与 adapter 保存，并重载完成单图生成（`runs/smoke_reload.jsonl`）。正式训练从基模重新开始，没有恢复该短训练的 checkpoint。

### 零样本与微调后推理

原始 LLaVA-1.5 在 126 张测试图上的零样本推理与评估已完成，预测和指标分别为 `runs/baseline.jsonl`、`runs/baseline_metrics.json`。对应的离线推理命令为：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m moral_llava.infer \
  --revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9 \
  --local-files-only \
  --data data/processed/test.json \
  --image-root data/raw/SMID_moralclip \
  --output runs/baseline.jsonl \
  --precision bf16 \
  --batch-size 8 \
  --max-new-tokens 256
```

runner 已加载微调后的 adapter，使用同一测试集与推理参数完成推理；对应命令为：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m moral_llava.infer \
  --revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9 \
  --local-files-only \
  --data data/processed/test.json \
  --image-root data/raw/SMID_moralclip \
  --adapter runs/moral-llava-1.5 \
  --output runs/finetuned.jsonl \
  --precision bf16 \
  --batch-size 8 \
  --max-new-tokens 256
```

两者均为贪心生成、最多生成 256 tokens、最大长度 2048，保持相同基模版本、无量化、BF16、问题和生成参数。加载本项目 adapter 时会优先读取其记录的基模 commit 和保存的 processor。批量推理只读取 human 问题与图片，忽略数据内的 gpt 答案。

单张图片使用同样的问题和输出协议：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m moral_llava.infer \
  --revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9 \
  --local-files-only \
  --image /path/to/image.jpg \
  --adapter runs/moral-llava-1.5 \
  --output runs/single.jsonl \
  --precision bf16
```

每行 JSONL 保留 `id`、`raw_prediction`、结构化 `prediction`、`parse_error` 和中文摘要 `answer`。只接受完整 JSON 或单个完整 JSON 代码块，标签不合法或格式错误时会保留原文并标记失败，不自动修补结果。

### 评估

```bash
python -m moral_llava.evaluate \
  --data data/processed/test.json \
  --predictions runs/baseline.jsonl \
  --output runs/baseline_metrics.json

python -m moral_llava.evaluate \
  --data data/processed/test.json \
  --predictions runs/finetuned.jsonl \
  --output runs/finetuned_metrics.json
```

本次 126 张测试图的结果如下。多数类基线仅使用训练标签为每个维度选择出现最多的状态，没有使用验证或测试标签选择预测。

| 方法 | micro-F1 | Exact match |
| --- | ---: | ---: |
| 原始 LLaVA-1.5 零样本 | 0.0000 | 0／126（0%） |
| 训练标签多数类基线 | 0.1722 | 39／126（30.95%） |
| BF16 LoRA 微调 | 0.7796 | 70／126（55.56%） |

微调后输出有效率为 126／126（100%），macro-F1 为 0.7539145311004785，样本平均 Jaccard 为 0.7247354497354497。零样本的 126 条输出均未通过本项目的结构化协议校验，按下述严格规则计零，不能将该零分单独解释为模型没有视觉道德判断能力。完整精度和逐类结果见 `runs/finetuned_metrics.json`，详细分析见 [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md)。

指标文件及运行日志已保存到本地。adapter 本地备份已完成：`runs/moral-llava-1.5-adapter.tar.gz` 已完整下载并解压到 `runs/moral-llava-1.5`，14 个文件的 SHA256 均与服务器一致，校验报告为 `runs/local_artifact_check.json`。

评估必须覆盖所有测试 ID，缺失、重复或多余 ID 都会报错。指标从 `raw_prediction` 重新解析，不信任缓存的结构化字段。格式错误和约束错误仍计入分母，其 exact match、Jaccard 及五维正确数为零，真实标签计为漏检。

报告包括：有效输出率；原始 11 个标签（10 个道德极性加 `non-moral`）的 micro／macro precision、recall、F1；逐样本 Jaccard；整条结果 exact match；每个维度的 accuracy 和三类 macro-F1。macro 指标使用固定类别集合，零分母类别记为零；维度三类为 `virtue`、`vice`、`not_annotated`。

这些是生成式分类指标，与论文的排序检索 MAP 不直接可比。当前测试集仅 126 张，结果只能反映这一划分上的表现，不能据此声称已经复现论文或验证广泛泛化能力。

## 当前验证范围

已完成真实数据转换、图片解码与文件哈希检查，以及无模型依赖的标签、转换、序列屏蔽、推理输出和评估测试。训练和推理 CLI 可以在未安装重量级依赖时显示帮助。

当前 Mac 环境没有安装 PyTorch／Transformers 等训练依赖。AutoDL A800 上已确认 PyTorch／Transformers／PEFT 导入、CUDA 和 BF16 可用，34 项单元测试及 CPU 梯度累积回归检查通过。准备阶段已完成基模缓存与 SHA256 核验、训练／验证全部样本的真实 processor 检查，以及单图离线 BF16 GPU 前向。

随后已通过两步短训练、验证、adapter 保存与重载生成，并完成测试集零样本基线、正式 3 轮训练及微调后的生成式测试集评估。结果仅反映当前 126 张测试图的划分；详情见 [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md) 和 [AUTODL.md](AUTODL.md)。

## 参考

- MoralCLIP 论文：<https://arxiv.org/abs/2506.05696>
- HF LLaVA-1.5-7B：<https://huggingface.co/llava-hf/llava-1.5-7b-hf>
- 原始 LLaVA 项目：<https://github.com/haotian-liu/LLaVA>
