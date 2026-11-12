# 从模型获取到最终评测

[返回项目首页](../README.md) · [直接查看结果](results.md) · [指标定义](protocols.md)

所有命令在仓库根目录运行。CPU 可以准备数据、复算指标；7B端到端训练和生成评测使用 NVIDIA CUDA GPU。原实验使用 A800 80GB。新模块训练实测峰值已分配显存15.18 GiB；基线两阶段训练的资源需求不同，复现原批量设置建议使用80GB卡。显存较小时，可降低 batch、提高梯度累积并启用梯度检查点，这会改变运行设置，应另外记录结果。

<a id="1-准备环境模型与数据"></a>

## 1. 准备环境、模型与数据

```bash
git lfs install
git clone https://github.com/Muc-26302077/moral-llava-clip.git
cd moral-llava-clip
git lfs pull
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

也可使用 `conda env create -f environment.yml`，然后 `conda activate moral_llava`。推荐 Python 3.11、Linux；PyTorch 需要与本机驱动兼容。已存在合适的 CUDA PyTorch 环境时，可保留该环境并安装其余固定版本依赖。

下载固定版本模型，默认放入被 Git 忽略的 `models/base/`：

```bash
python scripts/download_models.py moralclip
python scripts/download_models.py llava
export MORALCLIP_MODEL="$PWD/models/base/moralclip"
export LLAVA_MODEL="$PWD/models/base/llava"
```

MoralCLIP 为 `anaaa2/moralclip-base`，约599MB；LLaVA 为 `llava-hf/llava-1.5-7b-hf`，权重约14.1GB。下载脚本校验固定 SHA-256；详细版本、已训练组件和许可见 [模型说明](../models/README.md)。网络端点可通过 Hugging Face 标准环境变量配置。

使用已有本地模型时，直接设置上述两个变量，并验证目录完整性：

```bash
python scripts/download_models.py moralclip --verify-local "$MORALCLIP_MODEL"
python scripts/download_models.py llava --verify-local "$LLAVA_MODEL"
```

仓库已经提供可用的 SMID 项目衍生数据：训练2,138图、验证134图、测试126图，每图10条描述。直接从 Git 克隆获取 `data/raw/SMID_moralclip/`，无需另行下载原始 SMID 全集。官方来源、原图获取方式、标签和版权见 [数据说明](../data/README.md)。生成本项目的极性问答数据：

```bash
python run_moral_llava.py prepare
```

输出到 `data/polarity-v1/`：训练9,806问答、验证580问答、测试576问答及 `audit.json`。其中图像训练问答4,903条。原始图像划分与 A/B 组固定；未标注维度不变成负例。数据准备只执行一次，重复执行会拒绝覆盖；如需重建，使用 `prepare --output /path/to/new-data`，后续显式传入该目录。默认流程沿用 `data/polarity-v1/`。

下一步：[MoralCLIP R@1](#2-评测-moralclip-的-r1)。只需评测已有 Moral-LLaVA 时，可跳至 [第4步](#4-评测-ticr-和准确率)。

<a id="2-评测-moralclip-的-r1"></a>

## 2. 评测 MoralCLIP 的 R@1

```bash
python run_retrieval.py --model "$MORALCLIP_MODEL" \
  --expected-model-sha256 4ace10e40a63343dc270afe776a0cc851d28dc36fd13417de07dccb12e4bac2a \
  --device cuda --output results/runs/r1/test.json
```

CPU 推理将 `--device cuda` 改为 `--device cpu`。输出同时包含双向指标、逐条检索结果、相似度矩阵和报告。R@1 采用道德标签相关性：Top-1 与查询共享至少一个道德标签即命中；不等同于严格同图配对检索。定义见 [评测协议](protocols.md)。

仓库仅保留一组最终历史 R@1，位于 [results/moralclip-r1](../results/moralclip-r1/test.report.md)。直接复核它：

```bash
python scripts/score_retrieval.py --evaluation results/moralclip-r1/test.json
```

下一步：[两阶段训练](#3-训练-moralclipllava)。

<a id="3-训练-moralclipllava"></a>

## 3. 训练 MoralCLIP–LLaVA

已训练的推荐基线在 `models/moralclip-llava-polarity/`，可直接跳到第4步。重新训练执行：

```bash
python scripts/train_baseline.py \
  --base-model "$LLAVA_MODEL" --moralclip-model "$MORALCLIP_MODEL" \
  --output results/runs/baseline
```

脚本先用4,903条图像问答适配 `768→4096→4096` 投影器，1轮；再用9,806条图文独立问答训练投影器与语言 LoRA，3轮。MoralCLIP 和语言基座冻结，仅监督答案及 EOS；每阶段按验证答案损失选择权重，再进行完整测试。配置细节见 [结构与训练协议](protocols.md)。

首次检查可加 `--smoke` 并使用新的输出目录，例如 `results/runs/baseline-smoke`，每阶段只更新两步并在验证样本检查重载，不产生正式测试成绩。显存较小时可增加 `--gradient-checkpointing`，并组合 `--batch-size 1 --gradient-accumulation 8`。

正式输出：`projector/` 为第一阶段模型；`adapter/` 为按验证集选择的第二阶段模型；`trained.jsonl` 和 `trained-metrics.json` 为完整测试结果。新运行产生的 checkpoint 和日志保存在被 Git 忽略的 `results/runs/`，不作为历史过程文件提交。

断点续训使用 `run_moral_llava.py train --help` 中的 `--resume-from-checkpoint`；必须沿用相应阶段的原数据、模型和超参数。两阶段脚本要求新的运行目录，避免重复覆盖已完成结果。

下一步：[生成回答和评测](#4-评测-ticr-和准确率)。

<a id="4-评测-ticr-和准确率"></a>

## 4. 评测 TICR 和准确率

使用仓库已训练模型：

```bash
python run_moral_llava.py infer \
  --base-model "$LLAVA_MODEL" --moralclip-model "$MORALCLIP_MODEL" \
  --adapter models/moralclip-llava-polarity --local-files-only \
  --device cuda --precision bf16 --batch-size 1 \
  --output results/runs/baseline-predictions.jsonl
python run_moral_llava.py score \
  --predictions results/runs/baseline-predictions.jsonl \
  --output results/runs/baseline-metrics.json
```

使用新训练模型时，将 `--adapter` 改为 `results/runs/baseline/adapter`。图像路只输入图像和问题；文本路只输入固定 caption 0 和问题。自由贪心生成最多16个新 token，不强制输出合法标签。完整576条问答才可正式评分；`infer --limit 8` 仅用于检查流程。

评分包括五维宏平均 TICR、图像／文本准确率、平衡准确率、图文共同正确率、A/B 分组和有效输出率。TICR 包含共同错误，需与准确率一起看。

原 LLaVA 视觉塔的已微调对照模型在 `models/original-tower/`。评测时指定该 adapter，省去 `--moralclip-model`，其余数据、解码和评分相同。它是已微调对照，不能称作未经训练的原始 LLaVA。若要重训该历史对照，使用 `run_moral_llava.py train`，指定 `--initial-adapter models/image-lora --epochs 3 --learning-rate 2e-5 --batch-size 8 --eval-batch-size 8 --gradient-accumulation-steps 1 --precision bf16 --seed 42 --no-gradient-checkpointing` 及基模、输出目录；不要传入 MoralCLIP 参数。该初始化路径与本项目换塔基线不同。

下一步：[创新结构](#5-训练与评测创新结构)。

<a id="5-训练与评测创新结构"></a>

## 5. 训练与评测创新结构

**同场景语义新版：** 支持真实场景文字指导和无文字时预测语义条件，已完成 CPU 可行性诊断，正式7B收益待测，见[双模式视觉编码方案](scene-guided-vision.md)。以下命令与已发布权重对应旧问题条件版，正式历史成绩不变。

问题文本通过冻结的 LLaVA 输入嵌入提供条件，在 MoralCLIP 第11个编码块内部生成低秩残差的缩放和偏移；196个空间 token 受调节，CLS 不变。新增192,896参数，原模型全部冻结。推理时同样输入问题文本；当前实现不生成场景描述，也不生成整套编码器权重。详见 [结构与训练协议](protocols.md) 和 [成果文案](成果文案.md)。

资源检查、小样本 GPU 检查、正式训练依次运行，使用不同输出目录：

```bash
python run_text_guided_vision.py preflight \
  --base-model "$LLAVA_MODEL" --moralclip-model "$MORALCLIP_MODEL"
python run_text_guided_vision.py smoke \
  --base-model "$LLAVA_MODEL" --moralclip-model "$MORALCLIP_MODEL" \
  --output results/runs/guided-smoke
python run_text_guided_vision.py experiment \
  --base-model "$LLAVA_MODEL" --moralclip-model "$MORALCLIP_MODEL" \
  --output results/runs/guided
```

默认使用仓库的基线 adapter。要接着自己训练的模型实验，三个命令均增加 `--baseline-adapter results/runs/baseline/adapter`。模块会记录并验证实际基线哈希。`preflight` 可在 CPU 上运行，但需要完整本地模型；`smoke` 和 `experiment` 要求 CUDA。

默认训练3轮、rank 32、batch 8、学习率5e-4。每轮在完整验证集选模，图像准确率和合法输出率不能低于基线；再按共同正确率、图像 BA、TICR 选择，零初始化第0轮也参与。选模锁定后评测测试集。结果、原始回答、选模信息和模块权重在运行目录内。

加载已发布的实际训练终轮权重，并与同环境基线成对评测：

```bash
python run_text_guided_vision.py infer \
  --base-model "$LLAVA_MODEL" --moralclip-model "$MORALCLIP_MODEL" \
  --guidance models/text-guided-vision/epoch-3 \
  --output results/runs/guided-final-evaluation
```

也可使用 `models/text-guided-vision/epoch-0` 复现验证所选零残差模型；自己的训练产物使用运行目录中 `selection.json` 记录的 checkpoint。每套残差必须搭配它训练时的 baseline adapter，混用会拒绝加载。当前终轮 TICR 89.84%，同次基线90.20%，尚未证明改善。

下一步：[外部模型与结果复核](#6-外部-benchmark-与-moralise)。

<a id="6-外部-benchmark-与-moralise"></a>

## 6. 外部 benchmark 与 MORALISE

三个外部模型的最终576条回答和指标已经提供，见 [结果目录](../results/README.md)。重新生成 VLGuard 结果，在主环境执行：

```bash
python scripts/download_models.py vlguard-posthoc
python scripts/evaluate_benchmark.py --model vlguard-posthoc \
  --model-dir models/base/vlguard-posthoc --base-model "$LLAVA_MODEL" \
  --output results/runs/vlguard-posthoc.jsonl
```

将两处 `vlguard-posthoc` 改为 `vlguard-mixed` 即可评测 Mixed。使用原始公开权重的显式 HF 映射、原生 padding 和 Vicuna 提示包装；保留原 BF16 RoPE 频率。映射逐张量核验，但不宣称与旧版 Transformers 运行器逐 token 等价。

Qwen3-VL 需要较新的 Transformers，使用独立环境：

```bash
python -m venv .venv-qwen
.venv-qwen/bin/python -m pip install -r requirements-qwen.txt
.venv-qwen/bin/python scripts/download_models.py qwen3-vl-8b
.venv-qwen/bin/python scripts/evaluate_benchmark.py --model qwen3-vl-8b \
  --model-dir models/base/qwen3-vl-8b --output results/runs/qwen3-vl-8b.jsonl
```

三个模型使用同一 SMID 问题与评分规则，但按各自原生视觉处理与对话模板推理；Qwen 图像预算256–1280 visual tokens。完整推理后脚本自动评分，`--limit` 仅检查流程，不输出正式指标。

MORALISE 是另一个评测数据集，采用图文联合输入，S1/S2/S3 不属于 TICR。在主环境下载并评测基线：

```bash
python scripts/evaluate_moralise.py prepare
python scripts/evaluate_moralise.py infer \
  --base-model "$LLAVA_MODEL" --moralclip-model "$MORALCLIP_MODEL" \
  --output results/runs/moralise.jsonl
python scripts/evaluate_moralise.py score \
  --predictions results/runs/moralise.jsonl --output results/runs/moralise-metrics.json
```

下载约1.1GB的数据，按作者固定 revision 连接图片、原文本和规范标签。当前仅基线完成 MORALISE，未报告新增残差模块的该项成绩。严格答案格式及分母见 [评测协议](protocols.md)。

无需模型的最终结果复核：

```bash
python scripts/verify_results.py --output results/runs/verification.json
```

该命令复算10组 SMID 结果、MoralCLIP 双向 R@1 和7,441条 MORALISE 回答，并与归档指标逐组核对。成绩与限制见 [结果简报](results.md)。

<a id="7-按需导出项目组交付数据"></a>

## 7. 按需导出项目组交付数据

当前极性任务的每个 split 可独立导出 `trace.messages` 格式与带校验和的媒体清单，导出产物不提交 Git：

```bash
python scripts/delivery.py export --data data/polarity-v1/train.json \
  --image-root data/raw/SMID_moralclip --run-id smid-polarity-train \
  --output results/runs/delivery/train
python scripts/delivery.py validate --source-root results/runs/delivery/train
```

验证、测试分别将 `train` 改为 `validation`、`test`。完整接收端 token 展开与监督 mask 验收仍需接收方实际 processor／模板，格式检查本身不代表接收端验收完成。既有文档来源记录见 [delivery-sources.json](delivery-sources.json)，不预填未完成的 Intern-S2 评测。

[返回首页](../README.md) · [查看最终结果](results.md)
