# QA v1 小样本实测报告

2026-11-03（Asia/Shanghai）。已完成真实 LLaVA-1.5-7B GPU 基线推理、小样本 LoRA 训练、保存重载和评分。流程可运行，小样本训练改善了正确率；尚无完整测试集达标证据，不能进入第二阶段视觉适配。

使用项目已有 `.venv/bin/python`，其中 PEFT 0.14.0、Transformers 4.49.0 和 PyTorch 2.8.0 均可用，无需安装依赖。此前默认 Python 的缺少 `peft` 报错属于环境选择问题。本次 QA v1 专项测试 **15/15 通过**，包含随机小模型的混合图文反传、纯文字前向和视觉冻结检查。

已转换服务器完整原数据：训练 2,140 场景／25,680 条，验证 134 场景／1,608 条，测试 126 场景／1,512 条；原划分与 A/B 分组保留。转换审计见 [audit.json](../results/reference/binary-small-v1/data_audit_server.json)。

本轮固定 seed 42，先按场景抽样，再保留每场景六任务、两种模态的全部问答。训练 32 场景／384 条，验证 8 场景／96 条，测试 12 场景／144 条（冻结组 A/B 各 6 场景）。三部分场景无交集，抽样不依据标签或预测。图片问题不含 caption，文字使用固定 caption 0。本次测试每个主任务均包含正负样本。

基模固定为 `llava-hf/llava-1.5-7b-hf` 的 `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`。从旧图像 LoRA 继续训练 1 epoch、48 个优化步骤，BF16，batch 4、梯度累积 2、学习率 2e-5。仅语言 LoRA 更新，原视觉塔与投影器冻结。验证 loss 为 **0.230737**；保存后重新加载 adapter 完成测试。全部子命令耗时合计约 **145 秒**，其中训练命令约 89 秒，不包含此前的数据准备和环境核查。

下表只统计五个主任务的宏平均；`non_moral` 辅助项不进入主分数。每种模态 60 个主任务回答，共 60 个图文判断对；共同正确要求图片与文字同时答对。

| 方法 | 图片准确率 | 文字准确率 | 图文一致率 | 共同正确率 | 图片 balanced accuracy | 文字 balanced accuracy |
|---|---:|---:|---:|---:|---:|---:|
| 完整训练集众数对照 | 61.67% | 61.67% | 100.00% | 61.67% | 50.00% | 50.00% |
| 旧图像 LoRA + 新版问题 | 45.00% | 45.00% | 98.33% | 45.00% | 50.00% | 50.00% |
| 小样本 SFT 1 轮 | **71.67%** | **56.67%** | 68.33% | 48.33% | **69.64%** | **60.54%** |

旧 LoRA 的图片路 60/60 回答“是”，文字路 59/60 回答“是”，另一个为严格协议下无效的“是。”。因此，它的高一致率来自近乎常量输出，正确性防退化条件未通过。SFT 后主任务图片回答为 36“是”／24“否”，文字为 51“是”／9“否”；全部六任务共 **144/144 输出合法**，但文字仍明显偏向“是”。

训练后 A 组图文一致 **23/30（76.67%）**，B 组 **18/30（60.00%）**。整体两路 balanced accuracy 均大于 0.5，但这只是 12 场景结果，B 组也恰好在阈值上。评分器明确记录 `complete_grouped_benchmark=false`、`eligible_for_stage2=false`。不能把该结果写成完整 126 场景通过。

图片准确率由 45.00% 提高到 71.67%，文字由 45.00% 提高到 56.67%，说明小样本训练产生了有效变化；共同正确仅由 45.00% 提高到 48.33%，仍低于众数对照的 61.67%。当前瓶颈包括文字判断偏阳性和图文共同正确率不足。后续应在训练／验证集上确定正式训练方案，再做完整评测；本轮没有根据测试结果追加调参。

独立核验重新计算了准确率、balanced accuracy、一致率和共同正确率，均与评分文件一致。训练前后 448 个语言 LoRA 张量全部发生变化，保存张量均为有限值；原 adapter SHA-256 未变。旧结果与权重保留，本次没有提交或推送 Git。

可核查产物：

- [实验清单：命令、退出状态、耗时、抽样 ID、数据哈希](../results/reference/binary-small-v1/manifest.json)
- [汇总指标](../results/reference/binary-small-v1/summary.json)与[独立验证](../results/reference/binary-small-v1/verification.json)
- [旧 LoRA 原始回答](../results/reference/binary-small-v1/baseline.jsonl)与[训练后原始回答](../results/reference/binary-small-v1/trained.jsonl)
- [训练后完整评分：逐任务、辅助项、A/B](../results/reference/binary-small-v1/trained-metrics.json)
- [训练日志](../results/reference/binary-small-v1/train.log)与[adapter 元数据](../results/reference/binary-small-v1/training_metadata.json)
- [可复现的小样本运行脚本](../scripts/run_binary_small_sample.py)

公开归档包含原始回答、指标和日志；小样本 adapter 未打包。可先 `python 3.1/run_binary.py prepare`，再执行 `python 3.1/scripts/run_binary_small_sample.py`，脚本默认复用归档的抽样 ID，在公开数据上重训该小样本。`--local-files-only` 可选，基模缓存由 Hugging Face 环境变量控制。
