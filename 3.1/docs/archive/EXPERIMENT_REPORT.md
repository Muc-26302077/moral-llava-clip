# LLaVA-1.5 道德标签迁移实验

本报告属于旧版五维标签问答任务。当前正在使用 [ALIGNMENT.md](ALIGNMENT.md) 中的图文对比学习和道德约束训练入口；本报告 F1 与完全匹配率不能充作新任务 R@1/MAP。

2026-10-18 已完成训练和测试：LLaVA-1.5-7B 经 3 轮 BF16 LoRA 微调，在固定的 126 张测试图上获得 **Micro-F1 77.96%、Macro-F1 75.39%、完全匹配率 55.56%（70/126）**，126 条输出全部通过 JSON 与标签约束校验。

## 任务和数据

输入仅包含图片和统一问题，输出 care、fairness、loyalty、authority、sanctity 五个维度的 `virtue / vice / not_annotated`，另由五维状态约束 `non_moral`。`not_annotated` 指数据未标注，不能理解为确认中性。中文回答是程序按预测标签生成的摘要，没有额外训练视觉解释。

保留原数据的图片级划分：训练 2,140 张、验证 134 张、测试 126 张，共 2,400 张。生成的英文图片描述未进入输入或监督目标。划分之间 ID、文件路径、文件 SHA256 和解码后相同 RGB 的检查通过；没有做感知相似度或近重复图像检索。原论文写 2,401 张，本 ZIP 实际有 2,400 张，当前清单中的图片均存在。

这是把 MoralCLIP 数据标签迁移到 LLaVA 的指令微调基线，没有复现 MoralCLIP 的对比损失，也没有直接迁移论文 CLIP 视觉权重。

## 训练配置

| 项目 | 本次设置 |
| --- | --- |
| 基模 | `llava-hf/llava-1.5-7b-hf` |
| 固定 revision | `b234b804b114d9e37bb655e11cbbb5f5e971b7a9` |
| 设备 | 单张 A800-SXM4-80GB |
| 环境 | Python 3.12.3；PyTorch 2.8.0+cu128；Transformers 4.49.0；PEFT 0.14.0；Accelerate 1.3.0 |
| 精度 | BF16，无 4-bit 量化 |
| LoRA | rank 16、alpha 32、dropout 0.05；语言模型注意力及 MLP 线性层 |
| 可训练参数 | 39,976,960；视觉编码器、投影器和基础语言权重冻结 |
| 训练 | 3 epochs，804 optimizer steps，batch 8，梯度累积 1，seed 42 |
| 优化 | AdamW，学习率 2e-4，cosine，warmup 3%，weight decay 0 |
| 序列 | 最大长度 2048；实际 810–825 tokens，含 576 个图片 token；不截断 |
| 监督 | 仅答案和 EOS；图片、问题和 padding 的 label 为 -100 |
| 选择 | 每轮验证 loss 最低的 checkpoint；测试集不参与选择 |
| 时间 | Trainer 训练计时 1,350.60 秒，约 22.5 分钟，含轮末验证和保存 |
| 显存 | 每 2 秒 nvidia-smi 采样最大 73,073 MiB，约 71.36 GiB；不是精确 allocator 峰值 |

先在独立目录进行了两步短训练并重载 adapter，正式训练随后从干净基模重新开始。训练入口修复了 Transformers 4.49 对 LLaVA mean loss 的梯度累积归一化识别问题，CPU 梯度回归检查通过；本轮正式配置累积步数为 1。

| 轮次 | 步数 | 验证 loss |
| --- | ---: | ---: |
| 1 | 268 | 0.026465 |
| 2 | 536 | 0.026325 |
| 3 | 804 | **0.025624** |

最终导出第三轮 checkpoint-804 对应的 adapter。低验证 loss 本身不等于分类准确率。

## 同一测试集的结果

原模型与微调模型使用同一基模 revision、processor 设置、问题和测试 ID，均为 BF16、无量化、batch 8、贪心生成、最多 256 新 tokens。所有 126 条都保留在分母中。只按原始生成文本解析，不自动修补 JSON，也不丢弃失败样本。

另加入一个不看图片的常量对照：仅统计训练集每个维度的众数，五维均选出 `not_annotated`，因此固定输出 `non_moral=true`。它用于区分“学会格式”和“学会标签判断”。

| 指标 | 原 LLaVA 零样本 | 训练集众数对照 | LoRA 微调 | 相对众数变化 |
| --- | ---: | ---: | ---: | ---: |
| 合规输出率 | 0.00% | 100.00% | **100.00%** | 0.00 个百分点 |
| Micro-F1（11 标签） | 0.00% | 17.22% | **77.96%** | +60.74 个百分点 |
| Macro-F1（11 标签） | 0.00% | 4.30% | **75.39%** | +71.09 个百分点 |
| 样本平均 Jaccard | 0.00% | 30.95% | **72.47%** | +41.52 个百分点 |
| 完全匹配率 | 0.00% | 30.95%（39/126） | **55.56%（70/126）** | +24.60 个百分点 |

微调后的 Micro-precision 为 76.47%，Micro-recall 为 79.51%。11 标签包括 10 个道德极性标签及 `non-moral`。表中变化用未舍入分数相减后转为百分点，可能与显示值直接相减差 0.01。

原模型有 65 条 JSON 解析错误、61 条字段或取值约束错误，原文主要重复标签词表或错误数组格式，未观察到明显截断迹象。原模型端到端零分反映当前输出协议失败，不能直接解释为完全没有图片识别或道德相关能力；微调相对它的提升混合了格式学习与标签识别提升。诊断见 [基线分析](runs/baseline_diagnosis.md)。

## 五个维度

| 维度 | 三类 Macro-F1 | 准确率 |
| --- | ---: | ---: |
| care／关怀 | 85.47% | 85.71% |
| fairness／公平 | 73.67% | 80.95% |
| loyalty／忠诚 | 80.44% | 84.13% |
| authority／权威 | 73.01% | 77.78% |
| sanctity／纯洁 | 80.89% | 82.54% |

这里每维 Macro-F1 固定平均 `virtue / vice / not_annotated` 三类，与上面的 11 标签 Macro-F1 不同。五维平均准确率为 82.22%，它受未标注类别占比影响，只作辅助指标。

authority 和 fairness 是本次相对较弱的维度。单标签中 Subversion 的 F1 为 58.82%、召回率 50.00%（10/20）；Fairness、Betrayal、Cheating 的 F1 分别为 65.00%、66.67%、68.29%。完整逐类指标见 [指标 JSON](runs/finetuned_metrics.json)。

独立重算确认 ID 覆盖、原文解析和指标一致。56 张错误图共涉及 112 项维度错误，其中 95 项涉及 `not_annotated` 边界；详见 [错误分析](runs/error_analysis.md)。

## 结果范围

本次只有一个 seed、一个 126 张测试集，没有做重复实验或显著性检验。五维完全正确为 70 张，仍有 56 张至少一个维度与数据标注不一致。结果用于判断本划分上的标签迁移效果，不代表开放场景伦理推理能力；这些生成分类指标也不能和论文的检索 MAP 直接比较。

后续若继续改训练配置，应在训练／验证集上确定配置，保留新的独立测试集再作最终比较，避免反复根据这 126 张测试图调参。当前数据没有人工视觉解释，不能将程序生成的中文摘要写成经过验证的推理理由。

## 文件与复用

- [模型 adapter 压缩包](runs/moral-llava-1.5-adapter.tar.gz)：约 142 MiB，含 adapter、processor、训练参数和元数据，不含 13.16 GiB 基模。
- 解压目录：`runs/moral-llava-1.5/`，其中 `adapter_model.safetensors` 是 LoRA 权重；仍需配套基模运行。
- [权重和元数据 SHA256](runs/adapter_sha256.json)。
- 本地压缩包已完整下载并解压，14 个文件逐个 SHA256 与服务器一致；[本地校验记录](runs/local_artifact_check.json)。代码与数据的 10 个记录文件也与实验清单哈希一致。
- [微调预测与中文摘要](runs/finetuned.jsonl)、[完整指标](runs/finetuned_metrics.json)。
- [实验配置及输入文件哈希](runs/experiment_manifest.json)、[阶段状态](runs/experiment_status.json)、[训练日志](runs/logs/experiment.log)。
- [部署与推理说明](AUTODL.md)、[项目说明](README.md)。服务器项目目录为 `/root/autodl-tmp/moral-llava`，推理命令与本地目录结构一致。

代码的 34 项单元测试、真实 processor 全样本检查、离线前向、两步反传及保存／重载检查均已通过。本次正式训练和测试 runner 最终退出码为 0。
