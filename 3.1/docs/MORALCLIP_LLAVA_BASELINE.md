# 原始 MoralCLIP 视觉塔接入 LLaVA：极性问答基线

本实验实现用户确认的原始 `anaaa2/moralclip-base` 视觉权重接入，而非普通 OpenAI CLIP 的结构替换。实验分支：`codex/moralclip-polarity-baseline`。完成状态与实际指标以本次实验的 `status.json`、`summary.json`、`report.md` 为准。

## 模型与接入

- LLaVA：`llava-hf/llava-1.5-7b-hf`，revision `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`。
- MoralCLIP：`anaaa2/moralclip-base`，revision `1e337922befebde3f0cadb5749f37ae9521a0751`。`model.safetensors` SHA-256 为 `4ace10e40a63343dc270afe776a0cc851d28dc36fd13417de07dccb12e4bac2a`。本机来源 `/root/autodl-tmp/moralclip-base`。
- 从完整 CLIP checkpoint 严格提取全部 `vision_model.*` 张量；缺失、多余、形状或哈希不匹配即拒绝。加载后逐张量比对原权重（按运行精度转换），训练结束再次比对，排除视觉层随机初始化或被意外更新。
- 原 LLaVA ViT-L/14@336 替换为原始 MoralCLIP ViT-B/16@224，使用该模型的图像预处理器。选取倒数第二层隐藏状态，移除 CLS，得到 196 个 768 维 patch token。
- 新建 `Linear(768,4096) → GELU → Linear(4096,4096)` 多模态投影层，将 patch token 映射到 LLaVA 语言空间。CLIP 文本编码器和 512 维全局检索投影不进入此生成路径。
- 投影器通过 PEFT `modules_to_save` 与语言 LoRA 一起保存、按验证 loss 选模、重载。推理先恢复真实视觉塔及适配结构，再加载包含投影器的 adapter；不允许直接用旧视觉塔加载本次 adapter。

## 固定训练方案

从 LLaVA 基座和全新语言 LoRA 初始化，不续训历史存在性/极性 adapter。视觉塔和语言基座均冻结。

| 阶段 | 训练数据 | 可训练参数 | epochs | 学习率 | 优化步数 |
|---|---|---|---:|---|---:|
| 投影层适配 | 4903 条图像极性问答；290 条验证问答 | 新投影器 | 1 | 1e-3 | 613 |
| 极性问答联合训练 | 9806 条图像/文本独立问答；580 条验证问答 | 投影器、语言 LoRA | 3 | 投影器 1e-4；语言 LoRA 2e-5 | 3678 |

两阶段 batch 8、梯度累积 1、BF16、seed 42、cosine schedule、warmup ratio 0.03、weight decay 0；LoRA r=16、alpha=32、dropout=0.05。仅监督答案及 EOS，问题、图像和 padding 不计入 loss。分别使用该阶段验证答案 loss 最小的 checkpoint；测试不参与选模。

先完成两个阶段各两步 GPU smoke 和混合模态重载推理，再从固定随机初始化重新进行正式投影层适配，接着进行正式极性问答训练。smoke 权重不进入正式训练。

## 测试与解释

沿用 [TICR_POLARITY_PROTOCOL.md](TICR_POLARITY_PROTOCOL.md) 的数据划分和极性输出规则：关怀/伤害、公平/欺骗、忠诚/背叛、权威/颠覆、纯洁/污秽；非该维度标注场景不进入该维度。原 126 测试场景中 87 个符合极性任务，288 个场景—维度对、576 条问答，逐维分母为 72/42/63/55/56。

图像路仅输入图像及问题；文本路仅输入固定 caption0 及问题。自由贪心生成，最多 16 tokens，不强制合法答案或两路一致。训练全部结束后，评测投影层适配模型、联合训练模型，并与训练集众数对照比较。TICR、两路准确率、两路 balanced accuracy 及共同正确率采用五维等权宏平均；另保留逐维、A/B 分组和 micro 统计。A/B 两组 TICR 各 ≥60%，两路整体宏平均 BA 各 >50% 才满足既定要求。报告按场景 bootstrap 2000 次的区间，并独立复算原始回答。

本次数据与历史实验相同，不能声称全新盲测；原始 MoralCLIP 的上游训练数据与本测试集是否重叠尚未核实。旧 LLaVA 视觉塔实验使用不同语言 adapter 初始化和训练阶段，旧结果仅作历史参考，不能将两者差异单独归因于视觉塔。本阶段建立接入基线，不训练视觉编码器，也不实现生成文本反馈视觉编码。原始 MoralCLIP 双向检索 R@1 基准保持其独立定义。

## 运行与断点恢复

在仓库根目录安装 `requirements.txt` 并取得固定版本 LLaVA、MoralCLIP 权重。权重无需存入代码 Git 仓库。可用 `HF_HOME` 指向现有模型缓存。

```bash
screen -L -Logfile moralclip-polarity.screen.log -dmS moralclip-polarity \
  python -u 3.1/scripts/run_moralclip_polarity_experiment.py \
  --output 3.1/results/runs/moralclip-polarity-baseline \
  --moralclip-model /path/to/moralclip-base --local-files-only
```

也可将 `--moralclip-model` 设为 `anaaa2/moralclip-base`，允许联网时去掉 `--local-files-only`。记录完整命令、退出码、源代码哈希、数据哈希和模型来源。`status.json` 每 10 秒更新阶段与训练进度。

SSH 断开后 screen 中进程继续运行。原 runner 和训练子进程都退出后，使用相同命令加 `--resume`：先核验已完成产物、冻结配置和源码哈希，再从最近保存的 Trainer checkpoint 恢复。每个 epoch 保存 checkpoint，未保存阶段的中途进度无法恢复；无 checkpoint 的失败目录需人工核对后另建运行目录。不要对存活任务重复启动。

单模型推理：

```bash
python 3.1/run_moralclip_polarity.py infer \
  --adapter /path/to/experiment/adapter \
  --moralclip-model /path/to/moralclip-base --local-files-only \
  --device cuda --precision bf16 --batch-size 8 \
  --output /path/to/new-predictions.jsonl
python 3.1/run_moralclip_polarity.py score \
  --predictions /path/to/new-predictions.jsonl \
  --output /path/to/new-metrics.json
```

若替换缓存位置，推理必须明确指定 `--moralclip-model`；检查点元数据、config 和预处理哈希必须一致。最终 adapter 包含语言 LoRA 与完整投影器，但仍依赖两个固定版本的外部基模及项目协议代码。
