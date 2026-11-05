# MoralCLIP-LLaVA 极性问答基线

原始 MoralCLIP ViT-B/16 视觉塔接入 LLaVA-1.5-7B，完成投影层适配和极性问答训练。此目录保存语言 LoRA 与完整 768→4096→4096 投影器；外部冻结视觉塔及语言基座不重复分发。

- LLaVA revision：`b234b804b114d9e37bb655e11cbbb5f5e971b7a9`。
- MoralCLIP revision：`1e337922befebde3f0cadb5749f37ae9521a0751`；源权重 SHA-256：`4ace10e40a63343dc270afe776a0cc851d28dc36fd13417de07dccb12e4bac2a`。
- 本 adapter SHA-256：`318b609354f115162f14de9136005ded3cde31865dda8823559b1e0b79f79e2c`。
- 训练完成 3 轮，按验证 loss 选中第 1 轮 checkpoint-1226；全部 199 个视觉张量保持冻结并逐张量核验。
- 固定测试 TICR 90.21%，图像准确率 88.97%，文本准确率 92.32%（均为五维宏平均）。

必须先重建 MoralCLIP 视觉塔和适配投影器，再加载本 adapter；直接将本 adapter 加到原 LLaVA 视觉塔会发生结构/特征不匹配。请使用项目入口：

```bash
python 3.1/run_moralclip_polarity.py prepare
python 3.1/run_moralclip_polarity.py infer \
  --adapter 3.1/models/moralclip-llava-polarity \
  --moralclip-model /path/to/moralclip-base --local-files-only \
  --device cuda --precision bf16 --batch-size 8 --output predictions.jsonl
python 3.1/run_moralclip_polarity.py score --predictions predictions.jsonl --output metrics.json
```

用 `git lfs pull` 取得本 adapter，并下载两个固定版本外部模型；MoralCLIP 路径也可使用 `anaaa2/moralclip-base`，需联网时去掉 `--local-files-only`。完整方法与使用说明见 [接入协议](../../docs/MORALCLIP_LLAVA_BASELINE.md)，实验与限制见 [结果分析](../../results/reference/moralclip-llava-polarity-20261104/analysis.md)。
