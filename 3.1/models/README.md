# 已训练的 MoralLLaVA

当前交付的是训练后的语言 LoRA 和配套 processor/tokenizer；和指定基模一起加载可恢复实际评测模型。
基础 7B 权重需单独下载。训练时视觉塔、多模态投影器和基模权重冻结，因此这两个 adapter 包含该训练流程的全部参数更新。

| 版本 | 权重目录 | 选择依据 |
| --- | --- | --- |
| 最新：图文联合 MoralLLaVA | `../runs/crossmodal-joint-v1/epoch-002/` | 验证集五维完整一致率 99/134（73.88%）及质量约束；选中第 2 轮 |
| 旧版：图像标签 MoralLLaVA | `../runs/moral-llava-1.5/` | 3 轮训练后验证 loss 选择，作为初始化与对照 |

基模：`llava-hf/llava-1.5-7b-hf`，revision `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`。
最新版有 39,976,960 个可训练参数，BF16、seed 42；LoRA rank 16、alpha 32。
训练读取 2,140 个训练场景和 134 个验证场景，3 轮；第 2 轮权重由验证集选择，测试标签没有用于选择 checkpoint。
最新版的训练目标含图像/文本答案监督、五维状态 CE 和状态概率 JS；详情见相邻实验报告与 training_metadata.json。

权重通过 Git LFS 保存，先运行 `git lfs pull` 并执行 `python scripts/verify_release.py` 校验。
在 `3.1_evaluation/` 目录安装 `requirements-train.txt`（Python 3.11+，CUDA/PyTorch 版本需匹配环境）。

单图推理可直接使用按 metadata 指定版本的基模：

```bash
python -m moral_llava.infer \
  --image data/raw/SMID_moralclip/test/b10_p138_15.jpg \
  --adapter runs/crossmodal-joint-v1/epoch-002 \
  --output /tmp/moralllava-single.jsonl --precision bf16 --device cuda
```

新机器重新生成跨模态预测时，先下载固定版本的基模快照。
`BASE_SNAPSHOT` 必须指向以 revision 命名的本地快照目录，与已有推理器校验规则一致。

```bash
python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("llava-hf/llava-1.5-7b-hf", revision="b234b804b114d9e37bb655e11cbbb5f5e971b7a9"))'
export BASE_SNAPSHOT=/实际缓存路径/snapshots/b234b804b114d9e37bb655e11cbbb5f5e971b7a9
mkdir -p runs/local-protocol
python -m moral_llava.crossmodal prepare --output runs/local-protocol/protocol.json
python -m moral_llava.infer_crossmodal \
  --protocol runs/local-protocol/protocol.json --base-model "$BASE_SNAPSHOT" \
  --adapter runs/crossmodal-joint-v1/epoch-002 \
  --device cuda --precision bf16 --batch-size 8 --models both --decoding both \
  --output runs/local-evaluation
```

本地协议重新解析当前图片路径，其哈希与历史协议不同；重跑结果应与本地协议一起保存。
不覆盖历史协议与预测，不将两份协议下的预测混合评分。

模型使用须遵守上游 LLaVA/Vicuna/Llama 2 和相关组件的适用条款；LLaVA 的 Apache-2.0 代码许可不能替代模型许可。
固定版本模型卡的原文副本保存在本目录，在线入口为：
<https://huggingface.co/llava-hf/llava-1.5-7b-hf/blob/b234b804b114d9e37bb655e11cbbb5f5e971b7a9/README.md>。
本模型是研究任务上的微调版本，不代表原始 MoralCLIP 模型，也不构成已验证泛化能力的道德裁判系统。
