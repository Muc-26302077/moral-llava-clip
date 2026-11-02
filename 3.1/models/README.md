# 已训练模型

| 目录 | 模型 | 选择依据 |
| --- | --- | --- |
| `moralllava/` | 图文联合 MoralLLaVA LoRA | 第 2 轮；验证集完整一致率 99/134（73.88%）及质量约束 |
| `image-lora/` | 旧图像标签 LoRA | 验证损失选中 checkpoint-804，作为初始化与对照 |

两份权重均为 `adapter_model.safetensors`，通过 Git LFS 保存，并附 processor/tokenizer 和训练元数据。视觉塔、原投影器与基模冻结，LoRA 包含该训练流程的全部参数更新。原来的 `runs/crossmodal-joint-v1/epoch-002/` 和 `runs/moral-llava-1.5/` 已分别整理到上述目录。

在仓库根目录运行 `python 3.1/run.py evaluate`，自动下载 / 复用固定版本基模并加载主要交付模型。模型条款仍分别适用，不能将 LoRA 的分发理解为额外商业授权。

基模：`llava-hf/llava-1.5-7b-hf`，revision `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`。如通过 `--base-model` 复用缓存，路径须指向以该 revision 命名的快照目录；程序同时校验三份基模权重的固定 SHA-256。

训练记录见 [results/reference/runs/crossmodal-joint-v1](../results/reference/runs/crossmodal-joint-v1/)，方法见 [实测报告](../docs/archive/CROSSMODAL_IMPROVEMENT_RESULTS.md)。训练 3 轮，仅更新 39,976,960 个语言 LoRA 参数；测试不参与选参。后续条件蒸馏仅有方案与配置，未列为已训练模型。

基模及其上游 LLaVA / Vicuna / Llama 2 组件遵守相应条款，见 [固定版本模型卡](https://huggingface.co/llava-hf/llava-1.5-7b-hf/blob/b234b804b114d9e37bb655e11cbbb5f5e971b7a9/README.md)、本目录的 `BASE_MODEL_CARD.md` 和 [仓库 NOTICE](../../NOTICE.md)。LLaVA 的代码许可与模型权重条款分别适用。
