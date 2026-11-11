# 模型获取与已训练权重

[返回完整流程](../docs/workflow.md)

仓库通过 Git LFS 提供本项目训练得到的必要组件；MoralCLIP、LLaVA 和三个外部对照的完整基模按固定版本下载，避免重复上传第三方大权重。

| 目录 | 用途 | 依赖 |
| --- | --- | --- |
| `moralclip-llava-polarity/` | 已完成图文微调的 LoRA＋完整投影器，主要基线 | 固定 LLaVA＋MoralCLIP |
| `projector-only/` | 第一阶段投影器适配模型，基础阶段对照 | 固定 LLaVA＋MoralCLIP |
| `text-guided-vision/epoch-3/` | 新结构实际训练第3轮残差，约0.77MB | 上述图文微调基线 |
| `text-guided-vision/epoch-0/` | 验证所选零初始化残差 | 上述图文微调基线 |
| `original-tower/` | 原 LLaVA 视觉塔的极性微调对照 | 固定 LLaVA |
| `image-lora/` | 原视觉塔对照训练所需的图像 LoRA 初始化 | 固定 LLaVA；不用于换塔基线 |
| `base/` | 用户自行下载的第三方基模，不提交 Git | 按下表获取 |

| 下载名称 | Hugging Face ID | 固定 revision |
| --- | --- | --- |
| moralclip | [anaaa2/moralclip-base](https://huggingface.co/anaaa2/moralclip-base) | `1e337922befebde3f0cadb5749f37ae9521a0751` |
| llava | [llava-hf/llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf) | `b234b804b114d9e37bb655e11cbbb5f5e971b7a9` |
| qwen3-vl-8b | [Qwen/Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | `0c351dd01ed87e9c1b53cbc748cba10e6187ff3b` |
| vlguard-posthoc | [ys-zong/llava-v1.5-7b-Posthoc](https://huggingface.co/ys-zong/llava-v1.5-7b-Posthoc) | `3529d19cb2e919358b317ac308dba37effcc8b5b` |
| vlguard-mixed | [ys-zong/llava-v1.5-7b-Mixed](https://huggingface.co/ys-zong/llava-v1.5-7b-Mixed) | `757221ed23a5bec4bdaa885a967f37ee9bd5cd21` |

使用 `python scripts/download_models.py <下载名称>`；已有本地目录使用 `--verify-local /path/to/model`。两种方式均校验冻结的权重哈希，外部对照的分片指纹见 [sources.json](sources.json)。MoralCLIP 完整权重 SHA-256 为 `4ace10e40a63343dc270afe776a0cc851d28dc36fd13417de07dccb12e4bac2a`；主要图文微调 adapter 为 `318b609354f115162f14de9136005ded3cde31865dda8823559b1e0b79f79e2c`。

clone 后运行 `git lfs pull`；只有百余字节、以 `version https://git-lfs` 开头的文件是指针，不能用于加载。权重包保留相应 tokenizer、processor、adapter config 和训练元数据。MoralCLIP adapter 必须通过本项目入口先重建视觉塔和投影器，再加载权重，不能直接套在原 LLaVA 视觉塔上。

使用适配器不会改变上游权重条款。LLaVA／Llama 2、MoralCLIP、Qwen 和 VLGuard 分别遵守各自模型卡及上游许可，参见 [NOTICE](../NOTICE.md)。
