# 同场景语义引导 V2：验证所选模块

本次两组 GPU 训练均未产生通过质量约束的候选，验证集选择第0轮零残差。本目录用于复现选模结果，不代表训练后取得提升的模型。实际训练权重和优化器恢复状态保留在原运行目录，不替换图文微调基线。

结构支持真实同场景文字条件与图像预测语义条件；rank32新增114,080参数。该所选第0轮的残差输出为零，正式评测中独立模式、文字辅助模式都与同次基线逐条相同。

依赖固定的 LLaVA-1.5-7B、MoralCLIP，以及 `models/moralclip-llava-polarity/` 的既有 LoRA 和投影器。所选权重 SHA-256：`995469215a5930fe9c352870b058cefc373aac59958becc476ee730df8d8744d`。

[结构与训练方法](../../docs/scene-guided-vision.md) · [最终指标](../../results/scene-guided-vision/comparison.json) · [训练与验证核验](../../results/scene-guided-vision/training-summary.json)
