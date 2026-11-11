# 文本引导视觉残差

`epoch-0` 为验证选中的零初始化残差，TICR 90.20%；`epoch-3` 为实际训练终轮补充评测，TICR 89.84%。每套只保存加载所需的配置和权重，不保留中间轮次。

两者均依赖 `../moralclip-llava-polarity/` 及固定 MoralCLIP、LLaVA 基模。`guidance.json` 中记录基线 adapter SHA-256，混用其他基线会拒绝加载。运行见 [完整流程](../../docs/workflow.md#5-训练与评测创新结构)。
