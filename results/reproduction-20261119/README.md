# 2026-11-19 真实GPU复测证据

操作步骤见[复现指标指南](../../复现指标指南.md)，数值解释见[本次复测记录](../../本次整理与复测记录.md)。本目录将原先位于忽略目录 `results/runs/` 的最终证据复制为可随Git发布的归档，未重新编辑模型回答或指标。

| 目录 | 内容 |
| --- | --- |
| [main](main/verification.json) | 同次基线、V2独立输入及文字辅助各576条回答；双向MoralCLIP检索、日志、指标及独立复算 |
| [privacy](privacy/verification.json) | 已冻结私有输入下576条新的GPU回答、指标、日志和历史对照核验 |
| [archive](archive/archive-verification.json) | 12组历史SMID、7,441条MORALISE回答与检索的CPU复算 |
| [check](check/run.json) | 原运行环境、模型固定哈希和资产校验记录 |

记录中的绝对路径和 `results/runs/` 路径保留原运行来源；新运行由 `run_metrics.py` 按项目位置生成新的输出目录。主流程基座需按README下载；隐私原图/描述、隐私重训checkpoint、过程优化器状态不在本目录内。
