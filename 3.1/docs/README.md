# 方法与历史记录

- [最新实测报告](archive/CROSSMODAL_IMPROVEMENT_RESULTS.md)：方法、验证选择、测试结果及质量取舍。
- [评测协议](archive/CROSSMODAL_EVALUATION.md)：固定 A/B 划分和指标定义。
- [训练方案](archive/CROSSMODAL_IMPROVEMENT_PLAN.md)：图像 / 文本联合监督与 JS 一致性。
- [历史进度](archive/PROGRESS_CROSSMODAL.md)：不同阶段的目标与结果。
- [后续条件蒸馏方案](archive/CROSSMODAL_CONDITIONAL_DISTILLATION_PLAN.md)：仅有计划，尚未实现或正式训练。

历史原始文档、旧脚本、检索实验和旧依赖放在 `archive/`。其中路径与命令反映当时服务器的目录，只作研究过程参考；当前可运行入口以 [主 README](../README.md) 为准。全量重训仍需自行按许可取得公开副本排除的两张训练图片。

## 目录调整

| 原位置 | 当前位置 |
| --- | --- |
| `runs/crossmodal-joint-v1/epoch-002/` | `models/moralllava/` |
| `runs/moral-llava-1.5/` | `models/image-lora/` |
| 其余 `runs/` 记录 | `results/reference/runs/` |
| `outputs/crossmodal-joint-v1-evaluation/` | `results/reference/evaluation/` |
| 散落的历史 Markdown、命令、依赖及辅助脚本 | `docs/archive/` |

历史协议、预测、训练元数据和权重保留原内容。`scripts/verify_release.py` 按新位置核验原有哈希和验证集选择，不要求复原旧服务器路径。
