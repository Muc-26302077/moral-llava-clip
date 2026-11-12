# 最终评测结果

[结果对比表](../docs/results.md) · [评测定义](../docs/protocols.md) · [完整复现流程](../docs/workflow.md)

本目录保留最终指标、原始回答和复核所需证据。新生成的日志、checkpoint、试跑和测试结果放在 `results/runs/`，被 Git 忽略。指标文件是历史实测，不通过手工改数更新。

| 目录 | 内容 |
| --- | --- |
| [moralclip-r1](moralclip-r1/test.report.md) | 唯一一组最终双向 R@1；126图、1,260描述；包含逐条检索与矩阵 |
| [baseline](baseline/trained-metrics.json) | 第一阶段、图文微调、训练集众数对照；每组576条预测及指标 |
| [text-guided-vision](text-guided-vision/final-comparison.json) | 同次基线、验证所选第0轮、实际训练第3轮及配对差值 |
| [benchmarks](benchmarks/comparison-ticr.csv) | 原视觉塔、Qwen3-VL-8B、VLGuard Posthoc／Mixed 的完整预测与指标 |
| [moralise](moralise/moralise-metrics.json) | 图文微调基线的7,441条 MORALISE 原始回答、严格评分及格式诊断 |
| [scene-guided-vision](scene-guided-vision/cpu-feasibility.json) | 同场景描述双模式新版的最终 CPU 可行性诊断；真实30场景，不是7B生成 TICR |
| [history](history/final-metrics.json) | 较早的图像标签、三状态和存在性任务最终指标；不同任务单独归档，不用于当前极性方法排名 |

复核全部当前结果（仅 CPU，无需模型）：

```bash
python run_moral_llava.py prepare  # 首次使用执行
python scripts/verify_results.py --output results/runs/verification.json
```

脚本从原始回答重算10组 SMID 指标，包括 A/B 分组；从检索矩阵重算双向 R@1；从7,441条回答重算 MORALISE，逐组与归档指标比对。`baseline/trained` 是历史 batch 8 结果，`text-guided-vision/baseline` 是新模块同环境 batch 1 重跑结果，数值略有不同，不能混作同次配对。

`training-config.json`、预测中的历史 provenance 和 MORALISE `evaluation-data.json` 保留原运行指纹，其中旧机器路径仅作为来源记录。CPU 复算不依赖那些机器路径；重新推理 MORALISE 应按流程下载并重新生成当前机器的路径。必要权重统一放在 [models](../models/README.md)。当前文件完整性指纹见 [manifest.json](manifest.json)。
