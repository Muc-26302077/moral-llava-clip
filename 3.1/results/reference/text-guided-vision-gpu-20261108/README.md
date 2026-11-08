# 文本条件化视觉编码：GPU 实验归档

已完成真实 GPU 训练与完整 SMID TICR 生成评测。验证集最终保留 epoch 0，训练终轮的实际结果单列报告；当前配置未获验证集支持。

- [成果总结](成果总结.md)：已填入实测结果的汇报文案，区分验证选模与训练终轮。
- [结果对比](experiment-report.md)：本次基线、验证选中模型和固定终轮结果，含验证过程与配对区间。
- [原因分析](failure-analysis.md)：可核验观测、条件路径检查及后续假设。
- [机器可读比较](final-comparison.json)、[变化回答](changed-image-answers.json)、[核验记录](verification.json)。
- `formal/`：冻结配置、全部训练权重、验证回答、选模记录、两套主测试回答和原始自动报告。
- `fixed-final-evaluation/`：第 3 轮实际训练权重的补充测试回答、指标及同图换问题的视觉特征检查。
- `fixed-final-evaluation-plan.json`：正式选模和测试之前锁定的补充计划，不用于调整主选模结论。
- `smoke-batch1/`、`smoke-batch8/`：真实小样本训练和重载记录；不作为正式 TICR。
- `full-regression-133.log`、`manifest-sha256.json`：回归验证与归档校验。

汇报请使用本目录的成果总结.md。formal/ 内自动生成的原文仅对应验证集选中的主比较。

运行入口：`bash 3.1/scripts/run_text_guided_gpu.sh`；补充评测入口为 `3.1/scripts/evaluate_guided_fixed_final.py --run-root <该次运行目录>`，须先固定计划并完成主实验。原始执行目录保留在 `results/runs/text-guided-gpu-20261108/`；冻结配置保留原执行绝对路径，归档不改写来源信息。

原基线历史报告位于相邻目录 `moralclip-llava-polarity-20261104/analysis.md`。本次采用同一环境重跑基线进行配对比较。TICR 为五维宏平均，非池化比例；测试集曾用于历史实验。当前实验不提供 MORALISE 或 R@1 结果。
