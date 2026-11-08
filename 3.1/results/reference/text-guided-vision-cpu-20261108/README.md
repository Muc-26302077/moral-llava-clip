# 文本引导视觉编码：CPU 阶段实验记录

本目录记录已完成的 CPU 检查和真实图像诊断，不包含新结构的正式 TICR。

- [真实图像诊断报告](cpu-diagnostic.md)：20 条训练、10 条验证；准确率均为 60%，没有观察到改进。
- [完整诊断指标](cpu-diagnostic.json)：训练过程、条件敏感性、耗时与进程内存。
- [样本清单](samples.json)：固定 train/validation 问答，不使用测试样本训练。
- [运行环境预检](preflight.json)：模型及数据核验通过，CUDA 不可见。
- [验证范围与日志索引](verification.json)：最终结构 9 项、协议/报告 13 项检查通过；早期全量 129 项通过。后续完整重查因 2 GiB 资源限制被终止，不宣称最终完整测试集全部跑完。
- [当前成果文案](成果总结-待GPU实验.md)：区分已实现结构、已完成 CPU 诊断、待 GPU 实测结果。

新结构在 MoralCLIP 内部加入由问题文本控制的低秩残差，推理时仍然使用文本条件。
当前条件表示已知评价维度，不包含图像配套描述、自动生成场景描述或标准答案。

原始 MoralCLIP＋LLaVA 基线 TICR 为 90.21%，图像准确率 88.97%，文本准确率 92.32%，共同正确率 85.75%。
原始结果见相邻目录 `moralclip-llava-polarity-20261104/analysis.md`。
GPU 就绪后运行 `bash 3.1/scripts/run_text_guided_gpu.sh`，正式结果和自动回填文案保存于该次运行的 `formal/`。
