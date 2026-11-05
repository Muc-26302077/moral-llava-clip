# 指标 3.9 交付入口

原始代码包 `moralclip_3.9_eps20_20261103.zip` 包含 50 个文件；逐文件路径、大小、SHA-256 见 [归档清单](../delivery/inventory/3.9-archive-files.json)。包内 `3.9/README.md`、第三方声明、冻结参数和测试结果说明继续保留。它是隐私扰动与检索评测代码，不是 SFT 数据集，不应递归作为训练 source_root。

以下单条命令接收原始数据、冻结私有图文对和 MoralCLIP checkpoint，自动生成可迁移的评测配置、评测日志、运行时间、逐对 CSV 和结果 JSON：

```bash
python 3.9/run_evaluation.py \
  --data /path/to/SMID_moralclip \
  --private-data /path/to/frozen-private-pairs \
  --ckpt /path/to/moralclip-base \
  --output /path/to/new-results --batch-size 32 --device cuda
```

安装依赖前请阅读 ZIP 内 `moralclip_eval/requirements_privacy.txt`。入口仅评估已有私有图文对，不会重做扰动或候选选择；私有数据格式见包内 `privacy_pipeline/manifest_schema.py` 和 `validate_private_dataset.py`。运行时临时解压原代码，覆盖输入、模型和输出路径，不依赖原服务器的绝对路径。

历史测试为 126 张图、1260 对，图到文 R@1 保持率 91.2698%，文到图 77.0985%，双向几何平均 83.8855%。这些是归档报告值，本次未重新生成私有数据或复测；原始私有数据、候选缓存、外部 LDP-Slicing／CluSanT 实现和机器可复算的完整输出未随 ZIP 交付。该项仍需补齐可下载资产及 Intern-S2 评分与日志，详见[验收清单](../delivery/COMPLIANCE.md)。检索模型没有生成式 temperature、max_new_tokens 或 thinking 参数。
