# 项目组共享与数据交付

本目录按项目组 2026-11-05 通知整理已有的评测代码、数据和 SFT 数据。当前模型仍为 MoralCLIP + LLaVA；这里完成格式转换、材料归档和本项目复现验证。项目组的 Intern-S2 联测、Qwen3.5-VL 接收方校验分别在信息表中标明状态，不将其写成已完成的实验。

- [全部已上传文件与分支说明](INVENTORY.md)：默认 main、最新模型分支、历史分支和 3.9 ZIP 内部文件均有清单。
- [规则逐项对应与验收范围](COMPLIANCE.md)：区分硬性格式、建议项、本项目验证与接收方验证。
- [数据与评测信息表](INFORMATION.md)：数量、token、训练参数、评测参数、时间、模型、日志和结果。
- [导出配置](config.json)、[机器可读数据统计](reports/dataset-metadata.json)、[当前模型 token 实算](reports/current-model-token-statistics.json)。

## 使用哪个数据目录

每个表格单元对应一个独立的 **source_root**。只把所需的单元目录传给递归收集器；`delivery/` 本身含报告和多套数据，不是训练 source_root。验证集、测试集各自隔离，不能与训练集一起递归导入。两套任务是不同研究阶段的任务，不能自动合并训练。

| 任务 | 训练 source_root | 验证 source_root | 测试 source_root |
| --- | --- | --- | --- |
| 当前五维极性问答 | `polarity-v1/train`，9806 条 | `polarity-v1/validation`，580 条 | `polarity-v1/test`，576 条 |
| 历史三状态 JSON 标注 | `legacy-three-state-v1/train`，2138 条 | `legacy-three-state-v1/validation`，134 条 | `legacy-three-state-v1/test`，126 条 |

每个 source_root 仅含 `records.jsonl`、`manifest.json`、`assets/`。JSONL 采用 `trace.messages`，id、split、task、scene_id 等元数据均放在 trace 内；含图 user 使用有序 image/text 块，无图消息及 assistant 使用字符串。图片以完整 SHA-256 命名，manifest 逐图记录 path、sha256、bytes、width、height，图像字节与原始公开副本一致。

2 张已声明不公开分发的训练图及其记录不进入新交付包。测试原始 126 张图保留：当前极性任务依据已有协议使用其中 87 个有极性标注的场景，产生 288 对、576 问。缺失道德维度继续排除，不转换成负例。

版权与逐图署名沿用 [COPYRIGHT.md](../3.1/data/COPYRIGHT.md)、[image_attribution.csv](../3.1/data/image_attribution.csv) 和 [provenance.json](../3.1/data/provenance.json)。可按 scene_id／legacy id 及图像 SHA-256 对照原署名表。原始标注与历史日志保留在 `3.1/data`、`3.1/results/reference`，供历史结果复现；它们不作为接收方直接扫描的 SFT source_root。

## 下载、校验与重新导出

```bash
git lfs install
git clone https://github.com/Muc-26302077/moral-llava-clip.git
cd moral-llava-clip
git lfs pull
python 3.1/scripts/delivery.py validate \
  --source-root delivery/polarity-v1/train \
  --report /path/to/train-format-check.json
```

格式校验需要 Python 3.11+ 和 Pillow；项目依赖见根目录 requirements.txt。完整重新导出命令如下；输出目录必须尚不存在，不会覆盖当前交付文件：

```bash
python 3.1/scripts/build_delivery.py \
  --config delivery/config.json --output /path/to/new-delivery
```

整包核验会重新导出六个目录并逐字节对比 JSONL／manifest，检查场景与图像的跨 split 隔离，核对 token 统计所属数据、真实推理日志哈希及指标复算：

```bash
python 3.1/scripts/verify_delivery.py --output /path/to/delivery-verification.json
```

打包时必须包含整个 source_root，不能仅发 JSONL。例如：

```bash
tar -czf smid-polarity-train.tar.gz -C delivery/polarity-v1/train .
```

## 一条命令运行极性评测

准备根目录环境以及模型固定版本依赖，下载说明见[模型 README](../3.1/models/moralclip-llava-polarity/README.md)。数据和 checkpoint 是显式输入：

```bash
python 3.1/scripts/evaluate_delivery.py \
  --data delivery/polarity-v1/test \
  --ckpt 3.1/models/moralclip-llava-polarity \
  --moralclip-model /path/to/moralclip-base \
  --output /path/to/new-evaluation --device cuda --precision bf16 --batch-size 8
```

入口核对全部测试 ID、提示、目标、A/B 分组和图像字节，再生成 `answers.jsonl`、`metrics.json`、`inference.log`、含耗时与参数的 `run.json`。答案不会作为生成输入。禁止使用不完整测试或 smoke 结果代替正式结果。离线运行可增加 `--local-files-only`；HF_HOME 指向预先下载的固定版本缓存即可。

只复算归档答案时，可在上述命令中增加 `--predictions 3.1/results/reference/moralclip-llava-polarity-20261104/trained.jsonl`。此模式在 run.json 标明 `archive-replay`，不会声称发生了新推理。

R@1 的完整入口、固定权重和方向定义见 [R@1 协议](../3.1/docs/MORALCLIP_R1_PROTOCOL.md)；3.9 的数据 + checkpoint 一键入口见 [3.9 README](../3.9/README.md)。

## 接收方 processor 验收

飞书多模态规范面向接收方 Qwen3.5-VL sft-workflow，要求使用其 processor 和含 `{% generation %}` 的模板实算展开长度与监督 mask。项目 requirements.txt 是现有 LLaVA 环境，不能据此声称已经完成该接收方验收。接收方在自己的兼容环境中运行：

```bash
python 3.1/scripts/delivery.py validate \
  --source-root delivery/polarity-v1/train \
  --processor /path/to/receiver-processor \
  --max-seq-length 262144 --report /path/to/receiver-train-check.json
```

不传 processor 时只报告格式和媒体校验，`receiver_acceptance` 为 pending；不支持 generation mask、图像未加载、展开长度超限、监督为空均报错。正式接收仍以项目组接收方校验输出为准，并据其输出回填对应 token 栏。当前模型的实际训练 token 另列在信息表，统计脚本使用真实 LLaVA processor、196 个展开视觉 token 和现有训练序列构造函数。
