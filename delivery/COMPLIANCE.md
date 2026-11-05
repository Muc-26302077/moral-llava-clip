# 规则映射与验收范围

依据项目组通知，按统一格式整理并共享本课题已有材料。源文档全部公开读取：总交付模板 152 个文档块、纯文本规范 152 块、多模态规范分页合并后 304 块，全部子块完整；来源与正文摘要哈希见 [rule-sources.json](reports/rule-sources.json)。仅保存规则对应与必要短引，不上传网页脚本或用户资料。

| 来源 | 链接 |
| --- | --- |
| 总交付模板 | [多模态数据格式&交付内容](https://sii-czxy.feishu.cn/wiki/KKWAwsH2diOJzLksJX5c8dvOnPh) |
| 文本格式 | [SFT 数据交付格式（纯文本）](https://sii-czxy.feishu.cn/wiki/Hbe5wdHbSiDN5Ik9XL6ceNpSnYP) |
| 图像扩展格式 | [SFT 数据交付格式（多模态）](https://sii-czxy.feishu.cn/wiki/Y245wfcw6iSEOokGjbNc2wwYnxg) |
| 补充 benchmark 参考 | [腾讯文档 Benchmark 收集](https://docs.qq.com/sheet/DWkZaa2F5dWhRTkp5?tab=BB08J2)；作为项目组参考链接，不据此变更已冻结测试定义 |

## 硬性格式对应

| 来源位置／规则 | 本项目落实 | 验证方式／状态 |
| --- | --- | --- |
| 文本“原始文件和目录要求”：已有 source_root，递归可见 `*.jsonl`，逐行 JSON 对象 | 六个独立 source_root，数据、媒体与报告分开 | 全部 13360 条逐行解析通过 |
| 文本“支持的记录外层结构”：messages 或 trace；元数据与 messages 同在 trace | `trace.messages`，id/split/scene/task 保留在 trace 内 | exporter 与 validator 检查 |
| 文本“messages 字段”：合法小写 role，system 仅首位，真实 user、最后 user 后有监督 assistant | human/gpt 转 user/assistant；最终目标答案保留 | 缺 user、只有 prompt、全禁 loss 等负例测试 |
| 文本“tools 字段”：function.arguments 为对象或可解析对象字符串 | 当前数据无工具；validator 支持工具结构并拒绝数组、数字、坏 JSON 参数 | 工具参数负例测试 |
| 文本“哪些内容参与 loss”：最后真实 user 后的 assistant；布尔 loss/thinking 控制；不手写 mask | 当前单轮数据无历史轮次歧义、无手写 mask；校验布尔类型 | 结构检查及现有训练序列监督测试 |
| 图像 §1：图片随整个目录交付；不夹带脚本和日志 | source_root 仅 JSONL、manifest、assets | 六根目录逐一检查；说明提供 tar 整目录打包命令 |
| 图像 §2.2–2.3：有序 text/image 块，text 非空，image 仅 user | 去除手写 `<image>`，改成 image 块；纯文本保持字符串 | role、空块、未知块、video、手写占位符负例测试 |
| 图像 §3.1：jpg/jpeg/png/webp 小写扩展名匹配编码，≤20 MB，合法命名 | 按真实编码定扩展名，SHA-256 命名；未改图像字节 | Pillow 全部解码；真实大小、编码检查 |
| 图像 §3.2：相对 POSIX 路径，无绝对路径、`..`、`~`、隐藏目录、URL/base64 | 根内 `assets/<sha256>.<ext>`；额外拒绝符号链接 | 越界、隐藏、中文空格、URL、符号链接负例测试 |
| 图像 §3.3：manifest 必需，逐图 path/sha256/bytes/width/height，覆盖全部图像、不重复、不缺失 | 六个 manifest，逐根目录列出真实媒体 | checksum、尺寸、缺文件、多余文件、重复登记负例测试 |
| 图像 §3.4：v1 不接收 video | 只输出 image/text，video 拒收 | 自动测试 |
| 图像 §4、§6-5/6：processor 展开预算、截断后非空监督、generation-aware 模板 | 提供可调用 receiver processor 的验证入口；当前模型另用实际 processor 和训练序列构造函数统计 | 当前模型全部 13360 条预算和监督检查通过；Qwen3.5-VL 接收方验收待其环境运行 |

图像“≤8192 单图 token”“≤8 图”“建议单边≤4096”“建议 RGB”和 msswift 框架属于建议项。不能将它们误写为所有代码、日志、检索结果必须满足的 SFT 数据规则。本项目一条含图记录使用一张图片；不为格式整理而改写冻结图像或模型架构。

## 总交付信息对应

| 模板字段 | 对应材料／状态 |
| --- | --- |
| 数据路径、benchmark、条数、类型 | [INFORMATION.md](INFORMATION.md)；clone + source_root；87 场景／288 对／576 问 |
| 数据 + ckpt 一键入口 | 3.1 `evaluate_delivery.py` 已真实完整运行；3.9 `run_evaluation.py` 提供私有对评测入口，尚未准备完整外部私有数据复测 |
| 评测参数、时间、并发、模型回答与结果 | `reports/polarity-live-run.json`、`polarity-live-answers.jsonl`、`polarity-live-metrics.json`、`polarity-live-inference.log` |
| Intern-S2 基础评分与逐条日志 | 待项目组 Intern-S2 联测；未得到本课题该模型的实测，不填伪造分数，不要求用户为本次格式整理先提供模型 |
| 版本、条数、有图无图、类别分布、轮数、来源、采集模型、CoT、目标 benchmark | `config.json`、`reports/dataset-metadata.json` 与信息表 |
| 总 tokens、Trainable tokens、长度分布 | `reports/current-model-token-statistics.json` 为当前 MoralCLIP-LLaVA 实算；接收方 Qwen3.5-VL 栏保留 pending，待实际 receiver 输出回填 |
| Dataflow Config、SFT 参数 | 本项目数据转换配置 `config.json` + 实际执行脚本 `build_delivery.py`；真实训练配置 `frozen_config.json`；项目组 sft-workflow 专用 config 尚未提供，不将本项目 JSON 伪称其原生配置 |
| SFT 模型、loss 曲线、与基线比较 | 当前模型路径、逐点 loss CSV／SVG、原始 trainer_state 和多数类／projector-only／trained 对照 |
| 指标 3.9 的外部资产与复测 | ZIP 内容和缺口如实列出；未上传的私有样本、候选缓存与第三方资产仍需项目组另行获取 |

## 本次合并判定

本次变更的验收对象是 GitHub 项目的格式整理、完整性、可追溯性和现有模型评测入口。合并前检查格式和媒体、跨 split 场景及图像隔离、全部数据转换一致性、自动测试、实际完整极性推理、历史结果复算、原发布材料完整性及 Git 差异。

Intern-S2 联测和接收方 Qwen3.5-VL workflow 验收是后续项目组接收阶段的独立事项，状态始终显式保留。合并不表示这些未运行的实验已经通过，也不表示 3.9 全部外部数据已经包含在仓库中。若本次代码／数据检查失败，先修正，不将失败状态写成通过。
