# GitHub 已上传文件盘点

盘点时间：2026-11-05。仓库：[Muc-26302077/moral-llava-clip](https://github.com/Muc-26302077/moral-llava-clip)。以下是修改前成功 fetch 后的远端快照；不将本机未上传实验和外部下载权重计入已上传内容。

| 远端分支 | 快照 | 跟踪文件数 | 内容 |
| --- | --- | ---: | --- |
| main | `1748599` | 2549 | 已发布 3.1 历史模型、数据、结果与 3.9 实验 ZIP |
| codex/binary-moral-qa-v1 | `a63dbe2` | 2583 | 二元道德基础问答与完整实验流程 |
| codex/ticr-polarity-v1 | `d885a16` | 2626 | 五维条件极性 TICR、MoralCLIP 双向 R@1 |
| codex/moralclip-polarity-baseline | `be7bcd2` | 2675 | 原始 MoralCLIP 视觉塔接入、已训练 adapter 和独立复现 |
| codex/binary-moral-qa-v1-server-20261103 | `2bb3c82` | 89 | 旧服务器布局的代码／文档；含 6 个 notebook checkpoint 文件，属于历史分支 |

本次规范化分支以最新已上传、包含完整模型证据的 `be7bcd2` 为起点。历史服务器布局不重新混入当前项目。分支存在祖先关系，文件数不能相加作为总量。

全部路径均有可查询的逐文件清单：

- [最新已上传分支 2675 个文件](inventory/uploaded-files-be7bcd2.csv)：path、Git blob、Git 存储字节数、mode。
- [所有远端分支文件版本并集](inventory/all-remote-branch-files.csv)：2797 个 path/blob 组合，列出各版本所在分支；同路径不同内容分开记录。
- [分支快照与目录计数](inventory/remote-branches.json)：完整 commit SHA。
- [3.9 ZIP 内部 50 个文件](inventory/3.9-archive-files.json)：内部路径、解压字节数、SHA-256，共 315226 字节。

Git blob 与文件 SHA-256 是不同算法；清单中的 Git 字节数对 LFS 文件指针表示指针大小，不能作为真实模型权重大小。模型本体须通过 Git LFS 获取。

## 最新上传快照的目录用途

| 路径 | 上传内容与用途 |
| --- | --- |
| `README.md`、`environment.yml`、`requirements.txt` | 项目导航、环境与依赖 |
| `.gitignore`、`.gitattributes` | 忽略临时运行产物、管理 LFS 权重与文件换行 |
| `NOTICE.md` | 原数据、模型与第三方材料使用说明 |
| `3.1/data/raw/SMID_moralclip` | 2398 张公开图片、源 manifest；完整 126 张测试图；train/validation/test 来源标注 |
| `3.1/data/processed` | 2400 条历史 LLaVA JSON 数组，含 2 条受限图的历史标注；新交付明确排除这两条 |
| `3.1/data` 下版权／来源文件 | 逐图署名 CSV、排除清单、来源与许可证记录 |
| `3.1/models` | 历史 LoRA 与当前 MoralCLIP-LLaVA adapter、tokenizer、processor、训练元信息；外部基模单独下载 |
| `3.1/moral_llava` | 数据处理、CLIP/LLaVA 训练与推理、极性／二元／三状态评分、视觉塔接入及跨模态训练 |
| `3.1/run*.py`、`run.sh` | R@1、TICR、历史任务的命令行入口 |
| `3.1/scripts`、`3.1/tests` | 实验编排、评测、评分、发布核验及自动测试 |
| `3.1/results/reference` | 训练状态与 loss、逐条模型回答、参数、指标、独立复现及分析，属于实验记录 |
| `3.1/docs` | 正式协议、方法、历史方案与归档文档 |
| `3.1/benchmark/.gitkeep` | 空白预留目录，不含已交付的新 benchmark |
| `3.1/release_manifest.json` | 原 3.1 发布材料的文件校验清单；保留其历史验证范围 |
| `3.9/moralclip_3.9_eps20_20261103.zip` | LDP-Slicing／CluSanT 隐私管线、MoralCLIP 检索与成员推断代理实验、冻结配置与说明 |

原快照共 2398 张图、66 个 Python 文件、34 个 Markdown 文件、108 个 JSON、22 个 JSONL；此外包括模型权重、tokenizer、NumPy 检索矩阵、日志、Shell、CSV、环境配置和 ZIP。历史预测 JSONL 是评测记录，不作为 SFT messages 数据直接导入。

## 本次新增文件

- `delivery/polarity-v1/{train,validation,test}`：当前任务的三个标准 source_root。
- `delivery/legacy-three-state-v1/{train,validation,test}`：历史任务格式转换后的三个独立 source_root。
- `delivery/reports`：格式校验、实际 token 统计、数据分布、loss 曲线及新入口的真实推理／归档复算证据。
- `delivery/config.json` 与本目录文档：数据转换配置、下载方法、交付信息表、规则映射和本盘点。
- `3.1/scripts/{delivery,build_delivery,evaluate_delivery,training_statistics,verify_delivery}.py` 与 `3.1/tests/test_delivery.py`：转换、校验、评测、整包复现核验和边界测试。
- `3.9/README.md` 与 `3.9/run_evaluation.py`：可迁移的原始数据 + 私有数据 + checkpoint 评测入口。

当前改动未把本机 `multimodal-moral-benchmark-20261104` 整个实验目录、虚拟环境或外部基模权重上传。MORALISE、Qwen3-VL、VLGuard 的本机实验不在修改前的 GitHub 快照中，不能冒充原仓库已发布的材料。

整理后完整项目的逐文件真实字节数和 SHA-256 见 [project-manifest.json](project-manifest.json)，覆盖 6669 个文件，清单自身除外；加上清单共 6670 个跟踪文件。LFS 条目记录下载后的真实权重校验值。新增 3945 个图像路径全部复用原始公开图像字节；测试与原材料核验报告见 [review.json](reports/review.json)。
