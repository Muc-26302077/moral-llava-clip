# MoralCLIP–LLaVA：跨模态道德判断与文本引导视觉编码

本项目提供 MoralCLIP 检索、Moral-LLaVA 两阶段训练、图文道德判断评测，以及在视觉编码器内加入文本条件残差的创新结构。基座为 LLaVA-1.5-7B；外部对照为 Qwen3-VL-8B-Instruct、VLGuard Posthoc / Mixed（7B）。

**第一次使用请按 [完整流程指引](docs/workflow.md) 操作。** 已训练适配器、投影器和创新模块通过 Git LFS 提供，外部大模型从固定 Hugging Face 版本获取，也支持本地模型目录。

| 顺序 | 操作 | 指引与入口 |
| --- | --- | --- |
| 1 | 安装环境，获取 MoralCLIP、LLaVA 和 SMID 数据 | [准备环境、模型与数据](docs/workflow.md#1-准备环境模型与数据) |
| 2 | 评测 MoralCLIP 双向 R@1 | [运行 R@1](docs/workflow.md#2-评测-moralclip-的-r1) · `run_retrieval.py` |
| 3 | 训练 MoralCLIP–LLaVA，或直接加载已训练模型 | [两阶段训练](docs/workflow.md#3-训练-moralclipllava) · `scripts/train_baseline.py` |
| 4 | 独立输入图像／文本，评测 TICR 和准确率 | [生成与评分](docs/workflow.md#4-评测-ticr-和准确率) · `run_moral_llava.py` |
| 5 | 训练、加载文本引导视觉编码模块 | [创新方法](docs/workflow.md#5-训练与评测创新结构) · `run_text_guided_vision.py` |
| 6 | 比较外部模型、复算全部最终结果 | [外部 benchmark](docs/workflow.md#6-外部-benchmark-与-moralise) · [结果简报](docs/results.md) · [结果文件](results/README.md) |

当前 SMID 实测：图文微调基线 TICR **90.20%**（同环境重跑），新模块训练终轮 **89.84%**。验证集选择了零初始化残差，当前实验尚未证明新增模块带来改善。原视觉塔 Moral-LLaVA 的历史极性微调对照为 **91.36%**。不同训练初始化和公开模型的比较条件见 [结果简报](docs/results.md)。

```text
.
├── data/                  # 唯一一份 SMID 图片、清单、来源与版权
├── models/                # 最终适配器、投影器、视觉残差权重与获取说明
├── moral_llava/           # 最终训练、推理、结构与评分实现
├── scripts/               # 下载、两阶段训练、benchmark、结果复核与数据导出
├── results/               # 最终指标、逐条原始回答及必要评测证据
├── docs/                  # 完整流程、评测协议、结果简报与成果文案
├── tests/                 # 数据边界、监督 mask、模型结构和指标测试
├── run_retrieval.py        # MoralCLIP R@1
├── run_moral_llava.py      # 数据准备、训练、推理、TICR 评分
├── run_text_guided_vision.py # 文本引导视觉编码训练与评测
└── 3.9/                   # 独立的隐私保护实验包，非上述主流程依赖
```

在 CPU 上复核归档结果，无需重新训练或下载7B基模：

```bash
python -m pip install numpy Pillow
python run_moral_llava.py prepare
python scripts/verify_results.py --output results/runs/verification.json
```

完成 Git LFS 下载后，可用 `python scripts/verify_repository.py` 核对最终数据、模型和结果的文件哈希及文档链接。

开发检查：安装完整依赖后执行 `python scripts/run_checks.py`，按模块独立运行测试以降低内存占用。

数据为 SMID 的项目衍生版本，并非官方 SMID 全集。来源、逐图署名、两张训练图的排除记录和使用条款见 [数据说明](data/README.md) 与 [NOTICE](NOTICE.md)。需要项目组交付格式时，按 [按需导出](docs/workflow.md#7-按需导出项目组交付数据) 生成，不在仓库重复保存图片。
