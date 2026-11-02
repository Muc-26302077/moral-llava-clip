# 多模态大模型评测指标协作项目

本仓库用于共同维护评测指标、测试数据、训练模型与实验结果。当前已整理 **3.1 跨模态道德判断评测**，并为 **3.7 指标**预留目录。其他成员可以通过分支和 Pull Request 添加负责的内容。

| 指标大类 | 内容 | 状态 |
| --- | --- | --- |
| [3.1 跨模态道德判断评测](3.1/README.md) | 数据与版权、已训练 MoralLLaVA、评测协议和结果 | 已提供，其中 benchmark 留空 |
| [3.7 指标](3.7/) | 后续补充指标名称、方法和实验 | 预留，仅有 .gitkeep |

## 最短运行方式

校验数据和重新评分已有预测只需要 Python 3.11+；安装 Git LFS 以获取训练好的模型权重。

```bash
git lfs install
git clone https://github.com/Muc-26302077/moral-llava-clip.git
cd moral-llava-clip
git lfs pull
cd 3.1
bash run.sh verify
bash run.sh score
```

`verify` 校验交付文件、权重、数据划分和测试预测，`score` 根据原始回答重算评测指标。重新生成模型预测需另行下载固定版本的 LLaVA 基模、安装训练依赖并使用 CUDA GPU，详见 [3.1 README](3.1/README.md)。

## 项目结构

```text
.
├── 3.1/                         # 3.1 指标大类
│   ├── README.md                # 数据、指标、运行方式和已验证结果
│   ├── benchmark/               # 空白占位，仅含 .gitkeep
│   ├── data/                    # 数据、逐图署名、许可和来源
│   ├── models/                  # 已训练模型说明及基模条款
│   ├── runs/                    # 正式 LoRA 权重、训练记录、原始预测
│   ├── outputs/                 # 已有评测报告及独立核验
│   ├── moral_llava/             # 训练、推理和评分源码
│   ├── scripts/                 # 数据与交付校验
│   ├── tests/                   # 原有测试
│   └── run.sh                   # verify / score / infer
├── 3.7/                         # 预留指标大类，仅含 .gitkeep
├── CONTRIBUTING.md              # 成员管理与协作
├── NOTICE.md                    # 版权与第三方条款
└── publish_github.sh            # 维护者推送脚本
```

## 数据和模型

数据来自 SMID 的本地衍生包，包含完整的 126 张测试图片与测试划分。为便于核对训练来源，同时提供训练/验证清单和可分发图片；原始 2,400 张图中，2 张训练图片的官方元数据标为 `All Rights Reserved`，公开副本排除这 2 张，原始文件保留在服务器。详见 [数据版权说明](3.1/data/COPYRIGHT.md)。

模型提供验证集选择的图文联合 MoralLLaVA LoRA，以及作为对照的旧图像 LoRA。权重通过 Git LFS 保存，配合指定版本的 LLaVA-1.5-7B-HF 基模加载；基模需另行下载。见 [模型说明](3.1/models/README.md)。

