# 3.1：跨模态道德判断评测

本目录保存本项目实际使用的 SMID 衍生数据、训练完成的 MoralLLaVA、固定评测协议和已有实测结果。输入同一场景的图片或纯文字描述，分别生成五个道德维度的判断，评估两种输入下的完整一致性和正确性。

## 最短运行方式

在本目录运行。已有预测的重新评分不需要 GPU、基模或原服务器目录。

```bash
bash run.sh verify
bash run.sh score
```

`verify` 核对全部交付文件的 SHA-256、模型选择记录、测试覆盖和逐条原始回答；`score` 在新的 `runs/local-score-*` 目录生成 `metrics_constrained.json` 与 `metrics_free.json`。

重新生成模型预测需要 Python 3.11+、NVIDIA CUDA GPU、与驱动匹配的 PyTorch 和 `requirements-train.txt`。当前版本为 7B 模型的 BF16 推理，实际显存占用随批大小变化。需要另行下载公开基模，详见 [模型说明](models/README.md)。

```bash
python -m pip install -r requirements-train.txt
python -c 'from huggingface_hub import snapshot_download; print(snapshot_download("llava-hf/llava-1.5-7b-hf", revision="b234b804b114d9e37bb655e11cbbb5f5e971b7a9"))'
export BASE_SNAPSHOT=/实际缓存路径/snapshots/b234b804b114d9e37bb655e11cbbb5f5e971b7a9
bash run.sh infer
```

## 实验协议

原始衍生包有 2,400 个场景，保留 2,140 训练、134 验证、126 测试的原划分。测试场景在推理前固定为 A/B 两个不重叠子组，各 63 个；文本输入固定使用第 0 条描述，不按预测挑选描述或调整分组。A/B 来自同一数据集，不是两个独立外部测试集。

图片路输入真实图片与统一问题，文本路仅输入场景描述与统一问题。分别输出 `care`、`fairness`、`loyalty`、`authority`、`sanctity`，状态为 `virtue`、`vice` 或 `not_annotated`，并输出与五维一致的 `non_moral` 布尔值。`not_annotated` 表示数据集中未标注，不表示已确认道德中性。

主分析采用合法 JSON token 约束，允许全部 243 种五维组合，不向约束解码提供真值；自由生成作为独立补充分析。主指标分母为本组全部场景，无效回答计失败。

| 指标 | 定义 |
| --- | --- |
| 五维完整一致率 | 两路回答均有效且五维状态全部相同的场景数 / 全部场景数 |
| 逐维一致率 | 两路相同状态的维度数 /（5 × 全部场景数） |
| 两种输入均正确率 | 两路输出均等于参考标签的场景数 / 全部场景数 |
| 图像 / 文本标签完全匹配率 | 对应模态全部标签正确的场景数 / 全部场景数 |
| 结构化输出有效率 | 符合 JSON、字段与标签约束的回答数 / 全部回答数 |

## 数据集与版权

数据依据：Crone DL, Bode S, Murawski C, Laham SM (2018), *The Socio-Moral Image Database (SMID): A novel stimulus set for the study of social, moral and affective processes*, PLOS ONE 13(1): e0190954，[论文](https://doi.org/10.1371/journal.pone.0190954)、[官方数据](https://osf.io/2rqad/)。

本次使用工作区已有的 `SMID_moralclip.zip` 衍生包，并非官方 SMID 全集。原始图像、标签、生成描述和划分保留来源关系，处理后指令 JSON 位于 `data/processed/`。

已从 SMID 官方元数据为全部 2,400 个图片 ID 对应作者、标题、原始 URL 和许可原文，记录在 `data/image_attribution.csv`。不同图片适用不同许可，论文的 CC BY 4.0 不代表全部图片采用同一许可。

公开副本包含 **2,398 张图片，其中测试集 126 张完整保留**。2 张训练图片因官方元数据标为 `All Rights Reserved` 而排除，记录在 `data/excluded_images.json`；另有 3 张图片的许可原文为数字编码，保留并标明待核查。完整原包仍保存在服务器，未被修改。全量重训需要自行按原许可取得缺失训练图片；已有测试预测重算和测试集推理不受该排除影响。

版权、许可链接和引用见 [COPYRIGHT.md](data/COPYRIGHT.md)。本地衍生标签和描述的授权未随原包提供，未将其虚构为新的开放许可。

## 训练好的 MoralLLaVA

| 模型 | 位置 | 用途 |
| --- | --- | --- |
| 图文联合 MoralLLaVA | `runs/crossmodal-joint-v1/epoch-002/` | 本次主要交付的已训练版本 |
| 旧图像标签 MoralLLaVA | `runs/moral-llava-1.5/` | 继续训练初始化及对照 |
| LLaVA 基模 | `llava-hf/llava-1.5-7b-hf` | 另行下载，固定 revision 见下文 |

固定基模 revision：`b234b804b114d9e37bb655e11cbbb5f5e971b7a9`。图文联合版本冻结视觉塔、原投影器和基模权重，仅更新 39,976,960 个语言 LoRA 参数；目标含两路答案监督、五维状态 CE 和 JS 一致性损失。

训练 3 轮，以验证集一致率和质量约束选中第 2 轮，验证集完整一致率为 99/134（73.88%）。测试数据不参与 checkpoint 排名。训练配置、逐轮验证和 `selection.json` 随权重保存。下一轮条件蒸馏目前只有方案和配置，没有训练完成的新权重。

模型文件通过 Git LFS 管理，加载方式和适用条款见 [models/README.md](models/README.md)。

## 分步执行

```bash
bash run.sh verify
bash run.sh score
# 独立保存另一份评分
bash run.sh score --run-dir /tmp/moralllava-score-run2
# 重新推理，需要先设置 BASE_SNAPSHOT
bash run.sh infer --run-dir runs/local-inference-run2
```

`infer` 为当前图片路径生成本地协议，然后加载基模和最新 LoRA，运行真实图像/文本的自由生成与约束生成。新协议、逐条回答和指标保存在指定目录；本地协议哈希与历史协议不同，不混合两次协议的预测。已有输出目录不会覆盖。

原训练代码保留历史服务器的基模缓存路径。若要重训，需先取得缺失训练图片、配置基模路径，并使用新的输出目录。历史文档中的 AutoDL 路径与未入库的旧检索实验产物仅供参考。

## 已验证的主要指标

以下为固定 126 个测试场景上的结果，不是官方全量榜单，也不是新的独立盲测。完整结果位于 `outputs/crossmodal-joint-v1-evaluation/`，原始回答位于 `runs/crossmodal-joint-v1-test/`。

| 测试组 | 指标 | 旧图像 LoRA | 图文联合 MoralLLaVA | 变化（百分点） |
| --- | --- | ---: | ---: | ---: |
| A（63 场景） | 五维完整一致率 | 57.14% | **68.25%** | +11.11 |
| B（63 场景） | 五维完整一致率 | 57.14% | **60.32%** | +3.17 |
| 全部（126 场景） | 五维完整一致率 | 57.14% | **64.29%** | +7.14 |
| 全部（126 场景） | 两种输入均正确率 | 42.86% | 42.06% | -0.79 |

一致率提升伴随共同错误增加：两路一致但错误的场景从 18 增至 28。原始基模在约束输出下始终生成全 `not_annotated`，取得 100% 一致率，属于常量退化，不能证明其道德能力更强。

60% 门槛使用点估计，置信区间下界未达门槛；生成描述与图片的语义等价性仍需人工审核。方法、完整指标与局限见 [实测报告](CROSSMODAL_IMPROVEMENT_RESULTS.md)。

## Benchmark（预留）

`benchmark/` 当前仅含 `.gitkeep`，按要求留空。现有结果保存在 `outputs/`，后续在空目录补充 benchmark。
