# SMID 数据与项目衍生版本

[完整操作流程](../docs/workflow.md#1-准备环境模型与数据) · [标签与评测定义](../docs/protocols.md) · [版权与逐图来源](COPYRIGHT.md)

本仓库提供项目实际使用的 `SMID_moralclip` 衍生数据，随 Git 克隆获取，无需另外寻找训练包。它不是官方 SMID 全集。官方来源为 Crone & Bode, *The Socio-Moral Image Database (SMID)*, PLOS ONE 13(1), e0190954：[论文与数据入口](https://doi.org/10.1371/journal.pone.0190954)。希望取得原始 SMID 或单张原图时，应遵循该入口及官方元数据记录的下载与使用条款；`upstream/SMID_metadata.json` 和 `image_attribution.csv` 保存本项目图像的作者、原始 URL 和许可原文。

| 划分 | 原项目场景数 | 当前提供图片数 | 位置 |
| --- | ---: | ---: | --- |
| train | 2,140 | 2,138 | `raw/SMID_moralclip/train/` |
| validation | 134 | 134 | `raw/SMID_moralclip/validation/` |
| test | 126 | 126 | `raw/SMID_moralclip/test/` |

两张训练图按既有版权记录排除，ID及原因见 `excluded_images.json`；不改变验证／测试划分。训练／验证原清单位于 `raw/SMID_moralclip/smid_manifest.json`，测试原清单位于 `raw/SMID_moralclip/test/smid_manifest.json`。每条包含图像路径、道德标签和10条项目生成的英文描述。A/B 分组固定保存在 `groups.json`，文本路统一使用 caption 0。

首次使用在根目录运行 `python run_moral_llava.py prepare`，生成被 Git 忽略的 `data/polarity-v1/`；包含三份问答 JSON 和审计信息。极性任务仅使用相应维度有正／负标注的样本，`not_annotated` 不能解释为已确认的中性或负向。具体样本数及图文输入规则见 [评测协议](../docs/protocols.md)。

本次保留唯一一份原始图片；交付用 `trace.messages` 数据通过 [按需导出](../docs/workflow.md#7-按需导出项目组交付数据) 生成，不重复提交图片。完整来源哈希与授权边界见 `provenance.json`，逐图使用前阅读 [COPYRIGHT.md](COPYRIGHT.md)。原模型上游训练集与该固定测试集是否重合尚未全面核验。

MORALISE 是可选外部评测数据，不放入 SMID 训练目录。获取方式见 [外部评测步骤](../docs/workflow.md#6-外部-benchmark-与-moralise)。
