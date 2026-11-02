# 数据集

本目录保存本项目实际使用的 `SMID_moralclip.zip` 的解压内容及转换结果。
这是 SMID 来源的项目衍生版本，不能视为原始 SMID 全集或官方重新发布。

| 划分 | 原场景数 / 本次图片数 | 原始数据 | 指令数据 |
| --- | ---: | --- | --- |
| train | 2,140 / 2,138 | `raw/SMID_moralclip/train/` | `processed/train.json` |
| validation | 134 / 134 | `raw/SMID_moralclip/validation/` | `processed/validation.json` |
| test | 126 / 126 | `raw/SMID_moralclip/test/` | `processed/test.json` |

原始训练/验证清单位于 `raw/SMID_moralclip/smid_manifest.json`，测试清单位于
`raw/SMID_moralclip/test/smid_manifest.json`；原清单包含道德标签和每图 10 条描述。
测试 A/B 各 63 个场景，固定使用第 0 条描述，分组协议为 `../runs/crossmodal-v1/protocol.json`。
图片、清单和处理后 JSON 均原样保留，没有重新划分测试样本。

`image_attribution.csv` 逐图记录 ID、划分、相对路径、SHA-256 和署名状态。
`provenance.json` 记录原包哈希、公开来源及当前可确认的授权边界。
已从 SMID 官方元数据补齐全部 2,400 个图片 ID 的作者、标题、原始 URL 和许可原文；官方元数据副本位于 `upstream/`。
许可名称、缺失值和未知编码按原文保留；具体待核查项目见 `COPYRIGHT.md` 和 `provenance.json`。

**使用和转发前请阅读 [COPYRIGHT.md](COPYRIGHT.md)。** `not_annotated` 表示未标注，不能解释为已确认道德中性。

公开副本排除两张官方元数据标为 All Rights Reserved 的训练图，原始清单与标签保留；见 `excluded_images.json`。原包和原项目未修改，测试集全部图片完整提供。
