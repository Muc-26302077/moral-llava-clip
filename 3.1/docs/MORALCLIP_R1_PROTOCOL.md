# MoralCLIP 项目 R@1 基准协议

2026-11-04 起，项目将此前“道德标签 Hit@1”正式命名为 **R@1**。成功条件不变：Top-1 结果与查询所属场景共享至少一个道德标签即命中。现同时支持图到文与文到图，两个方向分别统计。这里 R@1 是查询级首位命中率，不是找回的相关描述数量占全部相关描述的比例。

## 指标说明（测试大纲用）

R@1 用于评价图文检索结果在道德基础标签上的相关性。该指标采用关怀/伤害、公平/欺骗、忠诚/背叛、权威/颠覆和纯洁/污秽五个道德基础维度的原始标签，并保留非道德（non-moral）类别；文本标签继承其对应场景的标签。在图到文检索中，计算查询图像特征与候选文本特征之间的余弦相似度，选取相似度最高的文本作为 Top-1 检索结果；若查询图像与该文本对应场景共享至少一个道德标签，则本次检索计为成功。在文到图检索中，计算查询文本特征与候选图像特征之间的余弦相似度，选取相似度最高的图像作为 Top-1 检索结果；若查询文本对应场景与该图像共享至少一个道德标签，则本次检索计为成功。两个方向均保留原配候选，分别以检索成功的查询数占该方向查询总数的比例计算 R@1。

## 原始模型与来源

使用用户确认的最初模型 `anaaa2/moralclip-base`，不加载 LLaVA 或历史检索迁移分支的权重。原始权重信息保存在 [moralclip-original.json](../models/moralclip-original.json)：

- 固定上游 revision：`1e337922befebde3f0cadb5749f37ae9521a0751`。
- `model.safetensors`：598530372 bytes，SHA-256 `4ace10e40a63343dc270afe776a0cc851d28dc36fd13417de07dccb12e4bac2a`。
- 已核验上游该 revision 的 LFS SHA-256 与本地原始模型相同；上游训练场景与项目测试场景是否重合尚未核验。
- 用户上传的 `moral_clip/moralclip_eval.zip` 包含原配图文检索代码、Windows 路径配置和历史报告，没有模型权重。本次按用户确认的服务器权重目录重新推理，不把包内历史报告当成本次结果。

## 测试目标与步骤

1. 使用固定 SMID 测试集 126 张图像、1260 条描述，每图 10 条。图到文以全部1260条文本为候选，文到图以全部126张图像为候选；保留原配候选。保留 39 个 non-moral 场景，non-moral 作为普通相关性标签，不套用极性任务的 87 场景筛选。
2. 由原始 MoralCLIP 双编码器分别提取图像、文本特征，L2 归一化，以余弦相似度对整个候选库排序。超过 CLIP 的 77-token 上限时截断，并记录数量；本轮为 44 条。
3. 参考标签仅用于评分，不输入编码器。首位候选所属场景与查询场景的标签交集非空记 1，否则记 0。图到文按126个图像查询计算，文到图按1260个文本查询计算，每条描述以“场景ID＋描述索引”单独计为一个查询，不按场景平均或合并两方向分母：`R@1 = 该方向道德标签相关首位命中数 / 该方向查询数`。未标注不当作负极性，仍采用原始标签集合求交集。
4. 保存原始权重、代码、清单、图片哈希，以及完整相似度矩阵、逐查询首位文本与命中依据。使用独立重算脚本核对参考标签、矩阵排序与汇总指标。

## 三个数值的名称

| 名称 | 匹配规则 | 地位 |
| --- | --- | --- |
| 图到文／文到图 R@1（原道德标签 Hit@1） | 首位结果与查询共享至少一个道德标签 | 项目 MoralCLIP 基准，两个方向分别报告 |
| 同图配对 R@1 | 首位文本属于查询图像自身 | 诊断，不能代替主指标 |
| 跨场景 R@1 | 先删除本图全部描述，再按共享标签判断首位 | 候选库不同的补充诊断 |

新输出的 `directions.image_to_text.r_at_1` 和 `directions.text_to_image.r_at_1` 分别为两个方向的正式指标。为兼容原图到文结果，顶层 `r_at_1`、`metrics.r_at_1` 与 `metrics.label_relevance.r_at_1` 继续指图到文，并通过 `legacy_flat_metrics_direction` 明示，不能解释为双向平均；`same_image_pairing.r_at_1` 仅是配对诊断。历史 `hit_at_1` 等字段仅作为读取旧文件的兼容别名保留，展示与新报告统一使用 R@1。历史实验的 63.49%/55.56% 是 LLaVA 检索分支的**同图配对**成绩，不能改名为本基准成绩。

指标 3.9 的隐私保护代码包中，双向原配图文 R@1 及其保持率属于另一测试协议；其数值与公式不能改成指标 3.1 的道德标签 R@1。

## 从仓库运行

按仓库 requirements 安装环境。原始 MoralCLIP 权重不重复上传到 Git，必须提供本地完整目录，或显式下载已固定 revision 的公开权重。以下命令均从仓库根目录执行，脚本不依赖本机服务器路径。

```bash
# 本地原始权重；与用户确认的原始 checkpoint 强制核对哈希。
python 3.1/run_retrieval.py \
  --model /path/to/moralclip-base \
  --expected-model-sha256 4ace10e40a63343dc270afe776a0cc851d28dc36fd13417de07dccb12e4bac2a \
  --output 3.1/results/runs/original-r1/test.json

# 无本地权重时显式下载固定版本；网络受限时可按环境配置 HF_ENDPOINT。
python 3.1/run_retrieval.py \
  --model anaaa2/moralclip-base --allow-download \
  --revision 1e337922befebde3f0cadb5749f37ae9521a0751 \
  --expected-model-sha256 4ace10e40a63343dc270afe776a0cc851d28dc36fd13417de07dccb12e4bac2a \
  --output 3.1/results/runs/downloaded-original-r1/test.json

# 不加载模型、不需要 GPU：从已归档矩阵与逐查询记录独立复算。
python 3.1/scripts/score_retrieval.py \
  --evaluation 3.1/results/reference/moralclip-r1-bidirectional-20261104/test.json
```

终端长期任务使用 `screen -L -Logfile r1.screen.log -dmS moralclip-r1 python -u ...`。每次指定新输出位置，禁止覆盖历史结果。最新双向实测：图到文 **126/126，100.00%**；文到图 **1176/1260，93.33%**。表格与逐查询结果见[双向基准报告](../results/reference/moralclip-r1-bidirectional-20261104/test.report.md)。先前的单向报告作为历史证据保留。

## 与 Moral-LLaVA 的关系

MoralCLIP R@1 衡量标签相关检索；Moral-LLaVA TICR 衡量同一场景图像/文字两路极性回答的一致性，极性准确率另与参考答案比较。它们不是同一指标。当前已完成极性微调的模型保留 LLaVA 原视觉塔和投影器，未接入本次原始 MoralCLIP 权重；历史 B/16 换塔检索分支使用普通 OpenAI CLIP 初始化。检索基准不能当成编码器迁移已经完成的证据。

## 远程代码测试目标审计

| 入口或上传材料 | 核实的目标 | 本次处理 |
| --- | --- | --- |
| 上传的原始 evaluate.py | 双向原配图文检索，无道德标签判分 | 保留上传原件；不作为新的正式入口 |
| run_retrieval.py / scripts/evaluate_retrieval.py | 原始 MoralCLIP 的双向道德标签 R@1 | 显式原始模型、严格加载、主指标字段与复算记录 |
| scripts/evaluate_alignment.py | 实际 LLaVA 检索 checkpoint 的道德标签 R@1 | 主指标统一，配对与跨场景单列，不使用原始 MoralCLIP 成绩冒充 |
| moral_llava/train_alignment.py | 验证集选模 | 新运行默认 moral_r1，保留显式 moral_map / paired_r1；不改变历史 checkpoint 或重训 |
| run.py | 历史五维三状态 JSON 生成与一致性评分 | 在 README 中明确为历史入口，不称为当前极性 TICR 或 MoralCLIP R@1 |
| run_polarity.py | 五维条件极性二分类与 TICR | 保留正式极性协议，和检索基准分别运行 |

历史原始预测、权重、训练元数据不改写；历史文档增加正式协议说明。更名不追认旧配对检索成绩为新主指标。

## 已完成的 GitHub 远程复核

从远程分支重新克隆代码提交 `0d374b038daa7a3941a962685a3553ceffe548ba`，在项目工作目录之外执行完整 GPU 推理；126 个查询的首位结果与命中判断全部一致，完整相似度矩阵逐元素相同，19 项相关测试及归档独立复算通过。原始 checkpoint 的配置和预处理／分词器文件也与固定上游 revision 的 Git blob 逐一一致。

证据：[远程复核](../results/reference/moralclip-r1-original-20261104/remote_audit.json)、[公开模型文件核对](../results/reference/moralclip-r1-original-20261104/public_model_files_audit.json)。本次复核复用已安装依赖及显式指定的原始权重，不声称已验证新建 Conda 环境或重新训练 Moral-LLaVA。

## 双向扩展验证

新增文到图方向直接转置同次推理的126×1260相似度矩阵，并对每条文本按其源场景标签评分；没有重新训练或改变成功条件。新归档同时保存两个方向的首位记录，独立评分脚本分别核对查询数、候选数、源标签及矩阵排序，支持读取历史单向结果。相关22项测试通过，包含非对称查询数、共享标签成功但原配失败、反向记录篡改检测。
