# 课题三：多模态大模型价值对齐与纠正方法研究

本仓库共享指标 **3.1 跨模态道德判断**的评测数据、SFT 数据构建程序、已训练模型、评测代码与最终结果，并提供指标 **3.9 隐私保护后跨模态价值一致性保持**的独立实验包。主要方法为 MoralCLIP–LLaVA，以及问题文本引导的视觉编码器低秩残差结构。

本 README 按项目组的[《多模态数据格式&交付内容》](https://sii-czxy.feishu.cn/wiki/KKWAwsH2diOJzLksJX5c8dvOnPh?from=from_copylink)组织评测和数据交付信息；统一格式依据其关联的[纯文本 SFT 规范](https://sii-czxy.feishu.cn/wiki/Hbe5wdHbSiDN5Ik9XL6ceNpSnYP)与[多模态 SFT 规范](https://sii-czxy.feishu.cn/wiki/Y245wfcw6iSEOokGjbNc2wwYnxg)。[Benchmark 收集表](https://docs.qq.com/sheet/DWkZaa2F5dWhRTkp5?tab=BB08J2)作为项目组参考。

**使用入口：** [完整训练与评测流程](docs/workflow.md) · [当前结果简报](docs/results.md) · [最终结果与逐条回答](results/README.md) · [模型获取](models/README.md) · [数据来源](data/README.md) · [创新方法成果文案](docs/成果文案.md)

## 一、评测 code & data

### 1. 评测数据交付信息表

| 项目组要求 | 本项目内容与位置 |
| --- | --- |
| 评测数据路径／下载方法 | Git 克隆获取 `data/raw/SMID_moralclip/`；运行 `python run_moral_llava.py prepare` 生成 `data/polarity-v1/test.json`。MORALISE 按[流程第6步](docs/workflow.md#6-外部-benchmark-与-moralise)从作者 Hugging Face 数据集获取。 |
| Benchmark 名称／类型 | SMID 道德标签相关检索 R@1；SMID 五道德基础条件极性问答 TICR／准确率，属于道德价值判断评测；MORALISE 为独立道德判断与规范识别评测。三者指标分别报告。 |
| 评测条数 | R@1：126图、1,260条描述；极性任务：87个有效场景、288个场景—维度对、576条图像／文本独立问答；MORALISE：S1为2,481题，S2／S3各2,480题。 |
| 评测脚本入口 | R@1：`run_retrieval.py`；极性任务的数据＋checkpoint 一键入口：`scripts/evaluate_delivery.py`，命令见下；MORALISE：`scripts/evaluate_moralise.py`。 |
| 被测模型／checkpoint | `models/moralclip-llava-polarity/` 为已训练基线；`models/text-guided-vision/` 为新增视觉残差。对照包括原视觉塔极性微调模型、Qwen3-VL-8B-Instruct、VLGuard 7B Posthoc／Mixed；固定版本见[模型说明](models/README.md)。 |
| 评测参数 | 本项目极性模型：自由贪心生成，`do_sample=false`，temperature 不启用，无 thinking 阶段，`max_new_tokens=16`，`max_length=2048`，BF16。历史基线 batch 8；新增模块及其同次基线 batch 1。R@1 非生成任务；MORALISE 最多生成64个 token。外部模型的原生模板和视觉处理见[协议](docs/protocols.md)。 |
| 评测时间／并行情况 | 标准交付入口的已归档基线完整评测：1进程、batch 8、576问，推理含加载61.014秒、总耗时61.317秒，见[原始运行记录][baseline-run]。该耗时只对应该次运行；新模块3轮训练与两次测试合计约52分钟。其他模型未在本表填报未经核实的统一耗时。 |
| 评测结果与逐条日志 | [结果索引](results/README.md)包含最终指标、逐条原始回答、检索矩阵与协议指纹；标准交付入口基线另有[原始回答][baseline-answers]、[评分][baseline-metrics]和[推理日志][baseline-log]。固定提交链接保留历史证据，其中旧机器路径仅为来源记录。 |
| Intern-S2 基础评分／逐条评测 log | **待项目组联测。当前没有 Intern-S2 的实测评分、回答和 eval 日志。** 本项目及其他公开对照的成绩不能替代这一交付项。 |

指标3.9的获取方式、数据＋checkpoint 入口和历史结果见 [3.9 交付说明](3.9/README.md)。该实验包尚缺外部私有图文对、候选缓存与完整复算资产，当前不声明已完成完整复测或项目组验收。

### 2. 数据＋checkpoint 一键评测

先按[流程第1步](docs/workflow.md#1-准备环境模型与数据)安装依赖、获取模型并准备数据。对生成的测试 JSON 运行：

```bash
python scripts/evaluate_delivery.py \
  --data data/polarity-v1/test.json \
  --ckpt models/moralclip-llava-polarity \
  --moralclip-model models/base/moralclip \
  --output results/runs/delivery-evaluation \
  --device cuda --precision bf16 --batch-size 8
```

入口从 Hugging Face 的固定版本或已有缓存加载 LLaVA 基座，核对完整测试集的 ID、提示、目标、A/B 分组与图像字节，再生成 `answers.jsonl`、`metrics.json`、`inference.log`、`run.json` 和实际评测输入；运行记录包含参数、时间、batch 与权重哈希。输出目录须为新目录。任意本地基模目录的加载方式见[流程第4步](docs/workflow.md#4-评测-ticr-和准确率)。

下文导出标准数据后，也可将 `--data` 换为 `results/runs/delivery/test`。增加 `--predictions results/baseline/trained.jsonl` 可复算归档回答，运行记录标为 `archive-replay`，不计作新模型推理。

## 二、SFT 数据交付信息表

| 项目组要求 | 当前五维极性 SFT 数据 |
| --- | --- |
| 数据路径／获取方法 | 原图、标签与描述随仓库提供，路径见[数据说明](data/README.md)。本项目训练输入由 `run_moral_llava.py prepare` 生成；项目组交付数据须按下文导出为 `trace.messages` JSONL，连同媒体整目录共享。 |
| 版本号／run_id | 任务协议 `smid-foundation-polarity-v1`；下文导出分别使用 `smid-polarity-train`、`smid-polarity-validation`、`smid-polarity-test`，写入各自 manifest。 |
| 数据条数／有图无图 | train：9,806条，其中有图／无图各4,903条；validation：580条，各290条；test：576条，各288条。有图记录分别共享1,381／79／87张图片。 |
| 平均轮数／agent／harness | 每条为单轮 user→assistant；非 agent 数据，无 harness。 |
| 类别分布 | 五维正／负极性：关怀／伤害、公平／欺骗、忠诚／背叛、权威／颠覆、纯洁／污秽。训练每模态对应1,211／668／1,043／978／1,003条，测试每模态72／42／63／55／56条；逐维正负计数见生成的 `data/polarity-v1/audit.json`。 |
| 长度分布／总 token／Trainable token | 当前 MoralCLIP–LLaVA 的实算见下表；接收方 processor 的 token 与监督 mask 统计须在其实际环境中校验回填，当前为待完成。 |
| Dataflow Config／数据构建 | 构建入口为 `run_moral_llava.py prepare`，配置来源为固定的 [groups.json](data/groups.json)、[excluded_images.json](data/excluded_images.json)与原始清单；标准格式转换入口为 [scripts/delivery.py](scripts/delivery.py)，参数见下文。接收方 sft-workflow 专用配置待对接，不将本项目脚本声明为该框架原生配置。 |
| SFT 训练参数参考 | 冻结 MoralCLIP 与语言基座；先用4,903条图像问答训练投影器1轮，再用9,806条图文问答训练投影器＋语言 LoRA 3轮；BF16、batch 8、seed 42、最大长度2,048。完整参数见[training-config.json](results/baseline/training-config.json)。msswift 为项目组推荐框架，本项目使用现有训练实现。 |
| 来源、构建做法、采集模型、CoT | 由 SMID 项目衍生标注构造问答，仅使用对应维度已有正／负标注的场景，文本路固定 caption 0。不将未标注当负例；没有新增模型采集，无 CoT；原英文描述生成模型尚未核实。 |
| 对应 benchmark／各自数据量 | 面向 SMID 极性 TICR 与准确率：train／validation／test分别9,806／580／576条。MORALISE 仅作外部评测，未用于训练或选模；不为三个公开对照额外构造训练数据。 |
| 验证结果：SFT 模型、loss、基线比较 | [已训练基线](models/moralclip-llava-polarity/README.md)、[创新模块](models/text-guided-vision/README.md)、[最终结果比较](docs/results.md)；已实测的基线[loss 曲线][loss-curve]、[投影器逐点 loss][projector-loss]与[图文微调逐点 loss][adapter-loss]使用固定提交归档链接。 |

下表来自[当前模型 token 实算归档][token-statistics]中 `polarity-v1` 项：按一遍数据统计，包含提示、每图196个展开视觉 token、答案与 EOS；不含 padding 与 epoch 重复，Trainable token 按 next-token shift 后的监督标签计数。这是 **MoralCLIP–LLaVA 口径，不是接收方 Qwen3.5-VL 验收统计**。

| 划分 | 总 token | Trainable token | 长度 min／p50／p95／max |
| --- | ---: | ---: | --- |
| train | 2,387,399 | 70,496 | 138／302／318／356 |
| validation | 140,930 | 4,188 | 144／302／318／318 |
| test | 139,671 | 4,136 | 140／302／318／318 |

## 三、统一数据格式与交付操作

本项目训练入口使用的问答 JSON 是内部格式；**交付前须转换为统一格式**。train、validation、test 各使用独立 `source_root`，每个目录仅含 `records.jsonl`、`manifest.json` 与 `assets/`。日志、代码和统计报告放在目录外；不得将整个仓库或三个 split 的父目录作为训练数据根目录递归导入。

JSONL 每行采用 `{"trace":{"id":"…","messages":[…]}}`，id、split、scene_id、task、modality 等元数据与 messages 同在 trace 内。含图 user 的 content 使用有序 image／text 块，image 引用 `assets/<sha256>.jpg` 等根内相对路径；纯文本消息与 assistant 答案使用字符串。保留真实 user 与可监督 assistant，不手写图像占位符或 loss mask。图片随整个目录交付；manifest 逐图登记 `path`、`sha256`、`bytes`、`width`、`height`。

已运行数据准备后，按 split 导出并校验：

```bash
for split in train validation test; do
  python scripts/delivery.py export \
    --data "data/polarity-v1/$split.json" \
    --image-root data/raw/SMID_moralclip \
    --run-id "smid-polarity-$split" \
    --output "results/runs/delivery/$split" || exit 1
  python scripts/delivery.py validate \
    --source-root "results/runs/delivery/$split" \
    --report "results/runs/delivery/$split-format-check.json" || exit 1
done
tar -czf results/runs/smid-polarity-train.tar.gz \
  -C results/runs/delivery/train .
```

验证、测试同样整目录打包，不能只发送 JSONL。导出目录须尚不存在；产物保存在被 Git 忽略的 `results/runs/`，供项目组共享，不重复提交原图。

格式与媒体校验不等于接收方验收。接收方在其兼容环境中，对三个 split 分别使用实际 Qwen3.5-VL processor 和包含 `{% generation %}` 的模板，检查图像展开长度、截断与非空监督，例如：

```bash
python scripts/delivery.py validate \
  --source-root results/runs/delivery/train \
  --processor /path/to/receiver-processor \
  --max-seq-length 262144 \
  --report results/runs/delivery/receiver-train-check.json
```

不传 processor 时，`receiver_acceptance` 保持 pending。**Intern-S2 评分与日志、接收方 token／监督 mask 验收、接收方 workflow 专用配置仍待对接**；本地检查通过不代表这些项目组交付项已完成。既有规范读取记录见 [delivery-sources.json](docs/delivery-sources.json)。

## 四、从模型获取到训练评测

```bash
git lfs install
git clone https://github.com/Muc-26302077/moral-llava-clip.git
cd moral-llava-clip
git lfs pull
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/download_models.py moralclip
python scripts/download_models.py llava
python run_moral_llava.py prepare
```

建议 Linux、Python 3.11、NVIDIA CUDA GPU；CPU 可准备数据与复算指标。已训练的 LoRA、投影器、视觉残差通过 Git LFS 提供；完整基座按固定 Hugging Face revision 下载，也支持本地目录，见[模型说明](models/README.md)。原实验使用 A800 80GB，新残差模块训练峰值已分配显存15.18 GiB；基线两阶段训练的资源需求另见流程。

| 步骤 | 入口与操作说明 |
| --- | --- |
| 1. 获取 MoralCLIP、LLaVA 与 SMID | [环境、模型、数据](docs/workflow.md#1-准备环境模型与数据) |
| 2. MoralCLIP 双向 R@1 | [运行检索](docs/workflow.md#2-评测-moralclip-的-r1) · `run_retrieval.py` |
| 3. Moral-LLaVA 两阶段训练 | [训练函数与参数](docs/workflow.md#3-训练-moralclipllava) · `scripts/train_baseline.py` |
| 4. TICR 与准确率 | [推理与评分](docs/workflow.md#4-评测-ticr-和准确率) · `run_moral_llava.py` |
| 5. 文章创新方法 | [文本引导视觉残差训练与评测](docs/workflow.md#5-训练与评测创新结构) · `run_text_guided_vision.py` |
| 6. 对照模型与 MORALISE | [外部 benchmark](docs/workflow.md#6-外部-benchmark-与-moralise) |
| 7. 项目组数据交付 | 按上文导出、校验、整目录打包，详见[交付操作](docs/workflow.md#7-按需导出项目组交付数据) |

## 五、最终结果与复核

SMID 极性任务使用五道德维度等权宏平均，单位为%。TICR 判断两路合法答案是否一致，需同时查看准确率；两路共同答错也可能一致。

| 方法／模型 | TICR | 图像准确率 | 文本准确率 | 共同正确率 |
| --- | ---: | ---: | ---: | ---: |
| MoralCLIP–LLaVA，仅训练投影器 | 74.37 | 90.61 | 70.63 | 67.81 |
| Moral-LLaVA，原视觉塔极性微调 | 91.36 | 93.95 | 92.32 | 88.81 |
| MoralCLIP–LLaVA，新增模块的同次基线 | 90.20 | 88.96 | 92.32 | 85.74 |
| 问题文本引导视觉残差，第3轮 | 89.84 | 88.61 | 92.32 | 85.38 |
| Qwen3-VL-8B-Instruct | 81.28 | 84.21 | 75.05 | 70.27 |
| VLGuard：LLaVA-v1.5-7B-Posthoc | 41.08 | 68.00 | 57.70 | 35.06 |
| VLGuard：LLaVA-v1.5-7B-Mixed | 76.14 | 79.62 | 69.04 | 62.40 |

新增模块在 MoralCLIP 第11个编码块内根据问题文本生成视觉残差调节信号，新增192,896参数，原模型冻结。当前第3轮相对同次基线 TICR 下降0.36个百分点，验证集选择第0轮零残差，尚未证明独立增益。本项目模型经过任务微调，三个外部公开对照直接推理，差异不能全部归因于结构。完整条件与限制见[结果简报](docs/results.md)。

MoralCLIP 的道德标签相关 R@1：图→文100.00%、文→图93.33%，不等同于严格同图配对 Recall@1。MORALISE 仅完成图文微调基线评测，S1／S2／S3分别为35.18%／0.59%／2.80%，三项定义不同，不作为 TICR 报告。

无需下载7B基座，可在 CPU 上复核最终结果：

```bash
python -m pip install numpy Pillow
python run_moral_llava.py prepare  # 首次使用执行；已准备则跳过
python scripts/verify_results.py --output results/runs/verification.json
python scripts/verify_repository.py
```

`verify_results.py` 复算10组 SMID 指标、双向 R@1 和7,441条 MORALISE 回答；`verify_repository.py` 检查数据、模型、结果哈希与本地文档链接，需已完成 Git LFS 下载。开发检查在完整依赖环境执行 `python scripts/run_checks.py`。新运行的日志和中间 checkpoint 放入 `results/runs/`，最终归档索引见 [results/README.md](results/README.md)。

## 六、目录与使用条款

```text
data/                    # 唯一一份 SMID 项目衍生数据、清单与来源
models/                  # 已训练组件、固定版本模型获取说明
moral_llava/             # 训练、推理、创新结构与指标实现
scripts/                 # 模型下载、训练、评测、交付与复核
results/                 # 最终指标、逐条回答及必要证据
docs/                    # 完整流程、协议、结果与成果文案
tests/                   # 数据、监督、结构与指标检查
3.9/                     # 独立隐私保护图文检索实验包
```

数据为 SMID 的项目衍生版本，并非官方全集；公开副本提供训练2,138图、验证134图、测试126图。两张训练图按版权记录排除，极性任务再按已有标注筛选场景。来源、逐图署名、排除记录与使用条款见[数据说明](data/README.md)、[数据版权](data/COPYRIGHT.md)及 [NOTICE](NOTICE.md)。模型与第三方材料分别遵守上游许可。

[baseline-run]: https://github.com/Muc-26302077/moral-llava-clip/blob/d791c40def8d9d211addfa832f931f68a03251c2/delivery/reports/polarity-live-run.json
[baseline-answers]: https://github.com/Muc-26302077/moral-llava-clip/blob/d791c40def8d9d211addfa832f931f68a03251c2/delivery/reports/polarity-live-answers.jsonl
[baseline-metrics]: https://github.com/Muc-26302077/moral-llava-clip/blob/d791c40def8d9d211addfa832f931f68a03251c2/delivery/reports/polarity-live-metrics.json
[baseline-log]: https://github.com/Muc-26302077/moral-llava-clip/blob/d791c40def8d9d211addfa832f931f68a03251c2/delivery/reports/polarity-live-inference.log
[token-statistics]: https://github.com/Muc-26302077/moral-llava-clip/blob/d791c40def8d9d211addfa832f931f68a03251c2/delivery/reports/current-model-token-statistics.json
[loss-curve]: https://github.com/Muc-26302077/moral-llava-clip/blob/d791c40def8d9d211addfa832f931f68a03251c2/delivery/reports/historical-training-loss.svg
[projector-loss]: https://github.com/Muc-26302077/moral-llava-clip/blob/d791c40def8d9d211addfa832f931f68a03251c2/delivery/reports/projector-loss.csv
[adapter-loss]: https://github.com/Muc-26302077/moral-llava-clip/blob/d791c40def8d9d211addfa832f931f68a03251c2/delivery/reports/adapter-loss.csv
