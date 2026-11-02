# 固定成对道德判断评测（v1）

## 目标与协议

复用未改动的 `runs/moral-llava-1.5`，比较原始 LLaVA 与旧版图像标签 LoRA 在同一场景图像输入、纯文本输入下的道德判断。不训练、不更改权重、不覆盖历史预测。

- 固定清单：`runs/crossmodal-v1/protocol.json`。
- 原始 126 张 SMID 测试图划为 A/B 两个不重叠子组，各 63 张。按五维完整标签组合分层，seed 42 的 SHA256 顺序，依次分配到当前较小组，平票选 A。它们属于同一原始划分，不是两个外部独立基准。
- 每张图固定原始生成描述 `captions[0]`，不根据结果挑选描述；原始图片标签作为参考，不传入模型。
- 图像输入：原任务问题 + 576 个展开图像 token + 图片像素。
- 文本输入：场景描述 + 同一标签约束问题（仅将 this image 改为 this scene），无图像 token，无图片像素。
- 现有描述为生成文本，尚未逐条人工核验其是否完整保留道德相关信息。当前结果是现有数据上的探索性一致性测量。

## 指标与门槛

主指标：两种输入均有效且五维状态完全一致的场景数 / 全部场景数。无效输出计失败，两边都解析失败也不算一致。`non_moral` 由五维推导，不重复作为第六维加权。

阶段门槛：LoRA 在 A、B 每组的主指标均 >= 60%；每组至少 38/63。门槛采用点估计，并同时报告 Wilson 95% 区间，不能将点估计达标写成区间下界达标。

补充指标：逐维一致率（分母 5N）、有效输出率、各模态与图片标签的 exact match / Micro-F1 / Macro-F1、两种输入同时正确率。

条件保持率另行定义：基模已经图文一致的场景中，LoRA 仍然图文一致的场景比例。基模一致场景为 0 时该比例为 null，不当作 0 或 100%。它不替代主一致率门槛。

一致不能等同正确：恒定输出也能取得 100% 一致率，必须同时查看标签准确率及两边同时正确率。`not_annotated` 只是数据未标注，不表示客观中性。

## 两种解码分别报告

推理前的 `runs/crossmodal-v1/preflight.json` 已固定：

1. **主分析 constrained**：基模和 LoRA 使用完全相同的合法 JSON token 树，包含全部 3^5=243 种标签组合，non_moral 严格派生。不读取参考标签，不在生成后修补输出。它是 token 级贪心约束生成，不是全答案最大似然分类。
2. **补充诊断 free**：相同问题与贪心生成参数，不加输出 token 约束；保留所有格式失败。用于观察格式遵循问题，不与 constrained 混算。

约束格式本身不保证准确性或跨模态一致性。两种协议分别报告，不能看到结果后切换主分析来达标。

## 执行

项目根目录内运行，全部使用本地权重：

```bash
# 清单已生成，无需重复 prepare；该命令会拒绝覆盖。
.venv/bin/python -B -m moral_llava.crossmodal prepare

# 原始模型和旧 LoRA × 两种输入 × 两种解码；输出目录必须不存在。
.venv/bin/python -B -m moral_llava.infer_crossmodal \
  --device cuda --precision bf16 --batch-size 8 --decoding both \
  --output runs/crossmodal-v1/full

# 从逐条原始输出独立重算主分析，不覆盖运行器保存的结果。
.venv/bin/python -B -m moral_llava.crossmodal score \
  --protocol runs/crossmodal-v1/protocol.json \
  --base-predictions runs/crossmodal-v1/full/base_constrained.jsonl \
  --lora-predictions runs/crossmodal-v1/full/lora_constrained.jsonl \
  --output runs/crossmodal-v1/recomputed_constrained.json

# 仅验证评分/分组，不训练或加载模型。
.venv/bin/python -B -m unittest discover -s tests -p test_crossmodal.py -v
```

每个模型/解码结果文件应有 252 条预测（126 场景 × 2 模态），四个文件合计 1008 条。所有 ID 与模态必须唯一且完整覆盖，否则评分拒绝执行。原始回答、解析结果、输入协议哈希、每条 prompt 哈希、生成 token 数和 EOS 状态均保留。

`--limit` 仅用于烟雾检查，不生成基准指标。1-token CPU 检查和失败后修正的 GPU 检查均是调试记录，不能当正式结果。

## 环境和溯源

当前 CUDA 已确认可用：NVIDIA A800-SXM4-80GB。新环境 torch 2.6.0+cu124、transformers 4.49.0、peft 0.14.0；历史实验 torch 2.8.0+cu128。两模型在本轮相同环境重新推理，不直接拼接历史图片输出和当前文本输出。运行状态记录模型版本、源码/协议/adapter 哈希及软件版本。
