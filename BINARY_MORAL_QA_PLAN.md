# 第一阶段：六项是/否问答；第二阶段：视觉适配可行性

日期：2026-11-03（Asia/Shanghai）。第一阶段代码已新增，正式数据转换、7B 模型推理及训练尚未执行。只允许短 CPU 小样例验证，不加载真实 7B 模型做 CPU 评测。第二阶段目前仅为设计，必须等待第一阶段真实评测过关后再实现。

## 1. 可行性与任务定义

把一次五维三状态 JSON 改为逐项是/否问答，在原 LLaVA 生成模型上可行。它减少输出格式负担，合并了正负极性，任务难度降低；指标上涨不能单独说明原五维道德能力提升。原五维输出及 80% 共同正确/90% 一致性条件蒸馏方案保留，新增流程不修改它们的标签或历史预测。

按用户确认，采用 **五个道德基础是否被标注＋non-moral 辅助项**，每个场景都生成每项问题及对应正/负样本，不做互斥六分类：

| task | 是 | 否 |
| --- | --- | --- |
| care | Care 或 Harm 有标注 | 两者均未标注 |
| fairness | Fairness 或 Cheating 有标注 | 两者均未标注 |
| loyalty | Loyalty 或 Betrayal 有标注 | 两者均未标注 |
| authority | Respect 或 Subversion 有标注 | 两者均未标注 |
| sanctity | Sanctity 或 Degradation 有标注 | 两者均未标注 |
| non_moral（辅助） | 原标签为 non-moral | 至少一个道德维度有标注 |

提问必须明确“是否涉及关怀/伤害（Care/Harm）维度”，不能把 Harm 当成 Care 正极性。否代表未标注，不代表客观无道德含义。该映射不能还原原来的 virtue/vice。

问题示例：根据本数据集的标注，以下图片中的场景是否涉及关怀/伤害（Care/Harm）维度？任一极性被标注即为是。未标注不表示现实中的道德中性。只回答是或否，不要解释。

文字路替换为“以下文本描述”，加入固定 caption 0，且不含图像 token；图片路只有图片和问题，不加入描述。答案统一为中文“是”或“否”，只有外围空白可忽略，标点、解释、英文 Yes/No 均不自动修补。

## 2. 数据拆分与原结构保持

完整服务器数据保留 2140/134/126 原场景划分，不重新划分 caption 或问题。每场景 6 项 × 2 模态 = 12 条，完整数据 train/validation/test 共 25680/1608/1512 条问答。每个 task 另存独立 split 文件，便于分任务训练或查看；默认使用同一模型混合训练全部问题。

GitHub 公开副本仍遵守既有排除清单：两张训练图不打包，不从服务器补传。显式读取 data/excluded_images.json 后，公开版本训练场景为 2138、问答 25656；验证与测试完整。转换报告记录排除 ID、每类正负数量、来源哈希和实际场景数量。未声明的缺图直接报错。

新数据写入 data/binary-v1，不覆盖 data/processed。生成器保留 scene_id、task、modality、caption_index、split 和原 A/B group，所有同场景问题始终在同一划分。原 A/B 各 63 场景从冻结协议复用，不按新预测重新分组。源数据校验全部完成后才建立输出目录，已存在则拒绝覆盖。

第一阶段使用原 LLaVA-1.5-7B 的原视觉塔、原投影器、同一语言解码器及原词表生成头；不增加分类预测头、不使用前轮条件蒸馏、不更换视觉塔。训练仅更新原语言 LoRA，视觉塔与投影器冻结。默认继续旧图像 LoRA，也可显式选择模型/adapter 作基线；当前图文联合 LoRA与二分类 SFT 必须分目录记录。

只监督答案及 EOS，问题、图像和 padding 全部屏蔽。训练记录混合图片和纯文本；文字路没有像素，图片实际进入 decoder。保留原 SFT 的验证 loss 选模规则，不用测试匹配率选择 checkpoint。正式训练建议 3 epochs、BF16、batch 4、梯度累积 2、LR 2e-5、seed 42；先做 GPU 2 步烟雾检查后才执行，当前不运行。

## 3. 第一阶段评测与过关条件

主分析是原生自由生成，是/否来自原始输出；没有合法输出树强制一致。旧五维结果不能直接转换成二分类后冒充新提示下的模型推理。

每个主任务分别统计图片准确率、文字准确率、balanced accuracy、阳性 F1、有效率、图文匹配率、共同正确率、共同错误数及常量输出标记。非道德辅助项单列，不进入主分数，避免重复的高频标签抬高分数。

**主要匹配率：五个主任务各自图文匹配率的宏平均。** A/B 分开判定均 >=60%；等价于各组 63×5=315 个场景-问题对至少 189 个一致。它不同于原来“五维全部相同才成功”的指标。另报每场景五个 presence 判断全部一致率、全部共同正确率，保留严格度对照，但不能与旧三状态指标视为相同任务。

仅匹配率过关不足以进入第二阶段。附加正确性防退化条件：两路在全部测试场景的五任务 macro balanced accuracy 均 >0.5。训练集众数常量对照必须一起报告；它可获得 100% 一致率，但不通过这一正确性条件。该最低防退化条件不是 80% 正确率承诺。

正式报告要求原 126 场景、完整五任务及 A/B 各 63 场景。小样例和 --limit 输出只检查流程，禁止写成正式达标结论。缺失/重复 ID、模态不配对、元数据或问题哈希变化直接拒绝评分；无效答案留在分母，两路都无效不计一致。各任务的区间以场景为单位，相关的五任务宏平均不冒充 630 个独立场景。

先运行已有权重的新二分类提示基线，必要时按同一架构完成二分类 SFT；配置在训练/验证上确定，测试仅对选中权重作最后比较。基线与新模型各自是独立推理记录，测试结果不得驱动追加调参。

## 4. 新增入口与未来执行命令

GitHub 仓库在 github-project/，正式项目路径为 3.1/；服务器根目录保留同样的新增模块供本地数据使用。原 3.1/run.py 继续执行旧任务，新增 3.1/run_binary.py。以下从 GitHub 仓库根目录运行，**这里只提供命令，未执行真实模型训练或推理**：

```bash
# CPU 数据转换，输出新目录；自动使用公开排除清单和冻结 A/B 分组
python 3.1/run_binary.py prepare

# 无 GPU 的训练集众数对照：只能检验评测及类别分布
python 3.1/run_binary.py majority --output 3.1/results/runs/binary-majority.jsonl
python 3.1/run_binary.py score --predictions 3.1/results/runs/binary-majority.jsonl \
  --output 3.1/results/runs/binary-majority-metrics.json

# GPU 恢复后：旧图像 LoRA 的新问题基线，默认每次生成最多 8 tokens
python 3.1/run_binary.py infer --device cuda --precision bf16 --batch-size 4 \
  --local-files-only --output 3.1/results/runs/binary-image-lora-baseline.jsonl

# GPU 恢复后：只改数据和语言 LoRA，不改原 LLaVA 架构
python 3.1/run_binary.py train --epochs 3 --batch-size 4 --eval-batch-size 4 \
  --gradient-accumulation-steps 2 --learning-rate 2e-5 --seed 42 --precision bf16 \
  --local-files-only --output-dir 3.1/results/runs/binary-sft-v1
python 3.1/run_binary.py infer --adapter 3.1/results/runs/binary-sft-v1 \
  --device cuda --precision bf16 --batch-size 4 --local-files-only \
  --output 3.1/results/runs/binary-sft-v1-test.jsonl
python 3.1/run_binary.py score --predictions 3.1/results/runs/binary-sft-v1-test.jsonl \
  --output 3.1/results/runs/binary-sft-v1-metrics.json
```

可选 --data 指向 tasks/care/test.json 作单任务诊断，但单任务结果不获得完整基准通过标记。训练与推理 --help 无需模型权重，准备/评分/众数对照仅用标准库。真实模型训练仍要求 CUDA。没有 GPU 时只运行新单元测试和两层随机小模型的混合模态前向/反向，CPU 上限 5 分钟，明显低于用户 40 分钟限制；不运行 7B CPU 评测。

## 5. 当前视觉结构核查

核查实际代码及 run_config 后，需要区分两条历史分支：

| 分支 | 视觉塔 | 视觉塔是否训练 | 图片是否进入语言 decoder |
| --- | --- | --- | --- |
| 旧图像 LoRA / 最新图文联合 LoRA / 本次第一阶段 | 原 CLIP ViT-L/14@336，1024维，24层 | 冻结 | 是，经原投影器扩为 576 图像 tokens |
| 历史 alignment-moral-b16 检索 | 换成 CLIP ViT-B/16@224，768维，新投影器 | 冻结；投影器参与训练 | 检索图像路不进入 decoder，投影后池化生成检索向量 |

因此，“我们修改过视觉编码器”在历史检索分支指的是换塔及新建投影器，并不等于已经训练视觉塔；当前标签生成主线仍使用原视觉塔。依据为 moral_llava/alignment.py 的 vision_model_id 分支、runs/alignment-moral-b16/run_config.json，以及 train_crossmodal.py 的语言 LoRA 冻结检查。

## 6. 第二阶段可行性与建议架构（尚不实现）

**可行：可以训练视觉编码器的少量参数，使其更容易通过固定语言解码器得到与文字一致且正确的标签。** 先复用第一阶段原 ViT-L/14，避免同时换模型与训练目标；不把检索用 B/16 adapter 直接加载到当前图像生成路径。

推荐结构：

```text
图片 -> 原 ViT-L/14（仅最后2个block的视觉LoRA可训练）
     -> 原多模态投影器（小学习率可训练）
     -> 第一阶段共享语言 decoder + 语言LoRA（冻结）
     -> 原生成头 -> 是/否
                       ^
文字 -> 同一冻结语言 decoder + 语言LoRA -> 文字概率/语义特征
                       | stop-gradient 教师指导图片
```

图片训练始终保留真实答案 CE。配对文字在训练集上该二分类答对且置信度 >=0.7 时，增加弱的文字到图片 KL；不合格就关闭这项指导。可另比较共享 decoder 内同一问题下的末端语义特征余弦对齐；不能直接把 1024 维视觉 patch 特征与 4096 维语言向量逐元素作差。具体概率需按完整“是/否”候选序列评分，不能假设中文答案一定单 token。

冻结语言参数仍允许图像梯度穿过 decoder 回传给视觉 LoRA 和投影器；激活显存开销依然存在，必须 GPU 烟雾检查。只有视觉最后两层的小幅适配无法保证改善，应与“仅投影器训练”的控制组比较，并监控图片准确率及非道德常量退化。

互相指导在技术上也可实现，但只有一侧经真实训练标签验证为更可靠时才向另一侧蒸馏，并采用 detach/EMA 教师；一开始双向互追容易形成共同错误。若同时放开语言 LoRA，则教师会漂移，需要另一轮独立消融，不混入首轮视觉适配。先做文字指导图片，效果明确后再考虑双向。

文字路冻结时，它的正确率是图文共同正确率的上限；若文字不够准确，训练视觉侧无法独自实现高共同正确率。因此，必须在第一阶段先查文字是否可靠，而不是看到 60% 匹配就假定文字能当强教师。

**启动条件：第一阶段真实完整评测通过匹配与防退化条件，并审核分歧/共同错误后，再制定、实现第二阶段实验。当前不改视觉编码器代码。**

## 7. 版本管理与恢复

实际 GitHub 检出位于 github-project/，原 main 为 13bc54ae3d1c0b5ce92a1a0357f3cba5f5cc5dbb。已建立 backup/pre-binary-v1-20261103 标签和 codex/binary-moral-qa-v1 功能分支，不改 main。服务器工作根目录另有旧方案快照 e1eaf51；它包含上一轮条件蒸馏方案及配置。

Git bundle 为完整 Git 历史备份；模型的 Git LFS 实体不在 bundle 内，原主线模型继续保留。已有 2026-11-02 文件级快照也保持不变。生成数据与训练输出均放入忽略目录，不提交原始图片或模型权重。

需要回到改动前方案时，优先在新目录创建工作树，不覆盖当前修改：

```bash
git -C github-project worktree add ../moral-llava-pre-binary backup/pre-binary-v1-20261103
```

新工作树的 Git LFS 权重需要原本地对象或 git lfs pull；服务器既有模型路径仍可继续使用。功能分支最终提交号、验证用时与远程同步状态在交付时报告；未推送时不能声称 GitHub 已更新。
