# 远端仓库独立运行与旧指标核验

2026-11-03（Asia/Shanghai）。从 GitHub 实际重新克隆代码、下载 LFS 权重，并在独立目录完成主交付模型全量推理。结论：远端主模型在旧版五维完整图文一致率上达到 60%，但整场景五维完全正确率没有达到 60%。最新 QA v1 小样本模型未上传远端，也没有旧版指标的实测结论。

远端：https://github.com/Muc-26302077/moral-llava-clip

- `main`：`13bc54ae3d1c0b5ce92a1a0357f3cba5f5cc5dbb`，主交付模型为 `3.1/models/moralllava/`。
- `codex/binary-moral-qa-v1`：`75bdf00f08c4788249860bb89dfb0a67a7496614`，新增二分类代码，仍保留旧模型。
- 新小样本 adapter 仅在本地 `runs/binary-small-20261103T022000Z/adapter/`，不能从当前远端获取。

本次独立克隆在 `/tmp/moral-llava-remote-audit-20261103/`。两份 adapter 从 GitHub LFS 实际下载，未从原项目复制；均为 159,974,600 字节，SHA-256 分别为：

| 模型 | SHA-256 |
|---|---|
| 主交付图文联合 LoRA | `22cbec8ded2c355f113cf9d55b0cd98e83372e2133b42af41918a09ae8e20fbb` |
| 旧图像 LoRA | `8c7adaa741656e41f34434c347624844dcaa5a61589c6c0fef5274cc005eeb02` |

基模使用已有 Hugging Face 缓存，但由远端代码验证全部三个分片的固定 SHA-256。revision 为 `b234b804b114d9e37bb655e11cbbb5f5e971b7a9`；没有使用原项目内的私有模型替代远端权重。

已完成的独立检查：

1. 在独立克隆执行 `python 3.1/run.py score`，重算已有原始回答，指标与说明一致。
2. 执行 `python 3.1/run.py verify`，结果 `PASS`，核验 2,540 个发布文件、2,398 张公开图片、完整 126 张测试图及模型来源和选模记录。
3. 入口迁移测试 4/4 通过，包括从其他工作目录调用、带空格路径及基模哈希校验。
4. 从 `/tmp` 调用远端 `run.py evaluate --limit 2 --decoding constrained`，基模与主模型的真实图文 GPU 短推理通过。
5. 使用远端推理模块和刚下载的主模型，重跑完整 126 场景、两种模态、自由／约束两种解码，共 **504 条新生成回答**。每种解码均完整覆盖 252 个 ID／模态对，与仓库参考回答逐条对比，**原始文本差异 0、结构化预测差异 0**。迁移协议仅图片绝对路径改变，场景、分组、描述、标签和问题哈希一致。另行从原始回答计算以下旧指标，与评分文件一致。

| 旧版指标 | A：63 场景 | B：63 场景 | 合计：126 场景 |
|---|---:|---:|---:|
| 五维完整图文一致率 | 68.25%（43/63） | 60.32%（38/63） | 64.29%（81/126） |
| 图片五维完全正确率 | 57.14%（36/63） | 52.38%（33/63） | 54.76%（69/126） |
| 文字五维完全正确率 | 50.79%（32/63） | 49.21%（31/63） | 50.00%（63/126） |
| 两路五维都完全正确率 | 42.86%（27/63） | 41.27%（26/63） | 42.06%（53/126） |
| 图片逐维平均准确率 | 82.22% | 82.22% | 82.22% |
| 文字逐维平均准确率 | 75.24% | 73.97% | 74.60% |

6. 从远端 QA v1 提交建立另一独立检出，专项测试 15/15 通过；用其公开数据执行 `run_binary.py prepare`，产生训练／验证／测试 2,138／134／126 场景、25,656／1,608／1,512 条问答，与该分支说明一致。该分支的真实新小样本模型成绩不在远端发布内容中。

适用范围：本次 GPU 为 A800，Python 3.12.3、PyTorch 2.8.0+cu128、Transformers 4.49.0、PEFT 0.14.0、BF16；完整主模型复跑 batch 为 8。此次未重新创建仓库声明的 Python 3.11／PyTorch 2.6 Conda 环境，也没有完整重跑原始基模的 126 场景对照；不将其写成这些项目的验收结论。本次主交付模型的全量回答已经与参考逐条相同，但不能承诺任意硬件、依赖版本或重新训练都逐条复现。公开训练集另排除两张图片，从公开数据重新训练也不等同于原 2,140 场景训练。

独立评测不需要原服务器业务目录，但需要 Git LFS 获取 adapter、固定版本 LLaVA 基模、可用依赖和支持 BF16 的 CUDA GPU。普通克隆默认进入旧版 `main`；QA v1 需显式切换对应分支。模型结构和超参数相同不构成准确率保证，只有指定权重与固定协议的实测结论。

证据：

- [全量逐条核验与独立重算](../results/reference/remote-repro-audit-20261103/verification.json)
- [发布验证 PASS](../results/reference/remote-repro-audit-20261103/verify.json)
- [远端 LFS 实际下载记录](../results/reference/remote-repro-audit-20261103/lfs-download.json)
- [完整主模型推理日志](../results/reference/remote-repro-audit-20261103/full-lora-replay.log)与[完成状态及环境](../results/reference/remote-repro-audit-20261103/full-lora-replay/status.json)
- [远端 QA v1 测试日志](../results/reference/remote-repro-audit-20261103/qa-tests.log)与[公开数据准备结果](../results/reference/remote-repro-audit-20261103/qa-prepare.json)

此次未改动或推送远端仓库。
