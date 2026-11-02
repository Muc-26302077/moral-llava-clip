# AutoDL 环境

项目目录：`/root/autodl-tmp/moral-llava`。正式 3 轮 BF16 LoRA 训练及测试集推理、评估已完成。测试集 126 条输出均有效，micro-F1 为 0.7796，exact match 为 70／126（55.56%）；详细结果见 [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md)。

## 训练前准备

- 单张 NVIDIA A800-SXM4-80GB，支持 BF16。
- Ubuntu 22.04、Python 3.12.3。
- 复用镜像自带的 PyTorch 2.8.0+cu128、torchvision 0.23.0+cu128。
- 独立 `.venv`，安装 Transformers 4.49.0、PEFT 0.14.0、Accelerate 1.3.0。
- 数据共 2,400 张，训练／验证／测试为 2,140／134／126。
- 服务器上 34 项单元测试、依赖检查、CLI 帮助检查通过。
- CPU 梯度累积回归检查通过，确认 loss 归一化修复生效；检查没有更新参数。
- 真实 processor 完整检查了训练与验证的 2,274 条样本：每条 576 个图片 token，总长度 810–825，答案与 EOS 的监督长度 36–51，无超长样本。

不要在这台镜像上直接安装 `requirements-train.txt`，它会替换镜像的 PyTorch。重新创建环境时用：

```bash
cd /root/autodl-tmp/moral-llava
/root/miniconda3/bin/python -m venv --system-site-packages .venv
source scripts/activate_autodl.sh
python -m pip install -r requirements-autodl.txt
```

日常登录后只需：

```bash
cd /root/autodl-tmp/moral-llava
source scripts/activate_autodl.sh
```

模型、pip 缓存和临时文件均放在 `/root/autodl-tmp` 数据盘。该数据盘随实例释放的保存规则以 AutoDL 为准；adapter 与结果需另行保存到本地。

## 基模

使用 `llava-hf/llava-1.5-7b-hf`，固定版本：

```text
b234b804b114d9e37bb655e11cbbb5f5e971b7a9
```

权重约 13.16 GiB，缓存位置为 `/root/autodl-tmp/cache/huggingface`。三个权重分片的大小与 SHA256 已通过校验，训练和推理均使用本地缓存。

需要重新下载时，当前实例可直连镜像；不必叠加 `/etc/network_turbo` 的代理。只影响当前命令：

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
  HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 \
  HF_HUB_ENABLE_HF_TRANSFER=1 \
  python scripts/download_model.py --report model_download_check.json
```

离线复查权重完整性可运行 `python scripts/download_model.py --verify-only --report model_download_check.json`，会逐块检查三个分片的 SHA256。

## 正式训练已完成

本轮使用普通 BF16 LoRA，未启用 4-bit 量化：训练 batch 8、梯度累积 1（有效 batch 8）、验证 batch 8，关闭梯度检查点，seed 42。正式训练从固定版本基模开始，没有从两步短训练的 checkpoint 恢复。

训练完成 3 epochs、804 steps，训练耗时 1,350.5993 秒（约 22.5 分钟）。各轮验证损失如下，最终选择 `checkpoint-804`，测试集不参与选择：

| Epoch | 验证 loss |
| ---: | ---: |
| 1 | 0.026465144008398056 |
| 2 | 0.026325300335884094 |
| 3 | 0.02562384493649006 |

`nvidia-smi` 每 2 秒采样记录的最大显存占用为 73,073 MiB（约 71.36 GiB）。这是采样最大值，并非 PyTorch 记录的精确峰值。

运行文件均位于服务器项目目录下：

| 用途 | 远程路径 |
| --- | --- |
| 实验 runner | `/root/autodl-tmp/moral-llava/runs/experiment_runner.py` |
| 阶段状态 | `/root/autodl-tmp/moral-llava/runs/experiment_status.json` |
| 训练及后续评估日志 | `/root/autodl-tmp/moral-llava/runs/logs/experiment.log` |
| GPU 采样 | `/root/autodl-tmp/moral-llava/runs/logs/experiment_gpu.csv` |
| 模型版本、文件哈希与推理配置 | `/root/autodl-tmp/moral-llava/runs/experiment_manifest.json` |
| 正式 checkpoint 与 adapter 输出 | `/root/autodl-tmp/moral-llava/runs/moral-llava-1.5` |

本次任务由 runner 完成。以下离线命令保留本轮参数；重新训练时必须将 `--output-dir` 改为新的空目录，例如 `runs/moral-llava-1.5-rerun`，以保留已有结果：

```bash
cd /root/autodl-tmp/moral-llava
source scripts/activate_autodl.sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m moral_llava.train \
  --revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9 \
  --local-files-only \
  --output-dir runs/moral-llava-1.5 \
  --epochs 3 \
  --precision bf16 \
  --batch-size 8 \
  --gradient-accumulation-steps 1 \
  --eval-batch-size 8 \
  --no-gradient-checkpointing \
  --num-workers 4 \
  --seed 42
```

每轮按验证集 loss 保存 checkpoint，最终选择验证损失最低的 adapter。runner 已在训练后完成 126 张测试图的 adapter 推理与评估；与零样本基线固定同一基模 revision，均为 BF16、无量化、batch 8、贪心生成、最多生成 256 tokens。复现命令见 README。

## 测试结果与本地文件

微调后 126／126 条输出有效（100%），micro-F1 为 0.7796101949025487、macro-F1 为 0.7539145311004785、样本平均 Jaccard 为 0.7247354497354497，exact match 为 70／126（55.56%）。原模型零样本输出 126 条均未通过本项目的结构化协议校验，按严格评估规则各指标为零；仅根据训练标签选择的多数类基线 micro-F1 为 0.17218543046357616，exact match 为 39／126（30.95%）。完整比较与适用范围见 [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md)。

`runs/finetuned_metrics.json` 及运行日志已保存到本地。adapter 本地备份已完成：`runs/moral-llava-1.5-adapter.tar.gz` 已完整下载并解压到 `runs/moral-llava-1.5`，14 个文件的 SHA256 均与服务器一致，校验报告为 `runs/local_artifact_check.json`。

## 已通过的短训练检查

正式训练前，已在独立目录 `runs/smoke-bf16` 完成 `--max-steps 2` 短训练，反传、参数更新、验证和 adapter 保存均通过。随后从该 adapter 重载并完成单图生成，结果保存在 `runs/smoke_reload.jsonl`。这些检查只验证训练与保存／重载流程，不作为正式实验成绩。

## 复查

```bash
python -m unittest discover -s tests -q
python scripts/check_trainer_loss.py
python scripts/check_model_load.py \
  --revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9
```

最后一项仅离线加载模型并做一次单图 BF16 前向，不反传、不更新参数、不生成答案。

## 准备阶段检查记录

基模已完整缓存，三个权重分片大小和 SHA256 均通过校验。单张图片的离线 BF16／SDPA 前向检查通过：输入 825 tokens（含 576 个图片 token），loss 有限；语言层 LoRA 共 39,976,960 个可训练参数，视觉与基础语言权重保持冻结。该次检查峰值 allocated 显存约 14.05 GiB，仅代表单图无梯度前向，不代表训练显存。

准备阶段记录的数据盘剩余约 87 GiB。服务器保存了 `model_download_check.json`、`processor_check.json`、`model_forward_check.json` 和 `deployment_environment.json`，可复核环境与检查结果。上述准备检查本身没有反传或更新参数；之后已完成短训练、零样本评估，以及正式训练和微调后评估。正式运行的显存采样见 `runs/logs/experiment_gpu.csv`，与单图无梯度前向的记录应分开解读。
