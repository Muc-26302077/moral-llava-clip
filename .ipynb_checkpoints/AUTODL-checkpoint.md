# AutoDL 环境

项目目录：`/root/autodl-tmp/moral-llava`。代码和数据已上传，尚未启动训练。

## 已准备

- 单张 NVIDIA A800-SXM4-80GB，支持 BF16。
- Ubuntu 22.04、Python 3.12.3。
- 复用镜像自带的 PyTorch 2.8.0+cu128、torchvision 0.23.0+cu128。
- 独立 `.venv`，安装 Transformers 4.49.0、PEFT 0.14.0、Accelerate 1.3.0。
- 数据共 2,400 张，训练／验证／测试为 2,140／134／126。
- 服务器上 34 项单元测试、依赖检查、CLI 帮助检查通过。
- CPU 梯度累积回归检查通过，确认 loss 归一化修复生效；检查没有更新参数。

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

模型、pip 缓存和临时文件均放在 `/root/autodl-tmp` 数据盘。该数据盘随实例释放的保存规则以 AutoDL 为准；实验完成后下载 adapter 和结果。

## 基模

使用 `llava-hf/llava-1.5-7b-hf`，固定版本：

```text
b234b804b114d9e37bb655e11cbbb5f5e971b7a9
```

权重约 13.16 GiB，缓存位置为 `/root/autodl-tmp/cache/huggingface`。下载和 GPU 前向检查的最终状态见本文末尾。

## 后续启动训练

以下命令会实际训练，目前没有执行。A800 先用普通 BF16 LoRA，省去 4-bit 量化；batch 4 是起始配置，训练时的显存占用仍需实际确认。

```bash
python -m moral_llava.train \
  --revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9 \
  --local-files-only \
  --output-dir runs/moral-llava-1.5 \
  --precision bf16 \
  --batch-size 4 \
  --gradient-accumulation-steps 2 \
  --eval-batch-size 4 \
  --no-gradient-checkpointing \
  --num-workers 4
```

训练默认 3 轮，按验证集 loss 选择 adapter。训练完成后，先保留原模型的测试集基线，再用同样设置评估 adapter，完整命令见 README。

## 复查

```bash
python -m unittest discover -s tests -q
python scripts/check_trainer_loss.py
python scripts/check_model_load.py \
  --revision b234b804b114d9e37bb655e11cbbb5f5e971b7a9
```

最后一项仅离线加载模型并做一次单图 BF16 前向，不反传、不更新参数、不生成答案。

## 本次准备状态

模型缓存下载中；尚未执行真实模型 GPU 前向。训练从未启动。
