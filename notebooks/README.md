# 按1117v2大纲7.3.1运行Jupyter

当前入口：[7.3.1_测试步骤_1117v2.ipynb](7.3.1_测试步骤_1117v2.ipynb)。依据用户2026年11月24日提供的`2025ZD0123500-赫然-多模态内生安全测试大纲1117v2.docx`第7.3.1节编写，共6个测试步骤、19个单元格，其中10个代码单元。

1. 检查环境、模型、视觉投影器、语言LoRA与推理参数。
2. 核验原始和隐私SMID图文、标准标签、配对、数量和异常。
3. 核验LDP-Slicing、CluSanT-adapted执行证据与选定模型的微调记录。
4. 固定同一最终MoralCLIP–LLaVA，分别评测原始/隐私图文问答并计算TICR。
5. 真实运行MoralCLIP原配图文检索，分别计算原始/隐私双向R@1。
6. 统计两种条件下图像/文本二分类准确率，导出逐组判定、CSV和用例记录。

在项目根目录执行：

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt -r notebooks/requirements.txt
python -m ipykernel install --user --name moral-llava --display-name "Python (moral-llava)"
python -m jupyter lab --allow-root notebooks/7.3.1_测试步骤_1117v2.ipynb
```

选择`Python (moral-llava)`内核，检查配置单元后依次运行。已有环境可跳过安装。新机器须先`git lfs pull`，并通过`python scripts/download_models.py llava`和`python scripts/download_models.py moralclip`下载固定基座。

`RUN_MODE='infer'`在CUDA/BF16环境重新生成两套各576条回答，两个条件使用完全相同的适配器。`RUN_MODE='archive'`复算已有原始/隐私回答，逐行核验最终适配器、视觉权重、基座、数据与模板指纹；不匹配时停止。默认`auto`有GPU选择infer、无GPU选择archive。步骤5另由`RETRIEVAL_MODE`控制：资源充足时真实编码；`auto`在无GPU且cgroup总内存不足4GiB时明确列为待运行，`skip`可手动跳过，`infer`启用真实编码。不把归档回答称为本次重新推理。

本机默认`MODEL_PROFILE='privacy-trained'`，使用独立隐私项目`runs/adapter`的最终轻量微调权重，对应两套已有完整回答；它与主项目干净数据训练基线、BLIP融合模型分别标识。切换`project-baseline`或自定义适配器时，两路均必须切换，并提供匹配归档，或用GPU重新推理。默认数据为LDP-Slicing＋CluSanT-adapted、固定median3图像后处理及caption 0，不混用BLIP融合版本。

外部隐私实验目录通过`SMID_PRIVACY_PROJECT`指定，运行模式也可用`OUTLINE731_MODE`和`OUTLINE731_RETRIEVAL`环境变量指定。还支持`POLARITY_ADAPTER`、`SMID_PRIVATE_ROOT`、`SMID_PRIVATE_QA`、`QA_ARCHIVE_CLEAN`和`QA_ARCHIVE_PRIVATE`。这些隐私资产及隐私适配器未随Git发布；缺少时Notebook给出对应路径错误，不替换为其他模型或数据。基座路径可通过`LLAVA_MODEL_DIR`和`MORALCLIP_MODEL_DIR`覆盖。

主表TICR按项目既有五维宏平均合法答案一致率计算；同时报告两路均命中标准标签的共同正确率，保留大纲“合规且一致”与项目TICR定义的区别。检索原始/隐私统一使用126个场景、各1条caption 0，Top1按原配场景ID判定；原有道德标签相关命中率单列辅助。测试记录分别显示60%一致率/检索门槛、50%准确率门槛和实测未达标项，不从预期结果填数。

每次输出到新的`results/runs/section731-1117v2-北京时间/`，包含配置、环境、指纹、训练/隐私证据、两套原始回答、逐组判定、检索矩阵、Top1、CSV和`7.3.1测试记录.md`。正式指标口径、检索候选范围和方向仍需在正式测试前核定。

旧版[测试大纲逐步复现.ipynb](测试大纲逐步复现.ipynb)保留为1102版历史流程，使用CLIP/MoralCLIP零样本条件极性自测。它不作为本次1117v2第7.3.1节的运行入口。
