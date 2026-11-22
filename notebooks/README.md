# 按测试大纲运行Notebook

打开 [测试大纲逐步复现.ipynb](测试大纲逐步复现.ipynb)，按单元格顺序运行。Notebook含29个单元格，将旧版7.3.1的4步、7.3.9的5步映射到新版合并用例XN_3_1，提供实际推理、数据校验、指标解释和结果导出。

在项目根目录执行：

```bash
source .venv/bin/activate
python -m pip install -r requirements.txt -r notebooks/requirements.txt
python -m ipykernel install --user --name moral-llava --display-name "Python (moral-llava)"
python scripts/download_models.py clip
python scripts/download_models.py moralclip
python -m jupyter lab notebooks/测试大纲逐步复现.ipynb
```

已有环境和本地模型可跳过安装、下载。本机可自动发现已有CLIP目录；新机器默认从`models/base/clip`和`models/base/moralclip`加载。Notebook的配置单元支持环境变量覆盖路径。

选择`Python (moral-llava)`内核，运行首个配置单元，检查模型和项目路径，然后执行“Restart Kernel and Run All Cells”。主流程是真实CLIP与MoralCLIP推理，支持CPU/CUDA；已有MoralCLIP权重直接加载，本次不重新微调。每次结果保存到新的`results/runs/outline-notebook-UTC时间/`，包含逐组预测、CSV主表、测试记录、环境、配置和哈希清单。

独立隐私资产通过`SMID_PRIVACY_PROJECT`指定。未提供时只跳过隐私推理，导出记录明确列出缺项。当前本机隐私数据为caption 0子集，没有覆盖大纲全部1,260条描述；已有成员攻击仅为proxy，尚无已证实的上游训练成员身份。Notebook的本地共同正确率不能替代独立合规评价或正式验收。

可选附录中的`RUN_PROJECT_METRICS=True`调用现有LLaVA/V2与检索完整流程，需要GPU和完整LLaVA基座；该入口固定是V2。V3部分读取本机最新已锁定的历史测试记录，并标注完整性、来源和未达到训练目标的状态。`RUN_PRIVACY_AUXILIARY=True`调用旧隐私包的独立辅助评测，需要其额外依赖。

大纲存在版本待核定项：新版合并节的具体步骤留空；正文阈值有60%/80%表述；原始共同正确子集的分母与全部有效输入的分母不同。Notebook明确保留这些边界，并分别显示两阈值与条件保持率。完整自测步骤、实际配置与未完成项均随每次运行导出。
