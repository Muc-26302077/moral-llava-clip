# 版权与第三方条款

本仓库包含项目代码、第三方数据的项目副本、模型适配器及评测记录。
各项材料按其原权利人规定的条款使用；本仓库未为全部内容设定统一的开源许可证。
项目新增代码和文档的版权归各自贡献者，除已注明的授权外，本次整理不新增对外许可。

| 材料 | 版权与条款入口 |
| --- | --- |
| SMID 图片、标签和衍生描述 | [数据版权说明](data/COPYRIGHT.md)；原图权利人各自保留权利 |
| MoralLLaVA LoRA、tokenizer/processor 等模型材料 | [模型说明](models/README.md)；保留基模和上游组件适用条款 |
| 原始 LLaVA 实现 | [LLaVA LICENSE](https://github.com/haotian-liu/LLaVA/blob/main/LICENSE)，代码为 Apache-2.0 |
| LLaVA-1.5-7B-HF 模型 | [指定版本模型卡](https://huggingface.co/llava-hf/llava-1.5-7b-hf/blob/b234b804b114d9e37bb655e11cbbb5f5e971b7a9/README.md)，遵守模型卡和 Llama 2 等上游适用条款 |

LLaVA 代码许可与模型权重使用条款分别适用。权重导出为 LoRA 或通过 Git LFS 保存不会改变相关限制。
已按官方 SMID 元数据为全部 2,400 张图片补入逐图归属；其中 2 张标为 All Rights Reserved、3 张保留数字许可编码，仍待核查。
衍生标签和描述的授权亦未随原包提供。当前仓库不代表已取得额外商业使用或公开再授权权利。
