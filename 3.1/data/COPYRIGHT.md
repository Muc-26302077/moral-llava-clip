# 数据版权、来源与引用

本说明针对本项目使用的 SMID 衍生数据包 `SMID_moralclip.zip`，适用于原始图片、清单、生成描述及转换数据。
本项目的整理、转换、训练和评测不改变第三方材料的著作权归属。

## 原始 SMID

SMID 由 Damien L. Crone、Stefan Bode、Carsten Murawski 和 Simon M. Laham 发布。
原始数据库及元数据入口：<https://osf.io/2rqad/>。

引用：Crone DL, Bode S, Murawski C, Laham SM (2018).
*The Socio-Moral Image Database (SMID): A novel stimulus set for the study of social, moral and affective processes*.
PLOS ONE 13(1): e0190954. <https://doi.org/10.1371/journal.pone.0190954>。

论文版权为 © 2018 Crone et al.，论文采用 Creative Commons Attribution 许可（CC BY 4.0）。
**论文的许可不等于所有图片采用同一许可。** 论文明确说明：图片来自 Wikimedia Commons、Flickr 等来源，
筛选时保留 Creative Commons 或类似宽松许可的图片；官方图片元数据包含 URL、标题、作者和许可证。
每张图片的著作权仍归其原摄影者或权利人，实际使用须遵守该图对应的署名、相同方式共享或其他适用条件。

## 当前衍生包的可核实范围

本次输入是项目工作区已有的 `SMID_moralclip.zip`。原包未附 LICENSE、COPYRIGHT、README 或逐图署名表。
本次整理另外从 SMID 官方 OSF 项目下载了 `SMID_metadata.json`：<https://osf.io/e7gej/>，
并将本项目 **2,400/2,400 个图片 ID** 与官方元数据对应，补入作者、标题、原始图片 URL、许可原文及适用许可链接。
逐图记录见 `image_attribution.csv`，官方原文副本见 `upstream/SMID_metadata.json`，下载来源与 SHA-256 见 `provenance.json`。
对应依据是原始 SMID ID；本次没有重新下载官方原图以逐字节证明图片相同，也未将元数据中的缺失值伪造为新署名。

官方元数据包含多种 CC BY、CC BY-SA、CC BY-NC、CC BY-ND、公共领域及 GFDL 等标记。
这些许可逐图适用；NC、ND、SA 和署名要求须分别遵守，不能将整个数据集按统一宽松条款再授权。
CC 许可正文可通过 CSV 中 `license_url` 查阅；GFDL 正文副本见 `upstream/`。
图片在本次整理中相对于本地输入包没有改动；输入包此前是否缩放或修改过图片尚未核实，不能声称其与原图完全相同。

官方元数据中的以下项目仍需单独核查，不能因其出现在 SMID 中就视作已获得再分发授权：

| ID | 划分 | 官方许可原文 | 当前状态 |
| --- | --- | --- | --- |
| b15_p445_5 | train | All Rights Reserved | 未确认再分发许可 |
| b15_p465_20 | train | All Rights Reserved | 未确认再分发许可 |
| b15_p345_20 | train | 10 | 数字许可编码，保留原文，待确认 |
| b10_p125_14 | validation | 10 | 数字许可编码，保留原文，待确认 |
| b15_p454_1 | test | 9 | 数字许可编码，保留原文，待确认 |

公开仓库排除上述两张标为 `All Rights Reserved` 的训练图片，明细见 `excluded_images.json`，对应原始记录和文件哈希仍保留。
本次提供 2,398 张图片，126 张测试图完整保留；完整原包和两张训练原图继续保存在服务器原项目中。
数字许可编码按官方原文记录，未擅自改写为其他许可。全量重新训练需自行按适用许可取得缺失图像。
原包此前对原图的变换及衍生标签/描述的具体授权仍须由相应提供者确认；本说明不新增任何再授权。

## MoralCLIP 相关材料与本项目改动

本地数据包名称含 `moralclip`；相关方法参考：<https://arxiv.org/abs/2506.05696>，
公开项目：<https://github.com/anaacondez/moralclip>。
仅凭压缩包名称无法证明其由该论文作者官方发布。五维离散标签、生成描述和衍生包本身的具体权利及分发条款，
仍需由包提供者或相应权利人确认。引用论文不构成额外授权。

本项目保留原始图片、标签、描述和划分，仅转换了供 LLaVA 使用的指令 JSON，
并生成固定评测分组、预测与统计结果。没有修改原图内容。
若使用原图、标签、描述或本项目转换数据，请引用 SMID，并注明所使用的衍生包和本仓库的具体 commit。
不得将整理者列作原始图片作者。

`references.bib` 提供已核实的 SMID 引用；`provenance.json` 提供来源与输入哈希。
如权利人认为某项材料的归属或分发不当，可联系仓库维护者或通过仓库 Issue 指出对应 ID，供维护者核查并处理。
