# 团队协作

本项目采用 GitHub 仓库协作，仓库管理员可添加其他成员。
创建仓库后进入 **Settings → Collaborators**（组织仓库通常显示 **Collaborators and teams**），
选择 **Add people**，输入成员 GitHub 用户名并发送邀请；成员接受后即可按权限参与。
多人长期维护时可将仓库放在 GitHub Organization 中，通过团队统一管理 Write/Maintain 权限。
当前尚未提供成员账号，因此本次不预先邀请任何成员。

新内容按研究章节存放在根目录；本组内容固定在 `3.1/`；`3.7/` 为预留指标大类，当前仅含 `.gitkeep`。
`3.1/benchmark/` 本次保留空白，后续 benchmark 在该目录补充。

修改前创建分支，提交后通过 Pull Request 合入主分支。
新增数据需带来源和适用许可；新增模型权重需更新模型卡和校验清单。
大文件先配置 Git LFS，禁止将 token、私钥或本机缓存加入仓库。
现有训练记录、固定协议和原始预测应保留原样，新实验请创建新目录。
