#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$project_root"
repository="Muc-26302077/moral-llava-clip"
command -v gh >/dev/null || { echo '请先安装 GitHub CLI。' >&2; exit 1; }
git lfs version >/dev/null || { echo '请先安装 Git LFS。' >&2; exit 1; }
gh auth status --hostname github.com >/dev/null 2>&1 || {
  echo '请先运行 gh auth login --hostname github.com --web。' >&2; exit 1;
}
can_push="$(gh api "repos/$repository" --jq '.permissions.push')"
[[ "$can_push" == true ]] || { echo '当前账号没有目标仓库写权限。' >&2; exit 1; }
python 3.1/scripts/verify_release.py
git lfs install --local
git lfs fsck
if [[ -n "$(git status --porcelain)" ]]; then
  echo '存在未提交修改，请先检查并提交。' >&2; exit 1
fi
if git remote get-url origin >/dev/null 2>&1; then
  remote_url="$(git remote get-url origin)"
  if [[ "$remote_url" != "https://github.com/$repository.git" &&
        "$remote_url" != "https://github.com/$repository" &&
        "$remote_url" != "git@github.com:$repository.git" ]]; then
    echo 'origin 指向其他仓库，请先检查。' >&2; exit 1
  fi
else
  git remote add origin "https://github.com/$repository.git"
fi
gh auth setup-git --hostname github.com
git push --set-upstream origin main
local_commit="$(git rev-parse HEAD)"
remote_commit="$(git ls-remote origin refs/heads/main | cut -f1)"
[[ "$local_commit" == "$remote_commit" ]] || { echo '远端提交校验失败。' >&2; exit 1; }
printf '上传完成：https://github.com/%s\ncommit: %s\n' "$repository" "$local_commit"
printf '成员管理：https://github.com/%s/settings/access\n' "$repository"
