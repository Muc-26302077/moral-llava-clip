# 在服务器运行：source scripts/activate_autodl.sh
export MORAL_LLAVA_ROOT="${MORAL_LLAVA_ROOT:-/root/autodl-tmp/moral-llava}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/root/autodl-tmp/cache/pip}"
export TMPDIR="${TMPDIR:-/root/autodl-tmp/tmp}"
export TOKENIZERS_PARALLELISM=false

if [ ! -f "$MORAL_LLAVA_ROOT/.venv/bin/activate" ]; then
    echo "找不到项目虚拟环境：$MORAL_LLAVA_ROOT/.venv" >&2
    return 1
fi

# 缓存放数据盘，省点系统盘空间。
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR" "$TMPDIR"
. "$MORAL_LLAVA_ROOT/.venv/bin/activate"
cd "$MORAL_LLAVA_ROOT" || return 1
