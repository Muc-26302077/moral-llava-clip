#!/usr/bin/env sh
set -eu
script_dir=$(CDPATH= cd -P "$(dirname "$0")" && pwd)
exec "${PYTHON:-python}" "$script_dir/run.py" "$@"
