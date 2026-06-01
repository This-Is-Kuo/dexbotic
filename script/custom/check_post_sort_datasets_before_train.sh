#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/post_sort_common.sh"
PROJECT_ROOT="$(post_sort_project_root)"
cd "${PROJECT_ROOT}"

echo "[INFO] Checking post-sort DM0 dexdata before training."
echo "[INFO] PROJECT_ROOT=${PROJECT_ROOT}"
python3 script/custom/check_post_sort_datasets_before_train.py "$@"
