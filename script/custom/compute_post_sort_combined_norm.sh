#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/post_sort_common.sh"
PROJECT_ROOT="$(post_sort_project_root)"
cd "${PROJECT_ROOT}"

export DEXBOTIC_DATASET_NAME="${DEXBOTIC_DATASET_NAME:-${POST_SORT_MIX_TRAIN}}"
export DEXBOTIC_BASE_MODEL="${DEXBOTIC_BASE_MODEL:-/dexbotic/checkpoints/DM0-base}"
export DEXBOTIC_NORM_NUM_WORKERS="${DEXBOTIC_NORM_NUM_WORKERS:-4}"
export DEXBOTIC_NORM_BATCH_SIZE="${DEXBOTIC_NORM_BATCH_SIZE:-32}"

norm_hash="$(python3 -c 'import hashlib, os; print(hashlib.md5(os.environ["DEXBOTIC_DATASET_NAME"].encode()).hexdigest()[:8])')"
cache_norm_dir="${PROJECT_ROOT}/dexbotic/norm_assets/${norm_hash}"
cache_norm_file="${cache_norm_dir}/norm_stats.json"
backup_norm_file="$(post_sort_combined_norm_stats_file)"

echo "[INFO] Stage 0: compute combined norm_stats"
post_sort_print_docker_state
echo "[INFO] PROJECT_ROOT=${PROJECT_ROOT}"
echo "[INFO] DEXBOTIC_DATASET_NAME=${DEXBOTIC_DATASET_NAME}"
echo "[INFO] old dataset path=$(post_sort_dataset_dir "${POST_SORT_OLD_TRAIN}")"
echo "[INFO] new dataset path=$(post_sort_dataset_dir "${POST_SORT_NEW_TRAIN}")"
echo "[INFO] DEXBOTIC_BASE_MODEL=${DEXBOTIC_BASE_MODEL}"
echo "[INFO] expected cache norm_stats=${cache_norm_file}"
echo "[INFO] backup norm_stats=${backup_norm_file}"
echo "[INFO] data disk output root=${POST_SORT_DATA_ROOT}"

post_sort_require_writable_dm0_root
post_sort_raise_nofile_limit
post_sort_require_dataset "${POST_SORT_OLD_TRAIN}"
post_sort_require_dataset "${POST_SORT_NEW_TRAIN}"

if [[ -f "${cache_norm_file}" ]]; then
  echo "[INFO] Existing cache norm_stats found; training code may reuse it unless refreshed manually:"
  echo "       ${cache_norm_file}"
fi
if [[ -f "${backup_norm_file}" ]]; then
  echo "[INFO] Existing backup norm_stats found; it will be overwritten only after compute succeeds:"
  echo "       ${backup_norm_file}"
fi

python3 playground/post_data_01_dm0_deltafix.py --task compute_norm_stats

if [[ ! -f "${cache_norm_file}" ]]; then
  echo "[ERROR] Expected norm_stats was not created: ${cache_norm_file}"
  exit 1
fi

mkdir -p "${POST_SORT_NORM_BACKUP_DIR}"
cp "${cache_norm_file}" "${backup_norm_file}"
echo "[OK] Backed up combined norm_stats to ${backup_norm_file}"
python3 -m json.tool "${backup_norm_file}" >/dev/null
