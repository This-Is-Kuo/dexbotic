#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/post_sort_common.sh"

echo "[INFO] Checking Docker/container and data disk mount."
post_sort_print_docker_state

paths=(
  "/mnt/datadisk"
  "/mnt/datadisk/guoyaokun"
  "/mnt/datadisk/guoyaokun/checkpoints"
  "/mnt/datadisk/guoyaokun/checkpoints/DM0"
)

for path in "${paths[@]}"; do
  if [[ -e "${path}" ]]; then
    echo "[OK] ${path} exists"
  else
    echo "[MISSING] ${path}"
  fi
done

post_sort_require_datadisk_mount

echo "[INFO] Disk space before write check:"
post_sort_print_disk_space

mkdir -p "${POST_SORT_DATA_ROOT}" || {
  echo "[ERROR] Cannot create ${POST_SORT_DATA_ROOT}."
  post_sort_print_chown_hint
  exit 1
}

test_dir="${POST_SORT_DATA_ROOT}/test_write"
if mkdir -p "${test_dir}" && touch "${test_dir}/ok.txt"; then
  echo "[OK] Data disk write check passed: ${test_dir}/ok.txt"
  rm -rf "${test_dir}"
else
  echo "[ERROR] Data disk path is not writable: ${POST_SORT_DATA_ROOT}"
  post_sort_print_chown_hint
  exit 1
fi

echo "[INFO] Disk space after write check:"
post_sort_print_disk_space
