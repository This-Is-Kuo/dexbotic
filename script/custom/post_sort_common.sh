#!/usr/bin/env bash

POST_SORT_DATA_ROOT="${POST_SORT_DATA_ROOT:-/mnt/datadisk/guoyaokun/checkpoints/DM0}"
POST_SORT_FINETUNE_ROOT="${POST_SORT_FINETUNE_ROOT:-${POST_SORT_DATA_ROOT}/finetune}"
POST_SORT_EVAL_ROOT="${POST_SORT_EVAL_ROOT:-${POST_SORT_DATA_ROOT}/eval}"
POST_SORT_NORM_BACKUP_DIR="${POST_SORT_NORM_BACKUP_DIR:-${POST_SORT_DATA_ROOT}/norm_stats/post_sort_old_new_combined}"

POST_SORT_OLD_TRAIN="post_origin_data_0423_dm0_dexdata_nobframes_train"
POST_SORT_OLD_TEST="post_origin_data_0423_dm0_dexdata_nobframes_test"
POST_SORT_NEW_TRAIN="post_data_merged_dm0_dexdata_nobframes_train"
POST_SORT_NEW_TEST="post_data_merged_dm0_dexdata_nobframes_test"
POST_SORT_MIX_TRAIN="${POST_SORT_OLD_TRAIN}+${POST_SORT_NEW_TRAIN}"

post_sort_project_root() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  cd "${DEXBOTIC_PROJECT_DIR:-$(cd "${script_dir}/../.." && pwd)}" >/dev/null 2>&1
  pwd
}

post_sort_in_docker() {
  [[ -f /.dockerenv ]] && return 0
  grep -qaE '/docker/|/kubepods/|/containerd/' /proc/1/cgroup 2>/dev/null
}

post_sort_print_docker_state() {
  if post_sort_in_docker; then
    echo "[INFO] Running inside Docker/container: yes"
  else
    echo "[WARN] Running inside Docker/container: no"
  fi
}

post_sort_require_datadisk_mount() {
  if [[ ! -d /mnt/datadisk/guoyaokun ]]; then
    echo "[ERROR] Container cannot see /mnt/datadisk/guoyaokun."
    echo "[ERROR] Restart Docker with:"
    echo "  -v /mnt/datadisk/guoyaokun:/mnt/datadisk/guoyaokun"
    echo "[ERROR] If using docker-compose, check:"
    echo "  volumes:"
    echo "    - /mnt/datadisk/guoyaokun:/mnt/datadisk/guoyaokun"
    return 1
  fi
}

post_sort_require_writable_dm0_root() {
  post_sort_require_datadisk_mount
  mkdir -p "${POST_SORT_DATA_ROOT}" || {
    echo "[ERROR] Cannot create ${POST_SORT_DATA_ROOT}."
    post_sort_print_chown_hint
    return 1
  }
  local test_dir="${POST_SORT_DATA_ROOT}/test_write"
  mkdir -p "${test_dir}" || {
    echo "[ERROR] Cannot create ${test_dir}."
    post_sort_print_chown_hint
    return 1
  }
  touch "${test_dir}/ok.txt" || {
    echo "[ERROR] Cannot write to ${test_dir}."
    post_sort_print_chown_hint
    return 1
  }
  rm -rf "${test_dir}"
}

post_sort_print_chown_hint() {
  echo "[HINT] Run this on the host if permissions are wrong:"
  echo '  sudo mkdir -p /mnt/datadisk/guoyaokun/checkpoints/DM0'
  echo '  sudo chown -R $(id -u):$(id -g) /mnt/datadisk/guoyaokun/checkpoints/DM0'
}

post_sort_print_disk_space() {
  df -h / || true
  df -h /mnt/datadisk || true
  df -h "${POST_SORT_DATA_ROOT}" || true
}

post_sort_raise_nofile_limit() {
  local target="${DEXBOTIC_ULIMIT_NOFILE:-65535}"
  local before
  before="$(ulimit -Sn)"
  if ulimit -n "${target}" 2>/dev/null; then
    echo "[INFO] nofile soft limit: ${before} -> $(ulimit -Sn)"
  else
    echo "[WARN] Could not raise nofile soft limit from ${before} to ${target}; hard limit is $(ulimit -Hn)"
  fi
}

post_sort_dataset_dir() {
  local name="$1"
  case "${name}" in
    post_origin_data_0423_dm0_dexdata_nobframes_train) echo "data/post_origin_data_0423_dm0_dexdata_nobframes_train" ;;
    post_origin_data_0423_dm0_dexdata_nobframes_test) echo "data/post_origin_data_0423_dm0_dexdata_nobframes_test" ;;
    post_data_merged_dm0_dexdata_nobframes_train) echo "data/post_data_merged_dm0_dexdata_nobframes_train" ;;
    post_data_merged_dm0_dexdata_nobframes_test) echo "data/post_data_merged_dm0_dexdata_nobframes_test" ;;
    *) echo "" ;;
  esac
}

post_sort_require_dataset() {
  local dataset_name="$1"
  local dataset_dir
  dataset_dir="$(post_sort_dataset_dir "${dataset_name}")"
  if [[ -z "${dataset_dir}" || ! -d "${dataset_dir}/jsonl" || ! -d "${dataset_dir}/video" ]]; then
    echo "[ERROR] Dataset ${dataset_name} is not present or incomplete under ${dataset_dir}."
    return 1
  fi
}

post_sort_latest_checkpoint() {
  local output_dir="$1"
  find "${output_dir}" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%f %p\n' 2>/dev/null \
    | sed -E 's/^checkpoint-([0-9]+) /\1 /' \
    | sort -n \
    | tail -n 1 \
    | cut -d' ' -f2-
}

post_sort_combined_norm_stats_file() {
  echo "${POST_SORT_NORM_BACKUP_DIR}/norm_stats.json"
}

post_sort_require_combined_norm_stats() {
  local norm_file
  norm_file="$(post_sort_combined_norm_stats_file)"
  if [[ ! -f "${norm_file}" ]]; then
    echo "[ERROR] Combined norm_stats not found: ${norm_file}"
    echo "[HINT] Run: bash script/custom/compute_post_sort_combined_norm.sh"
    return 1
  fi
}

post_sort_require_norm_stats_file() {
  local norm_file="$1"
  if [[ ! -f "${norm_file}" ]]; then
    echo "[ERROR] norm_stats not found: ${norm_file}"
    return 1
  fi
}

post_sort_require_eval_output_dir() {
  local output_dir="$1"
  case "${output_dir}" in
    "${POST_SORT_EVAL_ROOT}"/*) ;;
    *)
      echo "[ERROR] Eval output dir must be under ${POST_SORT_EVAL_ROOT}: ${output_dir}"
      return 1
      ;;
  esac
}

post_sort_num_gpus() {
  if [[ -n "${DEXBOTIC_NPROC_PER_NODE:-}" ]]; then
    echo "${DEXBOTIC_NPROC_PER_NODE}"
    return
  fi
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    echo "$(( $(tr -cd ',' <<< "${CUDA_VISIBLE_DEVICES}" | wc -c) + 1 ))"
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    local torch_gpu_count
    torch_gpu_count="$(python3 -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null || true)"
    if [[ "${torch_gpu_count}" =~ ^[0-9]+$ && "${torch_gpu_count}" -gt 0 ]]; then
      echo "${torch_gpu_count}"
      return
    fi
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    local smi_gpu_count
    smi_gpu_count="$(nvidia-smi -L 2>/dev/null | wc -l)"
    if [[ "${smi_gpu_count}" -gt 0 ]]; then
      echo "${smi_gpu_count}"
      return
    fi
  fi
  echo "1"
}
