#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/post_sort_common.sh"
PROJECT_ROOT="$(post_sort_project_root)"
cd "${PROJECT_ROOT}"

if ! command -v torchrun >/dev/null 2>&1 && [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
  source /opt/conda/etc/profile.d/conda.sh
  conda activate "${DEXBOTIC_CONDA_ENV:-dexbotic}"
fi

stage1_dir="${STAGE1_OUTPUT_DIR:-${POST_SORT_FINETUNE_ROOT}/post_sort_stage1_mix_fullft}"
if [[ -z "${STAGE1_CKPT:-}" ]]; then
  STAGE1_CKPT="$(post_sort_latest_checkpoint "${stage1_dir}")"
  if [[ -z "${STAGE1_CKPT}" ]]; then
    echo "[ERROR] STAGE1_CKPT not set and no checkpoint-* found under ${stage1_dir}"
    exit 1
  fi
fi

export DEXBOTIC_DATASET_NAME="${DEXBOTIC_DATASET_NAME:-${POST_SORT_NEW_TRAIN}}"
export DEXBOTIC_BASE_MODEL="${DEXBOTIC_BASE_MODEL:-${STAGE1_CKPT}}"
export DEXBOTIC_OUTPUT_DIR="${DEXBOTIC_OUTPUT_DIR:-${POST_SORT_FINETUNE_ROOT}/post_sort_stage2_new_slow_fullft}"
export DEXBOTIC_NORM_STATS_PATH="${DEXBOTIC_NORM_STATS_PATH:-$(post_sort_combined_norm_stats_file)}"
export DEXBOTIC_TARGET_TRAIN_STEPS="${DEXBOTIC_MAX_STEPS:-${DEXBOTIC_TARGET_TRAIN_STEPS:-10000}}"
export DEXBOTIC_NUM_TRAIN_STEPS="${DEXBOTIC_NUM_TRAIN_STEPS:-${DEXBOTIC_TARGET_TRAIN_STEPS}}"
export DEXBOTIC_BASE_LR="${DEXBOTIC_BASE_LR:-5e-6}"
export DEXBOTIC_MIN_LR="${DEXBOTIC_MIN_LR:-5e-7}"
export DEXBOTIC_WARMUP_STEPS="${DEXBOTIC_WARMUP_STEPS:-500}"
export DEXBOTIC_DEEPSPEED_CONFIG="${DEXBOTIC_DEEPSPEED_CONFIG:-./script/deepspeed/zero3.json}"
export DEXBOTIC_TRAIN_NUM_WORKERS="${DEXBOTIC_TRAIN_NUM_WORKERS:-8}"
export DEXBOTIC_TRAIN_BATCH_SIZE="${DEXBOTIC_TRAIN_BATCH_SIZE:-2}"
export DEXBOTIC_GRAD_ACCUM="${DEXBOTIC_GRAD_ACCUM:-8}"
export DEXBOTIC_SAVE_STEPS="${DEXBOTIC_SAVE_STEPS:-500}"
export DEXBOTIC_LOGGING_STEPS="${DEXBOTIC_LOGGING_STEPS:-10}"
export DEXBOTIC_WANDB_PROJECT="${DEXBOTIC_WANDB_PROJECT:-dm0-post-sort}"
export DEXBOTIC_WANDB_RUN_NAME="${DEXBOTIC_WANDB_RUN_NAME:-stage2_new_slow_fullft_lr5e-6_from_stage1}"
export WANDB_NAME="${WANDB_NAME:-${DEXBOTIC_WANDB_RUN_NAME}}"
export WANDB_RESUME="${WANDB_RESUME:-never}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export DEXBOTIC_NPROC_PER_NODE="${DEXBOTIC_NPROC_PER_NODE:-$(post_sort_num_gpus)}"
export DEXBOTIC_RESUME_FROM_CHECKPOINT="${DEXBOTIC_RESUME_FROM_CHECKPOINT:-0}"
if [[ "${DEXBOTIC_RESUME_FROM_CHECKPOINT}" == "1" ]]; then
  export DEXBOTIC_RESUME_CHECKPOINT="${DEXBOTIC_RESUME_CHECKPOINT:-latest}"
fi

echo "[INFO] Stage 2: new slow data full fine-tune from Stage 1"
post_sort_print_docker_state
post_sort_require_writable_dm0_root
post_sort_raise_nofile_limit
post_sort_require_dataset "${POST_SORT_NEW_TRAIN}"
post_sort_require_combined_norm_stats

if [[ ! -d "${STAGE1_CKPT}" ]]; then
  echo "[ERROR] Stage 1 checkpoint does not exist: ${STAGE1_CKPT}"
  exit 1
fi
if [[ "${DEXBOTIC_OUTPUT_DIR}" == /dexbotic/user_checkpoints/* || "${DEXBOTIC_OUTPUT_DIR}" == user_checkpoints/* || "${DEXBOTIC_OUTPUT_DIR}" == .*/user_checkpoints/* ]]; then
  echo "[ERROR] Refusing to write checkpoints to system-disk default path: ${DEXBOTIC_OUTPUT_DIR}"
  exit 1
fi

mkdir -p "${DEXBOTIC_OUTPUT_DIR}"
touch "${DEXBOTIC_OUTPUT_DIR}/write_check.ok"
rm -f "${DEXBOTIC_OUTPUT_DIR}/write_check.ok"

echo "[INFO] Stage 2 init checkpoint / warm-start checkpoint = ${STAGE1_CKPT}"
echo "[INFO] This is warm-start from Stage 1 weights, not optimizer/scheduler resume."
echo "[INFO] If DEXBOTIC_RESUME_FROM_CHECKPOINT=1, resume is resolved only from Stage 2 output dir: ${DEXBOTIC_OUTPUT_DIR}"
echo "[INFO] DEXBOTIC_BASE_MODEL=${DEXBOTIC_BASE_MODEL}"
echo "[INFO] DEXBOTIC_OUTPUT_DIR=${DEXBOTIC_OUTPUT_DIR}"
echo "[INFO] DEXBOTIC_DATASET_NAME=${DEXBOTIC_DATASET_NAME}"
echo "[INFO] DEXBOTIC_NORM_STATS_PATH=${DEXBOTIC_NORM_STATS_PATH}"
echo "[INFO] norm_stats source=external combined Stage 0 stats"
echo "[INFO] Stage 2 explicitly reuses Stage 0 combined norm_stats and disables new-only auto norm by setting DEXBOTIC_NORM_STATS_PATH."
echo "[INFO] DEXBOTIC_NUM_TRAIN_STEPS=${DEXBOTIC_NUM_TRAIN_STEPS}"
echo "[INFO] DEXBOTIC_BASE_LR=${DEXBOTIC_BASE_LR}"
echo "[INFO] DEXBOTIC_WARMUP_STEPS=${DEXBOTIC_WARMUP_STEPS}"
echo "[INFO] DEXBOTIC_DEEPSPEED_CONFIG=${DEXBOTIC_DEEPSPEED_CONFIG}"
echo "[INFO] DEXBOTIC_NPROC_PER_NODE=${DEXBOTIC_NPROC_PER_NODE}"
echo "[INFO] DEXBOTIC_RESUME_FROM_CHECKPOINT=${DEXBOTIC_RESUME_FROM_CHECKPOINT}"
if [[ "${DEXBOTIC_NPROC_PER_NODE}" -lt 2 ]]; then
  echo "[WARN] Stage 2 full fine-tune is memory-heavy. DEXBOTIC_NPROC_PER_NODE=${DEXBOTIC_NPROC_PER_NODE} may OOM on a single GPU; prefer exposing multiple GPUs or set CUDA_VISIBLE_DEVICES explicitly."
fi
df -h /
df -h /mnt/datadisk
df -h "${DEXBOTIC_OUTPUT_DIR}"

torchrun --standalone --nproc_per_node="${DEXBOTIC_NPROC_PER_NODE}" playground/post_data_01_dm0_deltafix.py \
  2>&1 | tee -a "${DEXBOTIC_OUTPUT_DIR}/train.log"
