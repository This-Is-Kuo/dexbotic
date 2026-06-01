#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${DEXBOTIC_PROJECT_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_ROOT}"

if ! command -v torchrun >/dev/null 2>&1 && [[ -f /opt/conda/etc/profile.d/conda.sh ]]; then
  source /opt/conda/etc/profile.d/conda.sh
  conda activate "${DEXBOTIC_CONDA_ENV:-dexbotic}"
fi

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,2,3}"
export DEXBOTIC_NPROC_PER_NODE="${DEXBOTIC_NPROC_PER_NODE:-$(( $(tr -cd ',' <<< "${CUDA_VISIBLE_DEVICES}" | wc -c) + 1 ))}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
DEXBOTIC_BASE_MODEL_USER_SET="${DEXBOTIC_BASE_MODEL+x}"
DEXBOTIC_NUM_TRAIN_STEPS_USER_SET="${DEXBOTIC_NUM_TRAIN_STEPS+x}"

latest_checkpoint() {
  find "${DEXBOTIC_OUTPUT_DIR}" -maxdepth 1 -type d -name 'checkpoint-*' -printf '%T@ %p\n' 2>/dev/null \
    | sort -nr \
    | head -n 1 \
    | cut -d' ' -f2-
}

checkpoint_dp_world_size() {
  local checkpoint="$1"
  find "${checkpoint}" -path '*/global_step*/*_optim_states.pt' -type f 2>/dev/null | wc -l
}

checkpoint_total_step() {
  local checkpoint="$1"
  python3 -c '
import json
import os
import re
import sys

checkpoint = sys.argv[1]
state_path = os.path.join(checkpoint, "trainer_state.json")
fallback = 0
match = re.search(r"checkpoint-(\d+)$", checkpoint)
if match:
    fallback = int(match.group(1))

try:
    with open(state_path, "r") as f:
        state = json.load(f)
except Exception:
    print(fallback)
    raise SystemExit

steps = [
    int(entry["step"])
    for entry in state.get("log_history", [])
    if isinstance(entry, dict) and isinstance(entry.get("step"), int)
]
print(max(steps) if steps else int(state.get("global_step") or fallback))
' "${checkpoint}"
}

export DEXBOTIC_DATASET_NAME="${DEXBOTIC_DATASET_NAME:-post_origin_data_0423_train}"
export DEXBOTIC_OUTPUT_DIR="${DEXBOTIC_OUTPUT_DIR:-${PROJECT_ROOT}/user_checkpoints/dexbotic/custom_dm0/post_origin_data_0423_deltafix_15k}"
export DEXBOTIC_TARGET_TRAIN_STEPS="${DEXBOTIC_TARGET_TRAIN_STEPS:-15000}"
export DEXBOTIC_WARM_RESTART_STEP="${DEXBOTIC_WARM_RESTART_STEP:-}"
if [[ -n "${DEXBOTIC_WARM_RESTART_STEP}" ]]; then
  export DEXBOTIC_BASE_MODEL="${DEXBOTIC_BASE_MODEL:-${DEXBOTIC_OUTPUT_DIR}/checkpoint-${DEXBOTIC_WARM_RESTART_STEP}}"
  export DEXBOTIC_RESUME_FROM_CHECKPOINT="${DEXBOTIC_RESUME_FROM_CHECKPOINT:-0}"
  export DEXBOTIC_LOG_STEP_OFFSET="${DEXBOTIC_LOG_STEP_OFFSET:-${DEXBOTIC_WARM_RESTART_STEP}}"
  export DEXBOTIC_NUM_TRAIN_STEPS="${DEXBOTIC_NUM_TRAIN_STEPS:-$((DEXBOTIC_TARGET_TRAIN_STEPS - DEXBOTIC_WARM_RESTART_STEP))}"
else
  export DEXBOTIC_BASE_MODEL="${DEXBOTIC_BASE_MODEL:-${PROJECT_ROOT}/checkpoints/DM0-base}"
  export DEXBOTIC_LOG_STEP_OFFSET="${DEXBOTIC_LOG_STEP_OFFSET:-0}"
  export DEXBOTIC_NUM_TRAIN_STEPS="${DEXBOTIC_NUM_TRAIN_STEPS:-${DEXBOTIC_TARGET_TRAIN_STEPS}}"
fi

if [[ -z "${DEXBOTIC_RESUME_FROM_CHECKPOINT:-}" ]]; then
  DEXBOTIC_LATEST_CHECKPOINT="$(latest_checkpoint)"
  if [[ -n "${DEXBOTIC_LATEST_CHECKPOINT}" ]]; then
      DEXBOTIC_CHECKPOINT_WORLD_SIZE="$(checkpoint_dp_world_size "${DEXBOTIC_LATEST_CHECKPOINT}")"
    if [[ "${DEXBOTIC_CHECKPOINT_WORLD_SIZE}" != "0" && "${DEXBOTIC_CHECKPOINT_WORLD_SIZE}" != "${DEXBOTIC_NPROC_PER_NODE}" ]]; then
      DEXBOTIC_WARM_RESTART_TOTAL_STEP="$(checkpoint_total_step "${DEXBOTIC_LATEST_CHECKPOINT}")"
      if [[ -z "${DEXBOTIC_BASE_MODEL_USER_SET}" ]]; then
        export DEXBOTIC_BASE_MODEL="${DEXBOTIC_LATEST_CHECKPOINT}"
      fi
      export DEXBOTIC_RESUME_FROM_CHECKPOINT=0
      if [[ -z "${DEXBOTIC_LOG_STEP_OFFSET:-}" || "${DEXBOTIC_LOG_STEP_OFFSET}" == "0" ]]; then
        export DEXBOTIC_LOG_STEP_OFFSET="${DEXBOTIC_WARM_RESTART_TOTAL_STEP}"
      fi
      if [[ -z "${DEXBOTIC_NUM_TRAIN_STEPS_USER_SET}" ]]; then
        export DEXBOTIC_NUM_TRAIN_STEPS="$((DEXBOTIC_TARGET_TRAIN_STEPS - DEXBOTIC_WARM_RESTART_TOTAL_STEP))"
      fi
      echo "Warm-restarting from ${DEXBOTIC_BASE_MODEL} because checkpoint world size ${DEXBOTIC_CHECKPOINT_WORLD_SIZE} != current ${DEXBOTIC_NPROC_PER_NODE}"
    else
      export DEXBOTIC_RESUME_FROM_CHECKPOINT=1
    fi
  else
    export DEXBOTIC_RESUME_FROM_CHECKPOINT=0
  fi
fi
if [[ "${DEXBOTIC_RESUME_FROM_CHECKPOINT}" == "1" ]]; then
  export DEXBOTIC_RESUME_CHECKPOINT="${DEXBOTIC_RESUME_CHECKPOINT:-latest}"
fi

export DEXBOTIC_DEEPSPEED_CONFIG="${DEXBOTIC_DEEPSPEED_CONFIG:-./script/deepspeed/zero3.json}"
export DEXBOTIC_TRAIN_BATCH_SIZE="${DEXBOTIC_TRAIN_BATCH_SIZE:-2}"
export DEXBOTIC_GRAD_ACCUM="${DEXBOTIC_GRAD_ACCUM:-8}"
export DEXBOTIC_TRAIN_NUM_WORKERS="${DEXBOTIC_TRAIN_NUM_WORKERS:-4}"
export DEXBOTIC_SAVE_STEPS="${DEXBOTIC_SAVE_STEPS:-500}"
export DEXBOTIC_LOGGING_STEPS="${DEXBOTIC_LOGGING_STEPS:-10}"
export DEXBOTIC_BASE_LR="${DEXBOTIC_BASE_LR:-2e-6}"
export DEXBOTIC_MIN_LR="${DEXBOTIC_MIN_LR:-5e-7}"
export DEXBOTIC_WARMUP_STEPS="${DEXBOTIC_WARMUP_STEPS:-200}"

# Continue logging to the original W&B run unless explicitly overridden.
export WANDB_RUN_ID="${WANDB_RUN_ID:-2jt50p4n}"
export WANDB_RESUME="${WANDB_RESUME:-allow}"

mkdir -p "${PROJECT_ROOT}"

torchrun --standalone --nproc_per_node="${DEXBOTIC_NPROC_PER_NODE}" playground/post_data_01_dm0_deltafix.py \
  2>&1 | tee -a "${PROJECT_ROOT}/train_post_origin_data_0423.log"
