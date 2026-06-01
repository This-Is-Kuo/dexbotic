#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/post_sort_common.sh"
PROJECT_ROOT="$(post_sort_project_root)"
cd "${PROJECT_ROOT}"

stage_dir="${STAGE2_OUTPUT_DIR:-${POST_SORT_FINETUNE_ROOT}/post_sort_stage2_new_slow_fullft}"
checkpoint="${STAGE2_CKPT:-$(post_sort_latest_checkpoint "${stage_dir}")}"
if [[ -z "${checkpoint}" || ! -d "${checkpoint}" ]]; then
  echo "[ERROR] Stage 2 checkpoint not found. Set STAGE2_CKPT or check ${stage_dir}."
  exit 1
fi

export DEXBOTIC_NORM_STATS_PATH="${DEXBOTIC_NORM_STATS_PATH:-$(post_sort_combined_norm_stats_file)}"
export DEXBOTIC_EVAL_DEVICE="${DEXBOTIC_EVAL_DEVICE:-cuda}"
export DEXBOTIC_EVAL_DEVICE_MAP="${DEXBOTIC_EVAL_DEVICE_MAP:-single}"
export DEXBOTIC_EVAL_SPLIT="${DEXBOTIC_EVAL_SPLIT:-both}"
export DEXBOTIC_OPENLOOP_STRIDE="${DEXBOTIC_OPENLOOP_STRIDE:-${DEXBOTIC_EVAL_INFERENCE_STRIDE:-1}}"
export DEXBOTIC_OPENLOOP_MAX_SAMPLES="${DEXBOTIC_OPENLOOP_MAX_SAMPLES:-0}"
export DEXBOTIC_OPENLOOP_MAX_EPISODES="${DEXBOTIC_OPENLOOP_MAX_EPISODES:-0}"
export DEXBOTIC_OPENLOOP_BATCH_SIZE="${DEXBOTIC_OPENLOOP_BATCH_SIZE:-${DEXBOTIC_EVAL_BATCH_SIZE:-8}}"
echo "[INFO] Eval checkpoint=${checkpoint}"
echo "[INFO] Eval using DEXBOTIC_NORM_STATS_PATH=${DEXBOTIC_NORM_STATS_PATH}"
echo "[INFO] Eval device=${DEXBOTIC_EVAL_DEVICE}; device_map=${DEXBOTIC_EVAL_DEVICE_MAP}; split=${DEXBOTIC_EVAL_SPLIT}"
echo "[INFO] Open-loop stride=${DEXBOTIC_OPENLOOP_STRIDE}; max_samples=${DEXBOTIC_OPENLOOP_MAX_SAMPLES}; max_episodes=${DEXBOTIC_OPENLOOP_MAX_EPISODES}; batch_size=${DEXBOTIC_OPENLOOP_BATCH_SIZE}"
if [[ "${DEXBOTIC_EVAL_DEVICE_MAP}" == "auto" ]]; then
  echo "[WARN] DEXBOTIC_EVAL_DEVICE_MAP=auto may split DM0 across GPUs and can trigger cross-device tensor errors."
  if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
    echo "[WARN] CUDA_VISIBLE_DEVICES exposes multiple GPUs (${CUDA_VISIBLE_DEVICES}) while device_map=auto is enabled."
  fi
fi
post_sort_require_writable_dm0_root
post_sort_raise_nofile_limit
post_sort_require_norm_stats_file "${DEXBOTIC_NORM_STATS_PATH}"

eval_root="${POST_SORT_EVAL_ROOT}/post_sort_stage2_new_slow"
post_sort_require_eval_output_dir "${eval_root}"
mkdir -p "${eval_root}"

run_eval() {
  local dataset_name="$1"
  local tag="$2"
  local out_dir="${eval_root}/${tag}"
  post_sort_require_dataset "${dataset_name}"
  post_sort_require_eval_output_dir "${out_dir}"
  mkdir -p "${out_dir}/arrays"
  echo "[INFO] Evaluating ${checkpoint} on ${dataset_name}; output=${out_dir}"
  local num_batches_args=()
  if [[ -n "${DEXBOTIC_EVAL_NUM_BATCHES:-}" ]]; then
    num_batches_args=(--num-batches "${DEXBOTIC_EVAL_NUM_BATCHES}")
  fi
  python3 openloop/eval_openloop.py \
    --checkpoint "${checkpoint}" \
    --exp-file playground/post_data_01_dm0_deltafix.py \
    --dataset-name "${dataset_name}" \
    --norm-stats "${DEXBOTIC_NORM_STATS_PATH}" \
    --batch-size "${DEXBOTIC_OPENLOOP_BATCH_SIZE}" \
    "${num_batches_args[@]}" \
    --num-workers "${DEXBOTIC_EVAL_NUM_WORKERS:-4}" \
    --diffusion-steps "${DEXBOTIC_EVAL_DIFFUSION_STEPS:-10}" \
    --inference-stride "${DEXBOTIC_OPENLOOP_STRIDE}" \
    --max-samples "${DEXBOTIC_OPENLOOP_MAX_SAMPLES}" \
    --max-episodes "${DEXBOTIC_OPENLOOP_MAX_EPISODES}" \
    --chunk_merge "${DEXBOTIC_EVAL_CHUNK_MERGE:-mean}" \
    --action-dim "${DEXBOTIC_EVAL_ACTION_DIM:-14}" \
    --plot-path "${out_dir}/openloop_action_plot.png" \
    --normalized-plot-path "${out_dir}/openloop_action_plot_normalized.png" \
    --metrics-path "${out_dir}/openloop_metrics.json" \
    --array-dir "${out_dir}/arrays" \
    --save-arrays true
  python3 openloop/tools/debug_openloop_metrics.py \
    --pred "${out_dir}/arrays/pred_raw.npy" \
    --gt "${out_dir}/arrays/gt_raw.npy" \
    --save_csv "${out_dir}/per_dim_raw_metrics.csv"
  python3 openloop/tools/debug_openloop_metrics.py \
    --pred "${out_dir}/arrays/pred_norm.npy" \
    --gt "${out_dir}/arrays/gt_norm.npy" \
    --save_csv "${out_dir}/per_dim_norm_metrics.csv"
  python3 openloop/tools/lag_correlation_check.py \
    --pred "${out_dir}/arrays/pred_raw.npy" \
    --gt "${out_dir}/arrays/gt_raw.npy" \
    --max_lag "${DEXBOTIC_EVAL_MAX_LAG:-20}" \
    --save_csv "${out_dir}/lag_correlation_raw.csv"
}

case "${DEXBOTIC_EVAL_SPLIT}" in
  old_test)
    run_eval "${POST_SORT_OLD_TEST}" "old_test"
    ;;
  new_test)
    run_eval "${POST_SORT_NEW_TEST}" "new_test"
    ;;
  both)
    run_eval "${POST_SORT_OLD_TEST}" "old_test"
    run_eval "${POST_SORT_NEW_TEST}" "new_test"
    ;;
  *)
    echo "[ERROR] Unsupported DEXBOTIC_EVAL_SPLIT=${DEXBOTIC_EVAL_SPLIT}; expected old_test, new_test, or both."
    exit 1
    ;;
esac

echo "[OK] Stage 2 openloop eval complete: ${eval_root}"
