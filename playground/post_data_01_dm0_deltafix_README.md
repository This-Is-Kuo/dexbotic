# post_data_01 DM0 Deltafix 实验说明

本文件对应实验入口：

- [post_data_01_dm0_deltafix.py](/home/guoyaokun/dexbotic/playground/post_data_01_dm0_deltafix.py:1)

它是当前 `post_data_01` 任务上推荐使用的 DM0 benchmark 入口，主要包含两项关键修正：

1. 不再对已经是 delta 的连续动作再次执行 `DeltaAction`
2. 支持通过环境变量切换数据集、输出目录、DeepSpeed 配置、warmup、batch size 等训练参数

如果你只想记一条主线，可以按下面顺序走：

1. 数据转换：用 [data_tools](/home/guoyaokun/dexbotic/data_tools:1)
2. 训练入口：用 [post_data_01_dm0_deltafix.py](/home/guoyaokun/dexbotic/playground/post_data_01_dm0_deltafix.py:1)
3. open-loop 评估：用 [openloop](/home/guoyaokun/dexbotic/openloop:1)

## 1. 支持的数据集

默认数据集名由环境变量控制：

- `DEXBOTIC_DATASET_NAME=post_data_01_default`
- `DEXBOTIC_DATASET_NAME=post_data_01_stateful_default`

推荐优先使用：

- `post_data_01_stateful_default`

因为它保留了 `left/right_delta_tcp` 进入 state。

## 2. 常用环境变量

这个 benchmark 目前支持以下常用环境变量：

- `DEXBOTIC_DATASET_NAME`
- `DEXBOTIC_BASE_MODEL`
- `DEXBOTIC_OUTPUT_DIR`
- `DEXBOTIC_WARMUP_STEPS`
- `DEXBOTIC_DEEPSPEED_CONFIG`
- `DEXBOTIC_NUM_TRAIN_STEPS`
- `DEXBOTIC_SAVE_STEPS`
- `DEXBOTIC_TRAIN_BATCH_SIZE`
- `DEXBOTIC_GRAD_ACCUM`
- `DEXBOTIC_TRAIN_NUM_WORKERS`
- `DEXBOTIC_WANDB_PROJECT`

推理相关：

- `DM0InferenceConfig.device_map`
- `DM0InferenceConfig.cuda_device`

通常不需要直接改代码，优先用环境变量覆盖。

## 3. 最短训练命令

### 3.1 使用 stateful 数据集训练

```bash
export DEXBOTIC_DATASET_NAME=post_data_01_stateful_default
export DEXBOTIC_OUTPUT_DIR=/dexbotic/user_checkpoints/dexbotic/custom_dm0/post_data_01_stateful_deltafix
export DEXBOTIC_WARMUP_STEPS=100
export DEXBOTIC_DEEPSPEED_CONFIG=./script/deepspeed/zero3_offload.json
export DEXBOTIC_TRAIN_BATCH_SIZE=1
export DEXBOTIC_GRAD_ACCUM=4
export DEXBOTIC_TRAIN_NUM_WORKERS=0
export DEXBOTIC_NORM_NUM_WORKERS=4
export DEXBOTIC_NORM_BATCH_SIZE=32
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=4 playground/post_data_01_dm0_deltafix.py
```

### 3.2 关闭 wandb

```bash
export WANDB_DISABLED=true
export DEXBOTIC_WANDB_PROJECT=none
```

## 4. norm stats 单独计算

```bash
export DEXBOTIC_DATASET_NAME=post_data_01_stateful_default
python playground/post_data_01_dm0_deltafix.py --task compute_norm_stats
```

## 5. open-loop 评估

推荐默认使用：

```bash
python openloop/eval_openloop.py \
  --checkpoint /dexbotic/user_checkpoints/dexbotic/custom_dm0/post_data_01_stateful_deltafix/checkpoint-1000 \
  --exp-file playground/post_data_01_dm0_deltafix.py \
  --dataset-name post_data_01_stateful_default \
  --episode-index 0 \
  --chunk_merge mean \
  --single-gpu-id 0 \
  --save-arrays true \
  --array-dir /dexbotic/openloop/artifacts/stateful_compare/checkpoint-1000
```

夹爪专项分析可改为：

```bash
--chunk_merge exp
```

## 6. 最常用调试脚本

per-dim 指标：

```bash
python openloop/tools/debug_openloop_metrics.py \
  --pred /dexbotic/openloop/artifacts/stateful_compare/checkpoint-1000/pred_norm.npy \
  --gt /dexbotic/openloop/artifacts/stateful_compare/checkpoint-1000/gt_norm.npy \
  --save_csv /dexbotic/openloop/artifacts/stateful_compare/checkpoint-1000/per_dim_metrics_norm.csv
```

lag 检查：

```bash
python openloop/tools/lag_correlation_check.py \
  --pred /dexbotic/openloop/artifacts/stateful_compare/checkpoint-1000/pred_norm.npy \
  --gt /dexbotic/openloop/artifacts/stateful_compare/checkpoint-1000/gt_norm.npy \
  --max_lag 20 \
  --save_csv /dexbotic/openloop/artifacts/stateful_compare/checkpoint-1000/lag_metrics.csv
```

obs-action 对齐检查：

```bash
python openloop/tools/check_post_data_action_alignment.py \
  --episode-jsonl /dexbotic/data/post_data_01_dexdata_stateful/jsonl/episode_000000.jsonl
```

## 7. 当前推荐基线

当前这条实验线的推荐基线是：

- 数据集：`post_data_01_stateful_default`
- benchmark：[post_data_01_dm0_deltafix.py](/home/guoyaokun/dexbotic/playground/post_data_01_dm0_deltafix.py:1)
- checkpoint：
  [checkpoint-1000](/home/guoyaokun/dexbotic/user_checkpoints/dexbotic/custom_dm0/post_data_01_stateful_deltafix/checkpoint-1000)

对应 open-loop 结果可参考：

- [stateful_compare](/home/guoyaokun/dexbotic/openloop/artifacts/stateful_compare:1)
- [DM0_Openloop_Debug_Report_post_data_01.md](/home/guoyaokun/dexbotic/docs/DM0_Openloop_Debug_Report_post_data_01.md:1)

## 8. 注意事项

1. `action_10` 当前仍是最弱维度，不建议只靠继续训练硬压。
2. 当前 DM0 实现里，连续 `state` 张量没有真正进入模型主干，这会影响部分内部控制量维度。
3. 旧的根目录 `debug_openloop_*` 和 `tmp_*` 目录是历史产物副本，新的产物统一放在：
   [openloop/artifacts](/home/guoyaokun/dexbotic/openloop/artifacts:1)
