# DM0 Post-Sort 训练与推理操作手册

本文档用于复制粘贴执行当前 DM0 `post_sort` 训练、续训和在线推理流程。

## 1. 当前目录约定

训练与推理入口：

```text
playground/post_data_01_dm0_deltafix.py
```

数据盘根目录：

```text
/mnt/datadisk/guoyaokun/checkpoints/DM0
```

输出目录：

```text
Stage 1: /mnt/datadisk/guoyaokun/checkpoints/DM0/finetune/post_sort_stage1_mix_fullft
Stage 2: /mnt/datadisk/guoyaokun/checkpoints/DM0/finetune/post_sort_stage2_new_slow_fullft
norm stats: /mnt/datadisk/guoyaokun/checkpoints/DM0/norm_stats/post_sort_old_new_combined/norm_stats.json
```

Stage 1 使用旧数据和新数据混合训练。Stage 2 默认自动选择最新 Stage 1 checkpoint
作为初始权重，仅使用新 slow 数据继续微调。

## 2. 进入 Docker

宿主机执行：

```bash
docker exec -it dexbotic-train-datadisk bash
```

容器内执行：

```bash
cd /dexbotic
```

首次使用时检查数据盘挂载和写权限：

```bash
bash script/custom/check_docker_datadisk_mount.sh
bash script/custom/check_post_sort_datasets_before_train.sh
```

## 3. Stage 0：计算 Norm Stats

首次训练或数据集变化后执行：

```bash
cd /dexbotic
bash script/custom/compute_post_sort_combined_norm.sh
```

预期输出文件：

```text
/mnt/datadisk/guoyaokun/checkpoints/DM0/norm_stats/post_sort_old_new_combined/norm_stats.json
```

## 4. Stage 1：混合数据训练

### 4.1 从基础模型开始训练

```bash
cd /dexbotic

export CUDA_VISIBLE_DEVICES=0,1,2,3
export DEXBOTIC_RESUME_FROM_CHECKPOINT=0

bash script/custom/train_post_sort_stage1_mix_full_ft.sh
```

默认配置：

```text
数据集: old train + new train
基础模型: /dexbotic/checkpoints/DM0-base
总步数: 10000
学习率: 2.5e-5
warmup: 1000
每 500 步保存 checkpoint
```

### 4.2 从 Stage 1 最新 checkpoint 续训

适用于同一训练任务、相同 GPU world size 下恢复 optimizer 和 scheduler。

```bash
cd /dexbotic

export CUDA_VISIBLE_DEVICES=0,1,2,3
export DEXBOTIC_RESUME_FROM_CHECKPOINT=1
export DEXBOTIC_RESUME_CHECKPOINT=latest

bash script/custom/train_post_sort_stage1_mix_full_ft.sh
```

执行新任务前，避免继承旧 shell 环境：

```bash
unset DEXBOTIC_BASE_MODEL
unset DEXBOTIC_OUTPUT_DIR
unset DEXBOTIC_LOG_STEP_OFFSET
unset DEXBOTIC_NUM_TRAIN_STEPS
unset DEXBOTIC_TARGET_TRAIN_STEPS
```

## 5. Stage 2：新 Slow 数据微调

### 5.1 从 Stage 1 最新 checkpoint 开始 Stage 2

这属于 warm-start：读取 Stage 1 权重，但重新初始化 optimizer 和 scheduler。

```bash
cd /dexbotic

export CUDA_VISIBLE_DEVICES=0,1,2,3
export DEXBOTIC_RESUME_FROM_CHECKPOINT=0
unset STAGE1_CKPT

bash script/custom/train_post_sort_stage2_new_slow_full_ft.sh
```

默认配置：

```text
数据集: new slow train
基础模型: 自动选择 Stage 1 最大步数 checkpoint
总步数: 10000
学习率: 5e-6
warmup: 500
每 500 步保存 checkpoint
```

### 5.2 指定 Stage 1 checkpoint 启动 Stage 2

```bash
cd /dexbotic

export CUDA_VISIBLE_DEVICES=0,1,2,3
export DEXBOTIC_RESUME_FROM_CHECKPOINT=0
export STAGE1_CKPT=/mnt/datadisk/guoyaokun/checkpoints/DM0/finetune/post_sort_stage1_mix_fullft/checkpoint-6500

bash script/custom/train_post_sort_stage2_new_slow_full_ft.sh
```

### 5.3 从 Stage 2 最新 checkpoint 续训

```bash
cd /dexbotic

export CUDA_VISIBLE_DEVICES=0,1,2,3
export DEXBOTIC_RESUME_FROM_CHECKPOINT=1
export DEXBOTIC_RESUME_CHECKPOINT=latest

bash script/custom/train_post_sort_stage2_new_slow_full_ft.sh
```

## 6. Open-Loop 评估

默认自动选择对应阶段最大步数 checkpoint，并同时评估 old test 和 new test。

Stage 1：

```bash
cd /dexbotic
export CUDA_VISIBLE_DEVICES=0
bash script/custom/eval_post_sort_stage1.sh
```

Stage 2：

```bash
cd /dexbotic
export CUDA_VISIBLE_DEVICES=0
bash script/custom/eval_post_sort_stage2.sh
```

只评估 new test：

```bash
export DEXBOTIC_EVAL_SPLIT=new_test
```

显式指定 checkpoint：

```bash
export STAGE1_CKPT=/mnt/datadisk/guoyaokun/checkpoints/DM0/finetune/post_sort_stage1_mix_fullft/checkpoint-6500
export STAGE2_CKPT=/mnt/datadisk/guoyaokun/checkpoints/DM0/finetune/post_sort_stage2_new_slow_fullft/checkpoint-2000
```

## 7. 在线推理服务

在线推理默认使用 FP32。FP32 模式会在加载后统一模型参数和浮点输入 dtype，
避免 `Float` 与 `BFloat16` 混用导致 HTTP 500。

### 7.1 使用 Stage 1 最新 checkpoint

```bash
cd /dexbotic

source script/custom/post_sort_common.sh
export CUDA_VISIBLE_DEVICES=0
export DEXBOTIC_EVAL_DEVICE=cuda
export DEXBOTIC_EVAL_DEVICE_MAP=single
export DEXBOTIC_EVAL_TORCH_DTYPE=float32
export DEXBOTIC_INFERENCE_MODEL_PATH="$(
  post_sort_latest_checkpoint \
    /mnt/datadisk/guoyaokun/checkpoints/DM0/finetune/post_sort_stage1_mix_fullft
)"

echo "checkpoint=${DEXBOTIC_INFERENCE_MODEL_PATH}"
python playground/post_data_01_dm0_deltafix.py --task inference
```

### 7.2 使用 Stage 2 最新 checkpoint

```bash
cd /dexbotic

source script/custom/post_sort_common.sh
export CUDA_VISIBLE_DEVICES=0
export DEXBOTIC_EVAL_DEVICE=cuda
export DEXBOTIC_EVAL_DEVICE_MAP=single
export DEXBOTIC_EVAL_TORCH_DTYPE=float32
export DEXBOTIC_INFERENCE_MODEL_PATH="$(
  post_sort_latest_checkpoint \
    /mnt/datadisk/guoyaokun/checkpoints/DM0/finetune/post_sort_stage2_new_slow_fullft
)"

echo "checkpoint=${DEXBOTIC_INFERENCE_MODEL_PATH}"
python playground/post_data_01_dm0_deltafix.py --task inference
```

### 7.3 显式指定 checkpoint

```bash
cd /dexbotic

export CUDA_VISIBLE_DEVICES=0
export DEXBOTIC_EVAL_DEVICE=cuda
export DEXBOTIC_EVAL_DEVICE_MAP=single
export DEXBOTIC_EVAL_TORCH_DTYPE=float32
export DEXBOTIC_INFERENCE_MODEL_PATH=/mnt/datadisk/guoyaokun/checkpoints/DM0/finetune/post_sort_stage1_mix_fullft/checkpoint-6500

python playground/post_data_01_dm0_deltafix.py --task inference
```

服务地址：

```text
POST http://127.0.0.1:7891/process_frame
```

启动日志应包含：

```text
eval torch_dtype = torch.float32
model floating parameter dtypes = ['torch.float32']
Model loaded successfully
```

## 8. UI Agent 调试

宿主机调用真实推理服务：

```bash
cd /home/guoyaokun/dexbotic/agent_debug_suite_1

python agent_debug_suite/debug_agent_from_dataset.py \
  --root <dataset_root> \
  --agent-file agent_debug_suite/dm0_agent.py \
  --agent-class DM0Agent \
  --episode 0 \
  --num-frames 5 \
  --agent-kwargs-json '{"base_url":"http://127.0.0.1:7891"}'
```

`DM0Agent` 会自动请求：

```text
http://127.0.0.1:7891/process_frame
```

在线 Agent 默认启用异步预取、soft-replace 和短暂动作融合，减少 chunk 边界停顿。

## 9. 常见问题

### 9.1 端口被占用

先停止旧推理服务：

```bash
pkill -f 'post_data_01_dm0_deltafix.py --task inference'
```

然后重新启动。

### 9.2 `mat1 and mat2 must have the same dtype`

确认使用最新自定义入口，并显式设置：

```bash
export DEXBOTIC_EVAL_TORCH_DTYPE=float32
```

启动日志必须包含：

```text
model floating parameter dtypes = ['torch.float32']
```

### 9.3 宿主机无法访问 Docker 内的 `7891`

创建容器时需要映射端口：

```bash
-p 7891:7891
```

如果 Agent 在另一个容器内运行，应使用 Docker network 中可访问的容器名：

```text
http://<policy-container-name>:7891
```

### 9.4 快速查看已有 checkpoint

```bash
find /mnt/datadisk/guoyaokun/checkpoints/DM0/finetune \
  -maxdepth 2 -type d -name 'checkpoint-*' | sort
```

当前已确认：

```text
Stage 1 最新: post_sort_stage1_mix_fullft/checkpoint-6500
Stage 2 最新: post_sort_stage2_new_slow_fullft/checkpoint-2000
```
