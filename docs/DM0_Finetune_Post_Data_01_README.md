# DM0 微调流程指南：post_data_01

这份 README 面向 `Dexmal/dexbotic` 中的自定义 LeRobot 风格数据 `post_data_01`，覆盖从 Docker 环境启动、数据转换、数据注册、训练前校验、norm stats 计算到 DM0 微调训练的完整流程。

当前推荐训练入口是：

- [playground/post_data_01_dm0_deltafix.py](/home/guoyaokun/dexbotic/playground/post_data_01_dm0_deltafix.py:1)

不要直接用默认 [dexbotic/exp/dm0_exp.py](/home/guoyaokun/dexbotic/dexbotic/exp/dm0_exp.py:1) 训练 `post_data_01`，因为当前 jsonl 已经包含 14D delta action，默认 DM0 pipeline 会重新 `AddAction + DeltaAction`，导致动作标签语义错误。

## 1. 路径约定

本项目默认 Docker 内工作目录为 `/dexbotic`。请确保宿主机仓库挂载到容器内 `/dexbotic`：

```bash
docker run -it --rm --gpus all --network host \
  --shm-size 64g \
  -v /home/guoyaokun/dexbotic:/dexbotic \
  dexmal/dexbotic \
  bash
```

进入容器后：

```bash
cd /dexbotic
conda activate dexbotic
pip install -e .
```

如果是 Blackwell GPU，例如 B100 / RTX 5090，使用官方专用镜像：

```bash
docker run -it --rm --gpus all --network host \
  --shm-size 64g \
  -v /home/guoyaokun/dexbotic:/dexbotic \
  dexmal/dexbotic:c130t28 \
  bash
```

容器内先确认路径：

```bash
ls -ld /dexbotic
ls /dexbotic/data/post_data_01_dexdata/jsonl | head
ls -l /dexbotic/data/post_data_01_dexdata/video/observation.images.chest/chunk-000/file-000.mp4
ls /dexbotic/data/post_data_01/videos/observation.images.chest/chunk-000/file-000.mp4
```

这些命令必须通过。当前数据注册文件和视频软链接都依赖 `/dexbotic/...`。

## 2. 下载基础模型

DM0-base 默认路径建议放在：

```text
/dexbotic/checkpoints/DM0-base
```

下载命令：

```bash
cd /dexbotic
mkdir -p checkpoints
git clone https://huggingface.co/Dexmal/DM0-base checkpoints/DM0-base
```

训练入口默认会读取：

```bash
DEXBOTIC_BASE_MODEL=/dexbotic/checkpoints/DM0-base
```

## 3. 数据目录结构

原始 LeRobot 风格数据：

```text
/dexbotic/data/post_data_01
├── data/
├── meta/
└── videos/
```

转换后的 DexData：

```text
/dexbotic/data/post_data_01_dexdata
├── jsonl/
│   ├── episode_000000.jsonl
│   └── ...
└── video/
    ├── observation.images.chest/
    ├── observation.images.left/
    └── observation.images.right/
```

当前转换结果字段：

| 字段 | 含义 |
|---|---|
| `images_1` | chest/main view |
| `images_2` | left camera |
| `images_3` | right camera |
| `state` | 16D：left_tcp(7) + right_tcp(7) + left_pinch + right_pinch |
| `action` | 14D：left_delta_tcp(6) + left_pinch + right_delta_tcp(6) + right_pinch |
| `prompt` | 语言指令 |
| `extra.timestamp` | episode 内相对 timestamp |
| `extra.frame_index` | 视频帧索引 |
| `extra.progress` | chest/left/right progress |

## 4. 数据转换

### 4.1 先转换 1 个 episode 做 smoke test

```bash
cd /dexbotic

python3 data_tools/convert_post_data_01_to_dexdata.py \
  --raw_root /dexbotic/data/post_data_01 \
  --output_root /dexbotic/data/post_data_01_dexdata_test \
  --video_mode symlink \
  --max_episodes 1 \
  --overwrite
```

检查 jsonl：

```bash
head -n 1 /dexbotic/data/post_data_01_dexdata_test/jsonl/episode_000000.jsonl
find /dexbotic/data/post_data_01_dexdata_test/video -type l | head
```

### 4.2 转换全量数据

确认 smoke test 没问题后运行：

```bash
python3 data_tools/convert_post_data_01_to_dexdata.py \
  --raw_root /dexbotic/data/post_data_01 \
  --output_root /dexbotic/data/post_data_01_dexdata \
  --video_mode symlink \
  --overwrite
```

`--video_mode` 选择：

| 参数 | 作用 | 建议 |
|---|---|---|
| `symlink` | video 下创建指向原始 mp4 的软链接 | 推荐，省空间，但要求 `/dexbotic` 挂载一致 |
| `copy` | 复制 mp4 到 dexdata/video | 更稳，但占空间 |
| `skip` | 不处理视频 | 仅调试 jsonl，不可用于正式训练 |

### 4.3 可选：stateful 转换

如果希望把 `observation.state.left_delta_tcp` 和 `observation.state.right_delta_tcp` 也放进 state，可用 stateful 版本：

```bash
python3 data_tools/convert_post_data_01_to_dexdata_stateful.py \
  --raw_root /dexbotic/data/post_data_01 \
  --output_root /dexbotic/data/post_data_01_dexdata_stateful \
  --video_mode symlink \
  --overwrite
```

stateful 输出 state 为 28D：

```text
0..6   left_tcp
7..13  right_tcp
14     left_pinch
15     right_pinch
16..21 left_delta_tcp
22..27 right_delta_tcp
```

注意：当前 DM0 实现中连续 `state` 参数没有真正进入模型 prefix，仅作为 batch 字段和后处理相关输入。因此 stateful 数据是否带来收益，需要实验验证。

## 5. 数据注册

当前注册文件：

- [dexbotic/data/data_source/post_data_01.py](/home/guoyaokun/dexbotic/dexbotic/data/data_source/post_data_01.py:1)

内容核心是：

```python
POST_DATA_01_DATASET = {
    "default": {
        "data_path_prefix": "/dexbotic/data/post_data_01_dexdata/video",
        "annotations": "/dexbotic/data/post_data_01_dexdata/jsonl",
        "frequency": 1,
    },
}

meta_data = {
    "non_delta_mask": [6, 13],
    "periodic_mask": None,
    "periodic_range": None,
}

register_dataset(POST_DATA_01_DATASET, meta_data=meta_data, prefix="post_data_01")
```

训练时数据集名是：

```bash
post_data_01_default
```

如果使用 stateful 数据，需要另建注册文件，例如 `dexbotic/data/data_source/post_data_01_stateful.py`，并把路径改为：

```python
"data_path_prefix": "/dexbotic/data/post_data_01_dexdata_stateful/video"
"annotations": "/dexbotic/data/post_data_01_dexdata_stateful/jsonl"
```

对应数据集名建议为：

```bash
post_data_01_stateful_default
```

## 6. 训练前数据校验

必须先跑对齐检查：

```bash
python3 scripts/check_dm0_data_alignment.py \
  --dexdata_root /dexbotic/data/post_data_01_dexdata \
  --jsonl_dir /dexbotic/data/post_data_01_dexdata/jsonl \
  --video_dir /dexbotic/data/post_data_01_dexdata/video \
  --num_episodes 10 \
  --num_samples_per_episode 20 \
  --output_dir /dexbotic/debug_data_check/post_data_01
```

输出文件：

| 文件 | 作用 |
|---|---|
| `summary.json` | 总览统计、action/state 统计、warning/error 数 |
| `per_episode_report.csv` | 每个 episode 的 step、timestamp、frame 对齐 |
| `per_camera_video_report.csv` | 每个视频的 fps、frame count、start_time、B-frame |
| `suspicious_samples.csv` | 可疑样本明细 |
| `sample_frames/` | 抽样帧可视化，图片上叠加 step/action/state 信息 |

正式训练前必须满足：

- `num_errors = 0`
- `prompt_empty_count = 0`
- action dim 全部一致，当前应为 14
- state dim 全部一致，当前应为 16 或 stateful 的 28
- timestamp 从 0 开始并单调递增
- `frame_index` 从 0 开始并连续
- 视频 `start_time` 为 0 或已确认转换时做了 episode 内归零
- 多相机 frame count 和 start_time 一致
- 抽样帧肉眼看起来和动作/状态时序一致

全量无抽帧扫描可用：

```bash
python3 scripts/check_dm0_data_alignment.py \
  --dexdata_root /dexbotic/data/post_data_01_dexdata \
  --jsonl_dir /dexbotic/data/post_data_01_dexdata/jsonl \
  --video_dir /dexbotic/data/post_data_01_dexdata/video \
  --num_episodes 0 \
  --num_samples_per_episode 0 \
  --output_dir /dexbotic/debug_data_check/post_data_01_full_noframes
```

## 7. 训练入口和 action 语义

使用：

```bash
playground/post_data_01_dm0_deltafix.py
```

这个入口做了两件关键修正：

1. 保留 jsonl 里的 14D `action`
2. 禁用二次 delta：`DeltaAction(enable=False)`

当前 action 维度：

| 维度 | 含义 |
|---|---|
| 0..5 | left_delta_tcp |
| 6 | left_pinch |
| 7..12 | right_delta_tcp |
| 13 | right_pinch |

`non_delta_mask=[6,13]`，夹爪维度不参与 delta 化。

DM0 模型内部默认 `action_dim=32`、`chunk_size=50`。训练时会把 14D action pad 到 32D，模型预测 50 步 action chunk；推理/评估时取前 14D。

## 8. 环境变量参数表

`playground/post_data_01_dm0_deltafix.py` 支持以下常用环境变量：

| 环境变量 | 默认值 | 说明 |
|---|---|---|
| `DEXBOTIC_DATASET_NAME` | `post_data_01_default` | 注册数据集名 |
| `DEXBOTIC_BASE_MODEL` | `/dexbotic/checkpoints/DM0-base` | DM0-base 路径 |
| `DEXBOTIC_OUTPUT_DIR` | `./user_checkpoints/...` | checkpoint 输出目录 |
| `DEXBOTIC_DEEPSPEED_CONFIG` | `./script/deepspeed/zero3.json` | DeepSpeed 配置 |
| `DEXBOTIC_NUM_TRAIN_STEPS` | `1000` | 总训练 step |
| `DEXBOTIC_SAVE_STEPS` | `50` | checkpoint 保存间隔 |
| `DEXBOTIC_TRAIN_BATCH_SIZE` | `2` | 单卡 batch size |
| `DEXBOTIC_GRAD_ACCUM` | `4` | 梯度累积步数 |
| `DEXBOTIC_TRAIN_NUM_WORKERS` | `4` | dataloader workers |
| `DEXBOTIC_WARMUP_STEPS` | `1000` | warmup steps |
| `DEXBOTIC_WANDB_PROJECT` | `dm0_sft_post_data_01` | wandb project；设为 `none` 可关闭 |
| `DEXBOTIC_NORM_NUM_WORKERS` | `4` | 计算 norm stats 的 workers |
| `DEXBOTIC_NORM_BATCH_SIZE` | `32` | 计算 norm stats 的 batch size |

常用关闭 wandb：

```bash
export WANDB_DISABLED=true
export DEXBOTIC_WANDB_PROJECT=none
```

## 9. 计算 norm stats

正式训练前建议单独计算一次 norm stats：

```bash
cd /dexbotic

export DEXBOTIC_DATASET_NAME=post_data_01_default
export DEXBOTIC_BASE_MODEL=/dexbotic/checkpoints/DM0-base
export DEXBOTIC_NORM_NUM_WORKERS=4
export DEXBOTIC_NORM_BATCH_SIZE=32

python3 playground/post_data_01_dm0_deltafix.py --task compute_norm_stats
```

如果显存或 dataloader 环境不稳定，可先降 worker：

```bash
export DEXBOTIC_NORM_NUM_WORKERS=0
```

norm stats 会按 dataset name hash 缓存到默认 norm assets 路径。训练入口开启 `auto_norm` 时，如果没找到缓存，也会尝试自动计算；但为了可控，推荐训练前手动计算。

## 10. Smoke Train

正式训练前先跑一个很短的 smoke train：

```bash
cd /dexbotic

export DEXBOTIC_DATASET_NAME=post_data_01_default
export DEXBOTIC_BASE_MODEL=/dexbotic/checkpoints/DM0-base
export DEXBOTIC_OUTPUT_DIR=/dexbotic/user_checkpoints/dexbotic/custom_dm0/post_data_01_smoke
export DEXBOTIC_DEEPSPEED_CONFIG=./script/deepspeed/zero3_offload.json
export DEXBOTIC_NUM_TRAIN_STEPS=5
export DEXBOTIC_SAVE_STEPS=5
export DEXBOTIC_TRAIN_BATCH_SIZE=1
export DEXBOTIC_GRAD_ACCUM=1
export DEXBOTIC_TRAIN_NUM_WORKERS=0
export DEXBOTIC_WARMUP_STEPS=1
export WANDB_DISABLED=true
export DEXBOTIC_WANDB_PROJECT=none
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=1 playground/post_data_01_dm0_deltafix.py
```

Smoke train 通过标准：

- 能构建 dataset/index cache
- 能读取视频帧
- batch 中 `images/actions/states/image_masks` shape 正常
- loss 能正常打印
- checkpoint 能保存

## 11. 正式训练命令

### 11.1 单机 4 卡示例

```bash
cd /dexbotic

export DEXBOTIC_DATASET_NAME=post_data_01_default
export DEXBOTIC_BASE_MODEL=/dexbotic/checkpoints/DM0-base
export DEXBOTIC_OUTPUT_DIR=/dexbotic/user_checkpoints/dexbotic/custom_dm0/post_data_01_deltafix
export DEXBOTIC_DEEPSPEED_CONFIG=./script/deepspeed/zero3_offload.json
export DEXBOTIC_NUM_TRAIN_STEPS=1000
export DEXBOTIC_SAVE_STEPS=50
export DEXBOTIC_TRAIN_BATCH_SIZE=1
export DEXBOTIC_GRAD_ACCUM=4
export DEXBOTIC_TRAIN_NUM_WORKERS=4
export DEXBOTIC_WARMUP_STEPS=100
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

torchrun --nproc_per_node=4 playground/post_data_01_dm0_deltafix.py
```

### 11.2 单机 8 卡示例

```bash
export DEXBOTIC_DEEPSPEED_CONFIG=./script/deepspeed/zero3.json
export DEXBOTIC_TRAIN_BATCH_SIZE=2
export DEXBOTIC_GRAD_ACCUM=4
export DEXBOTIC_NUM_TRAIN_STEPS=30000
export DEXBOTIC_SAVE_STEPS=1000
export DEXBOTIC_WARMUP_STEPS=1000

torchrun --nproc_per_node=8 playground/post_data_01_dm0_deltafix.py
```

有效 batch size 计算：

```text
global_batch = nproc_per_node * DEXBOTIC_TRAIN_BATCH_SIZE * DEXBOTIC_GRAD_ACCUM
```

例如 4 卡、单卡 batch 1、grad accum 4：

```text
global_batch = 4 * 1 * 4 = 16
```

## 12. 推荐参数起点

| 场景 | GPU | DeepSpeed | batch | grad accum | workers | 建议 |
|---|---:|---|---:|---:|---:|---|
| smoke test | 1 | zero3_offload | 1 | 1 | 0 | 先确认链路 |
| 4x 4090 | 4 | zero3_offload | 1 | 4 | 4 | 稳妥起步 |
| 8x 4090 | 8 | zero3_offload | 1 | 4 | 4-8 | 显存紧张时保持 offload |
| 8x A100/H100 | 8 | zero3 | 2-4 | 1-4 | 8-16 | 可提高吞吐 |

如果遇到 dataloader 卡住或共享内存问题：

```bash
export DEXBOTIC_TRAIN_NUM_WORKERS=0
```

如果遇到 CUDA 内存碎片：

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## 13. 训练前 Checklist

- [ ] Docker 容器内 `/dexbotic` 挂载正确。
- [ ] `/dexbotic/checkpoints/DM0-base` 存在。
- [ ] `/dexbotic/data/post_data_01_dexdata/jsonl` 存在。
- [ ] `/dexbotic/data/post_data_01_dexdata/video` 软链接可读。
- [ ] `scripts/check_dm0_data_alignment.py` 检查 `num_errors=0`。
- [ ] 抽样帧 `sample_frames/` 肉眼检查正常。
- [ ] 使用 `playground/post_data_01_dm0_deltafix.py`，不是默认 `dm0_exp.py`。
- [ ] `DEXBOTIC_DATASET_NAME` 与注册文件一致。
- [ ] norm stats 已基于当前数据重新计算。
- [ ] smoke train 5 step 跑通。
- [ ] 正式训练输出目录是新的或你明确想续训/覆盖。

## 14. 常见问题

### 14.1 容器里找不到 jsonl

检查：

```bash
ls /dexbotic/data/post_data_01_dexdata/jsonl
```

如果不存在，通常是 Docker 启动时没有挂载：

```bash
-v /home/guoyaokun/dexbotic:/dexbotic
```

### 14.2 jsonl 存在但视频打不开

检查软链接目标：

```bash
readlink /dexbotic/data/post_data_01_dexdata/video/observation.images.chest/chunk-000/file-000.mp4
ls /dexbotic/data/post_data_01/videos/observation.images.chest/chunk-000/file-000.mp4
```

如果原始视频不存在，要么恢复 `/dexbotic/data/post_data_01/videos`，要么重新转换并使用 `--video_mode copy`。

### 14.3 loss 正常但 open-loop 偏置很大

优先检查：

- 是否误用了默认 DM0 action pipeline。
- norm stats 是否来自当前数据和当前 pipeline。
- gripper 维度 6/13 的开合方向是否和部署端一致。
- action 连续维单位是否是训练/执行一致的 delta TCP。
- timestamp-frame 对齐检查是否仍然通过。

### 14.4 要不要用 stateful 数据

如果你注册了 `post_data_01_stateful_default`，可这样训练：

```bash
export DEXBOTIC_DATASET_NAME=post_data_01_stateful_default
export DEXBOTIC_OUTPUT_DIR=/dexbotic/user_checkpoints/dexbotic/custom_dm0/post_data_01_stateful_deltafix
torchrun --nproc_per_node=4 playground/post_data_01_dm0_deltafix.py
```

但请记住：当前 DM0 forward 中连续 `state` 没有真正进入 prefix embedding，所以 stateful 的收益不是必然的，需要通过 open-loop 和实际评估确认。

## 15. 参考报告

- [DM0 数据链路训练前审计报告：post_data_01](/home/guoyaokun/dexbotic/docs/DM0_Data_Preflight_Report_post_data_01_2026-05-12.md:1)
- [DM0 Open-loop Debug Report](/home/guoyaokun/dexbotic/docs/DM0_Openloop_Debug_Report_post_data_01.md:1)
- [post_data_01 DM0 Deltafix 实验说明](/home/guoyaokun/dexbotic/playground/post_data_01_dm0_deltafix_README.md:1)
