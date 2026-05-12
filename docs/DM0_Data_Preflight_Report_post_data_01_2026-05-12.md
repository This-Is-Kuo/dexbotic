# DM0 数据链路训练前审计报告：post_data_01

日期：2026-05-12

## 1. 检查范围

本次检查覆盖 `post_data_01` 从 LeRobot 风格数据到 DexData/jsonl/video，再到 DM0 dataloader/action normalization/训练入口的主要链路。新增并运行了训练前对齐检查脚本：

```bash
python3 scripts/check_dm0_data_alignment.py \
  --dexdata_root /dexbotic/data/post_data_01_dexdata \
  --jsonl_dir /dexbotic/data/post_data_01_dexdata/jsonl \
  --video_dir /dexbotic/data/post_data_01_dexdata/video \
  --num_episodes 10 \
  --num_samples_per_episode 20 \
  --output_dir /dexbotic/debug_data_check/post_data_01
```

同时执行了全量无抽帧扫描：

```bash
python3 scripts/check_dm0_data_alignment.py \
  --dexdata_root /dexbotic/data/post_data_01_dexdata \
  --jsonl_dir /dexbotic/data/post_data_01_dexdata/jsonl \
  --video_dir /dexbotic/data/post_data_01_dexdata/video \
  --num_episodes 0 \
  --num_samples_per_episode 0 \
  --output_dir /dexbotic/debug_data_check/post_data_01_full_noframes
```

当前宿主机 shell 中没有 `/dexbotic` 挂载，脚本自动将 `/dexbotic` 映射到仓库根目录 `/home/guoyaokun/dexbotic` 完成审计。项目官方 Docker 启动说明使用 `-v /path/to/dexbotic:/dexbotic`，因此如果正式训练在 Docker 容器内执行，并且容器内 `/dexbotic` 正确指向当前仓库根目录，则当前注册路径和视频软链接是匹配的；这不是数据格式问题，而是 Docker volume mount 前置条件。

推荐 Docker 启动方式：

```bash
docker run -it --rm --gpus all --network host \
  -v /home/guoyaokun/dexbotic:/dexbotic \
  dexmal/dexbotic
```

注意：镜像 Dockerfile 自身 `WORKDIR` 是 `/app`，但官方运行方式会额外挂载源码到 `/dexbotic`，并在容器内 `cd /dexbotic` 工作。本项目当前数据注册文件和视频软链接都按这个运行约定组织。

## 2. 相关代码文件

| 文件 | 作用 | 风险结论 |
|---|---|---|
| [data_tools/convert_post_data_01_to_dexdata.py](/home/guoyaokun/dexbotic/data_tools/convert_post_data_01_to_dexdata.py:186) | post_data_01 专用转换；读取 tasks/episode meta/parquet/video，生成 `images_1/2/3`、16D state、14D action、prompt、extra。 | 字段生成清晰；但 `frame_idx = row["frame_index"]`，没有显式用视频 PTS 校正。当前数据 start_time 为 0，因此本批通过。 |
| [data_tools/convert_post_data_01_to_dexdata_stateful.py](/home/guoyaokun/dexbotic/data_tools/convert_post_data_01_to_dexdata_stateful.py:1) | stateful 变体，将 `observation.state.left/right_delta_tcp` 追加到 state，输出 28D state。 | 当前 `post_data_01_dexdata` 实测是 16D state，不是 stateful 28D。是否需要 stateful 取决于训练配置和实验设计。 |
| [script/convert_data/convert_lerobot_to_dexdata.py](/home/guoyaokun/dexbotic/script/convert_data/convert_lerobot_to_dexdata.py:1) | 官方示例 LeRobot 转 DexData。 | 示例脚本用 `frame_index` 直接映射视频帧，也不处理视频 start_time/PTS。 |
| [dexbotic/data/data_source/post_data_01.py](/home/guoyaokun/dexbotic/dexbotic/data/data_source/post_data_01.py:4) | 注册 `post_data_01_default`，路径指向 `/dexbotic/data/post_data_01_dexdata/{jsonl,video}`，`non_delta_mask=[6,13]`。 | 与官方 Docker 挂载方式一致；需确认训练容器内存在 `/dexbotic`。 |
| [dexbotic/data/data_source/__init__.py](/home/guoyaokun/dexbotic/dexbotic/data/data_source/__init__.py:23) | 自动 import data_source 目录和 `DEXBOTIC_DATA_PATH` 外部目录。 | 注册机制正常。 |
| [dexbotic/data/dataset/dex_dataset.py](/home/guoyaokun/dexbotic/dexbotic/data/dataset/dex_dataset.py:114) | 构建 jsonl index、按 episode/frame 加载、执行 action transform、加载图像、tokenize、返回 batch item。 | 会按 jsonl 行号取 step；若使用 `AddAction` 会对末尾 frame 随机重采样。 |
| [dexbotic/data/dataset/transform/multimodal.py](/home/guoyaokun/dexbotic/dexbotic/data/dataset/transform/multimodal.py:111) | `LoadMultiModal` 加载 `images_*`，视频用 decord 按 `frame_idx` 直接 `get_batch`。 | 不考虑 PTS/start_time；若视频首帧 start_time 非 0 或 PTS 异常，会有错位风险。本批 start_time=0。 |
| [dexbotic/data/dataset/transform/action.py](/home/guoyaokun/dexbotic/dexbotic/data/dataset/transform/action.py:229) | `AddAction`、`DeltaAction`、`AddTrajectory`、`ActionNorm`。 | 官方 DM0 默认链路会从 state 构造 action 并 delta 化；对当前已含 raw 14D delta action 的数据不适用。 |
| [dexbotic/exp/dm0_exp.py](/home/guoyaokun/dexbotic/dexbotic/exp/dm0_exp.py:94) | DM0 官方训练/compute norm 配置。 | 默认 `AddAction -> DeltaAction(enable=True)`，会忽略 jsonl 中已有 action 语义，不建议直接用于 post_data_01。 |
| [playground/post_data_01_dm0_deltafix.py](/home/guoyaokun/dexbotic/playground/post_data_01_dm0_deltafix.py:94) | post_data_01 修正版训练配置，保留 jsonl action，禁用二次 delta。 | 这是当前数据更匹配的训练入口。 |
| [dexbotic/model/dm0/dm0_arch.py](/home/guoyaokun/dexbotic/dexbotic/model/dm0/dm0_arch.py:35) | DM0 模型，默认 `action_dim=32`、`chunk_size=50`。 | state 参数传入但 prefix 不使用 state；训练主要依赖图像+文本预测 action。 |
| [dexbotic/data/collator.py](/home/guoyaokun/dexbotic/dexbotic/data/collator.py:1) | 将 dataset item 中 `action/state/image/image_masks` 映射为 `actions/states/images/image_masks`。 | shape 一致时正常 stack。 |
| [scripts/check_dm0_data_alignment.py](/home/guoyaokun/dexbotic/scripts/check_dm0_data_alignment.py:1) | 本次新增训练前检查脚本。 | 可作为正式训练前 gate。 |

## 3. DM0 官方数据格式要求

官方 DexData 每个 episode 一个 jsonl，每行一个 step。DM0 可用字段为：

| 项目 | 官方定义 | 当前自定义数据 | 是否一致 | 风险 | 建议 |
|---|---|---|---|---|---|
| 数据注册 | `register_dataset(..., prefix=...)`，训练用 `prefix_key`。 | `post_data_01_default`。 | 是 | 路径是 `/dexbotic/...`，宿主机 shell 不存在，但 Docker 官方挂载后应存在。 | 启动容器时使用 `-v /home/guoyaokun/dexbotic:/dexbotic`，并在容器内检查。 |
| 图像 key | `images_1/2/3...`，推荐主视角、左手、右手。 | `images_1=chest`，`images_2=left`，`images_3=right`。 | 是 | 无字段风险。 | 保持 `num_images=3`。 |
| 图像引用 | `{"type":"video","url":rel,"frame_idx":idx}`。 | 完全匹配。 | 是 | `post_data_01_dexdata/video` 中软链接目标是 `/dexbotic/data/post_data_01/videos/...`。 | Docker 容器内 `/dexbotic` 挂载正确时可读；否则视频不可读。 |
| prompt | `prompt` 或 `conversations`。 | `prompt` 非空，全量只有 1 个任务 prompt。 | 是 | 单 prompt 不影响格式，但 norm 不会按任务区分。 | 可接受。 |
| state | 官方示例常见 7D，也会 pad 到模型 action_dim。 | 当前全量 16D。 | 格式一致，语义需确认 | DM0 当前 forward 实际不使用连续 state。 | 不把 state 当作模型已使用条件；如要用 state 需改模型。 |
| action | 可由 state 构造，也可显式提供。DM0 输出 32D，数据 pad。 | 当前显式 14D：左 delta_tcp6、左 pinch、右 delta_tcp6、右 pinch。 | 需用自定义配置才一致 | 官方默认 DM0 配置会二次构造/delta action。 | 使用 `playground/post_data_01_dm0_deltafix.py`。 |
| action normalize | compute norm 后训练时 `ActionNorm(use_quantiles=True)`。 | 修正版配置会对当前 raw action 计算/使用 q01/q99。 | 是 | 必须重新基于当前数据计算。 | 训练前运行 compute_norm_stats。 |
| 多相机同步 | 同 step 使用相同 frame_idx。 | 全量检查三相机帧数一致，start_time 一致。 | 是 | 视频有 B-frame。 | decord 通常按显示帧索引可读，但需保留可视化抽查。 |

## 4. 当前数据实测概况

全量扫描输出目录：[debug_data_check/post_data_01_full_noframes](/home/guoyaokun/dexbotic/debug_data_check/post_data_01_full_noframes/summary.json)

| 指标 | 结果 |
|---|---|
| jsonl episode 数 | 83 |
| step 总数 | 37122 |
| 每 episode step | min 249，mean 447.25，max 780 |
| prompt 空值 | 0 |
| action dim | 全部 14 |
| state dim | 全部 16 |
| NaN/Inf | action/state 均为 0 |
| near-zero std 维度 | action/state 均无 |
| 视频引用数 | 249 |
| 缺失视频 | 0；宿主机检查通过 `/dexbotic -> /home/guoyaokun/dexbotic` remap 成立，Docker 内需确认 `/dexbotic` volume |
| 视频 fps/resolution | 30 FPS，224x224 |
| 视频 start_time 非 0 | 0 |
| B-frame 视频 | 249/249 |

## 5. Action Space 检查

当前 jsonl action 维度为 14：

| 维度 | 当前转换脚本语义 | 全量 min | 全量 max | std | 风险 |
|---|---|---:|---:|---:|---|
| 0..5 | `action.left_delta_tcp` | -0.2064 至 -0.0133 | 0.0093 至 0.1202 | 0.0020 至 0.0104 | 单位和旋转表示需人工确认。 |
| 6 | `action.left_pinch` | 0.0 | 0.9573 | 0.4044 | gripper 开合方向需人工确认。 |
| 7..12 | `action.right_delta_tcp` | -0.5252 至 -0.1228 | 0.0207 至 0.2506 | 0.0019 至 0.0228 | 右臂第 12 维存在较大负极值，建议抽样复核是否真实动作。 |
| 13 | `action.right_pinch` | 0.0 | 0.9646 | 0.3651 | gripper 开合方向需人工确认。 |

重要结论：

- 当前数据 action 是 raw delta TCP + raw pinch，不是官方默认从 state shift 得到的 absolute action。
- 不能用 [dexbotic/exp/dm0_exp.py](/home/guoyaokun/dexbotic/dexbotic/exp/dm0_exp.py:94) 默认 action pipeline 直接训练，否则会产生二次 delta/错误标签。
- 应使用 [playground/post_data_01_dm0_deltafix.py](/home/guoyaokun/dexbotic/playground/post_data_01_dm0_deltafix.py:94)，其 `DeltaAction(enable=False)` 会保留 jsonl action。
- `non_delta_mask=[6,13]` 与当前 gripper 维度一致。
- 模型内部 action_dim 默认 32，训练 pipeline 会将 14D action pad 到 32D，然后输出 50x32 chunk；推理配置再截取前 14D。
- `ActionDenorm -> AbsoluteAction` 对当前 deltafix 推理有潜在语义风险：`AbsoluteAction` 会尝试用 state 加 action，但 state/action 维度不一致时依赖 padding 后的 32D 表示。部署时若期望输出 raw delta action，应单独确认 post_data_01 推理链路是否需要跳过 `AbsoluteAction`。

## 6. Observation Space 检查

图像：

- 当前提供 `images_1/images_2/images_3`，对应 chest/left/right。
- 全量视频均为 224x224、30 FPS。
- 10 episode 抽样命令已生成可视化帧到 [debug_data_check/post_data_01/sample_frames](/home/guoyaokun/dexbotic/debug_data_check/post_data_01/sample_frames)。
- `LoadMultiModal` 会按 key 排序并截取前 `num_images` 个；当前 key 命名确保顺序为 1、2、3。
- `LoadMultiModal` 用 decord 按 frame index 读取，不按 timestamp/PTS seek。

state：

- 当前 state 维度全量为 16：left_tcp(7) + right_tcp(7) + left_pinch + right_pinch。
- state 无 NaN/Inf，无 near-zero std。
- 当前 DM0 模型 forward/inference 中 `states` 参数没有进入 prefix embedding，实际训练主要不是 state-conditioned。state 仍会用于 padding/norm/推理后处理，但不是强观测条件。
- 若希望模型显式看到 `observation.state.left/right_delta_tcp`，需要使用 stateful 转换生成 28D state，并且进一步确认模型结构是否使用 state。

## 7. 视频第一帧和 timestamp-frame 对齐

重点检查结果：

| 项目 | 结果 |
|---|---|
| jsonl `frame_index` 是否从 0 开始 | 全量 checked episode 均从 0 开始 |
| frame 是否连续 | 全量 checked episode 连续 |
| timestamp 是否从 0 开始 | 全量 checked episode 从 0.0 开始 |
| timestamp 是否单调递增 | 是 |
| `round((timestamp - episode_start_time) * fps)` 与 frame_index 最大误差 | 0 |
| 视频 start_time | 249/249 均为 0.0 |
| 多相机 start_time | 每 episode 三相机一致 |
| 多相机 frame count | 每 episode 三相机一致 |
| B-frame | 249/249 has_b_frames=2 |

结论：本批数据未复现“视频第一帧不是从时间 0 开始”的问题；转换脚本虽然没有显式减 episode 起始时间，但当前 jsonl timestamp 已经 episode 内归零，视频容器 start_time 也为 0，因此 frame_index/timestamp 对齐通过。

B-frame 是 warning，不是直接阻塞。由于训练读取按 decord `get_batch(frame_idx)` 获取显示帧，通常不会受 dts 顺序影响；但如果未来用 `CAP_PROP_POS_MSEC` 或按时间 seek，就必须重新验证。

## 8. 发现的问题和修复建议

| 优先级 | 问题 | 证据 | 建议 |
|---|---|---|---|
| P0 | Docker 容器内必须存在 `/dexbotic` volume mount。 | 官方 README/Tutorial 使用 `-v /path/to/dexbotic:/dexbotic`；[post_data_01.py](/home/guoyaokun/dexbotic/dexbotic/data/data_source/post_data_01.py:6) 和视频软链接均依赖 `/dexbotic/...`。 | 按官方方式启动容器：`-v /home/guoyaokun/dexbotic:/dexbotic`。容器内通过下面命令确认后，此项不再阻塞。 |
| P0 | 不能用官方默认 DM0 exp 训练当前数据。 | 默认 pipeline 包含 `AddAction` 和 `DeltaAction(enable=True)`。 | 使用 `playground/post_data_01_dm0_deltafix.py`。 |
| P1 | norm stats 必须基于当前数据和 deltafix pipeline 重算。 | action 语义为 raw 14D delta action。 | 训练前运行 `python3 playground/post_data_01_dm0_deltafix.py --task compute_norm_stats`。 |
| P1 | 当前数据是 16D state，不是 stateful 28D。 | 全量 state_dims={16:37122}。 | 如果实验目标是让模型看到 delta_tcp observation，需要重新用 stateful converter 转换并注册。 |
| P1 | 所有视频含 B-frame。 | 全量 249/249 has_b_frames=2。 | 当前 frame-index 读取通过；若出现视觉错位，优先转码为无 B-frame 或逐帧图片验证。 |
| P2 | gripper 方向/单位无法从代码自动确认。 | 维度 6/13 范围 0..约 0.96。 | 人工确认 0/1 含义、开合方向、训练部署端是否一致。 |
| P2 | action 连续维单位/旋转表示需确认。 | 转换脚本只命名为 delta_tcp，未写明 m/mm、rad/degree。 | 人工对照采集端控制接口。 |

## 9. 训练前 Checklist

- [x] 数据路径存在：宿主机通过 remap 后存在；Docker 训练容器内需确认 `/dexbotic` 挂载。
- [x] jsonl 数量正确：83。
- [x] episode 数量正确：83。
- [x] 每个 episode step 数合理：249..780。
- [x] 视频数量与相机数量匹配：83 x 3 = 249。
- [x] 图像/视频可读：remap 后 ffprobe/ffmpeg 可读，已抽样落图。
- [x] prompt 非空。
- [x] action 维度一致：14。
- [x] state 维度一致：16。
- [x] action/state 无 NaN、Inf。
- [x] action/state 无 near-zero std。
- [ ] norm stats 基于当前数据重新计算。
- [ ] raw/norm 数据没有混用：需在训练入口确认使用 deltafix。
- [x] 视频第一帧时间戳检查通过。
- [x] timestamp-frame_index 对齐检查通过。
- [x] 多相机帧数和时间一致。
- [x] 随机抽样可视化已生成，建议人工肉眼抽查。
- [ ] dataloader 能正常取 batch：当前 Python 环境缺少训练依赖，未执行。
- [ ] batch tensor shape 正确：待训练环境执行 smoke test。
- [ ] 小规模 overfit/smoke train 跑通。

## 10. 是否建议现在开始训练

暂不建议在未进入训练容器确认路径前直接正式训练。数据内容本身在 timestamp/frame/video 对齐上通过了本次审计；路径项在 Docker 按官方方式挂载后应解除。

容器内必须先执行：

```bash
ls -ld /dexbotic
ls /dexbotic/data/post_data_01_dexdata/jsonl | head
ls -l /dexbotic/data/post_data_01_dexdata/video/observation.images.chest/chunk-000/file-000.mp4
ls /dexbotic/data/post_data_01/videos/observation.images.chest/chunk-000/file-000.mp4
```

若这些命令都通过，路径/软链接项不阻塞训练。剩余必须项是使用 `playground/post_data_01_dm0_deltafix.py` 并重新计算 norm stats，不能直接使用官方默认 `dexbotic/exp/dm0_exp.py`。

完成容器路径检查和 norm stats 后，可以先跑 dataloader smoke test 和 one-episode overfit，再进入正式 DM0 微调。
