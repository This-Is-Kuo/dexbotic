# 上周工作说明：2026-05-18 至 2026-05-24

## 说明范围

本说明按 2026-05-18 到 2026-05-24 这个自然周整理。当前分支 `dev` 在该时间段内没有可检索到的 git commit，因此本文主要基于当前工作区中的训练日志、评估结果和未提交修改归纳。

上周工作的重点是重新训练 DM0 在 `post_origin_data_0423` 数据集上的模型，并围绕训练结果做数据检查、断点续训和 open-loop 评估。

## 一、重新训练目标

本轮重新训练的目标是验证 `post_origin_data_0423` 数据在 DM0 上的可训练性，并产出一个更完整的 15k step checkpoint，用于后续 open-loop 对比和机器人动作效果验证。

训练任务主要围绕以下问题展开：

- 将 `post_origin_data_0423` 数据接入 DM0 训练流程。
- 使用 delta action 形式重新训练模型。
- 检查训练数据的 state/action 维度、视频引用、timestamp 和动作连续性。
- 支持训练中断后的 checkpoint resume，保证训练可以从中间结果继续推进。
- 对关键 checkpoint 做 open-loop 评估，观察训练步数增加后的指标变化。

## 二、训练数据和预检查

训练数据集为 `post_origin_data_0423_train`，测试/评估数据集为 `post_origin_data_0423_test`。

在训练前，对转换后的 DexData 做了数据质量检查。检查结果保存在：

- `debug_data_check/post_data_overlay_0423_dexdata/summary.json`
- `debug_data_check/post_data_overlay_0423_dexdata/per_episode_report.csv`
- `debug_data_check/post_data_overlay_0423_dexdata/per_camera_video_report.csv`
- `debug_data_check/post_data_overlay_0423_dexdata/suspicious_samples.csv`

抽查结果如下：

| 项目 | 结果 |
| --- | --- |
| 检查 episode 数 | 10 |
| 检查样本行数 | 3739 |
| action 维度 | 14 |
| state 维度 | 16 |
| 空 prompt 数量 | 0 |
| 引用视频数量 | 30 |
| 缺失视频数量 | 0 |
| error 数量 | 0 |
| warning 数量 | 0 |

结论：抽查范围内数据结构正常，state/action 维度符合 DM0 训练要求，视频引用完整，没有发现明显数据错误。

## 三、重新训练配置

本轮训练使用脚本：

- `scripts/train_post_origin_data_0423_4gpu.sh`

训练输出目录：

- `user_checkpoints/dexbotic/custom_dm0/post_origin_data_0423_deltafix_15k`

主要训练配置如下：

| 配置项 | 设置 |
| --- | --- |
| 数据集 | `post_origin_data_0423_train` |
| 目标训练步数 | 15000 |
| 训练方式 | DM0 SFT / delta action |
| GPU 数量 | 4 卡 torchrun |
| Deepspeed | zero3 |
| per-device batch size | 2 |
| gradient accumulation | 8 |
| base learning rate | 2e-6 |
| min learning rate | 5e-7 |
| warmup steps | 200 |
| save steps | 500 |
| logging steps | 10 |
| W&B run | `post_origin_data_0423_deltafix_15k` |

训练过程中也补充了续训逻辑：脚本会自动识别最新 checkpoint，并根据当前 GPU world size 决定是直接 resume 还是 warm restart。同时支持保持日志 step 连续，避免 W&B 曲线和 checkpoint 编号断裂。

## 四、训练过程

训练过程中先遇到过一次保存失败，原因是磁盘空间不足，日志中出现：

- `PytorchStreamWriter failed writing file`
- `No space left on device`

随后清理/恢复环境后，于 2026-05-23 从 `checkpoint-14500` 继续训练，并成功训练到 `checkpoint-15000`。

最终训练完成记录如下：

| 项目 | 结果 |
| --- | --- |
| resume 起点 | `checkpoint-14500` |
| 最终 step | 15000 |
| 最终 epoch | 3.4091 |
| 最后一次 step loss | 0.2134 |
| 最后一次 grad norm | 2.5537 |
| 最后一次 learning rate | 1.0305e-6 |
| 训练 runtime | 4865.51 秒 |
| train samples/s | 197.307 |
| train steps/s | 3.083 |
| trainer_state 记录的 train_loss | 0.00762 |

最终模型和 tokenizer 等文件已保存到：

- `user_checkpoints/dexbotic/custom_dm0/post_origin_data_0423_deltafix_15k`

关键 checkpoint 包括：

- `checkpoint-10000`
- `checkpoint-10500`
- `checkpoint-11000`
- `checkpoint-11500`
- `checkpoint-12000`
- `checkpoint-12500`
- `checkpoint-13000`
- `checkpoint-13500`
- `checkpoint-14000`
- `checkpoint-14500`
- `checkpoint-15000`

## 五、训练结果评估

训练后对关键 checkpoint 做了 open-loop 评估，结果目录为：

- `openloop/artifacts/post_origin_0423_15k_strided_key_checkpoints_b20/`

评估设置：

| 项目 | 设置 |
| --- | --- |
| 评估数据集 | `post_origin_data_0423_test` |
| 评估样本数 | 80 |
| inference stride | 50 |
| plot | 关闭，仅保存指标 |

关键 checkpoint 的 open-loop 指标如下：

| checkpoint | raw action MSE | raw action MAE | normalized MSE | normalized MAE |
| --- | ---: | ---: | ---: | ---: |
| 10000 | 0.000985 | 0.005664 | 0.057009 | 0.153300 |
| 12500 | 0.001068 | 0.005757 | 0.056565 | 0.152084 |
| 15000 | 0.001013 | 0.005686 | 0.056831 | 0.151988 |

从结果看：

- `checkpoint-15000` 的 normalized MAE 最低，为 0.151988。
- `checkpoint-12500` 的 normalized MSE 最低，为 0.056565。
- `checkpoint-10000` 的 raw action MSE 和 raw action MAE 略低，但 normalized 指标不如后续 checkpoint。
- 整体来看，10000 到 15000 step 之间指标变化幅度不大，模型已经进入相对稳定区间。

综合 normalized MAE 和最终训练完整性，本轮训练可以优先使用 `checkpoint-15000` 作为后续实验 checkpoint；如果更关注 normalized MSE，也可以保留 `checkpoint-12500` 作为对照。

## 六、补充评估结果

还进行过一次更完整样本量的 open-loop 评估，对比 `checkpoint-5500` 和 `checkpoint-6000`，评估样本数为 33150：

| checkpoint | raw action MSE | raw action MAE | normalized MSE | normalized MAE |
| --- | ---: | ---: | ---: | ---: |
| 5500 | 0.001410 | 0.007025 | 0.068090 | 0.170720 |
| 6000 | 0.001468 | 0.006920 | 0.067555 | 0.170500 |

这组结果说明，在较早训练阶段从 5500 到 6000 step 时，normalized 指标已有小幅改善。后续 10000 到 15000 step 的关键 checkpoint 对比中，normalized MAE 进一步下降到约 0.152，说明继续训练对归一化动作预测仍有收益，但收益逐步变小。

## 七、为重新训练做的配套修改

为了支撑这轮重新训练，上周还做了一些配套工程改动：

- 增加 `data_tools/convert_lerobot_v2_post_origin_to_dexdata.py`，支持把 LeRobot v2 / origin_102 数据转换成 DexData。
- 增加 `data_tools/prepare_post_origin_dexdata.py`，串联数据转换、注册和预检查。
- 增加 `data_tools/diagnose_origin102_action_alignment.py`，诊断 action 是同帧目标还是未来帧目标，并检查 quaternion 符号翻转。
- 增强 `scripts/check_dm0_data_alignment.py`，增加 timestamp 抖动、严格递增、action/state step 突刺等检查。
- 修改 `playground/post_data_01_dm0_deltafix.py`，支持 15k 训练、环境变量配置、checkpoint resume、日志 step offset 和最终模型保存。
- 修改 `dexbotic/exp/base_exp.py`，训练初始化时按 `LOCAL_RANK` 设置 CUDA device，减少多卡设备绑定问题。
- 修改 `dexbotic/exp/dm0_exp.py`，使 norm stats 计算不依赖在线加载 image processor。
- 增强 `openloop/eval_openloop.py`，支持 inference stride、无图评估、plot 失败不影响 metrics 保存。
- 修改 `openloop/tools/compare_openloop_checkpoints.py`，支持批量 checkpoint 对比时透传 stride 和 no-plots 参数。

## 八、结论

上周主要完成了 `post_origin_data_0423` 上的 DM0 重新训练，并成功产出 15k step 模型。训练过程中处理了磁盘空间导致的 checkpoint 保存失败问题，随后从 `checkpoint-14500` 成功续训到 `checkpoint-15000`。

从 open-loop 指标看，`checkpoint-15000` 在 normalized MAE 上表现最好，可以作为当前主推荐模型；`checkpoint-12500` 在 normalized MSE 上略优，可以作为备选对照。整体指标从早期 5500/6000 step 到后期 10000/12500/15000 step 有改善，但后期提升趋于平稳，后续更适合结合真实/仿真 rollout 效果继续判断 checkpoint 选择。
