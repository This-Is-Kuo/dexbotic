# DM0 Open-loop 评估链路开发与问题排查报告

## 1. 项目背景

本次工作围绕 `Dexbotic/DM0` 在自定义快递分拣数据 `post_data_01` 上的微调与 open-loop 评估异常展开。

当前任务基于自采集的 LeRobot 风格数据复现 DM0，前期已经完成：

- LeRobot 数据到 DexData/jsonl 的转换
- 自定义数据源注册
- 自定义 DM0 benchmark 配置
- 模型训练与 checkpoint 保存
- open-loop action 预测可视化

最初虽然已经能跑通训练和评估，但 open-loop 结果和预期差距较大，因此本轮工作的重点不是改模型大架构，而是系统排查：

- 数据 target 是否定义正确
- state/action 是否时间对齐
- open-loop 评估是否正确展开 chunk
- 归一化链路是否正确
- 哪些问题来自模型本身，哪些问题来自评估展示方式

---

## 2. 最初 open-loop 的主要异常

在最早的 `checkpoint-400` open-loop 图上，主要问题表现为：

- 连续动作维度趋势不完全一致，部分维度存在明显 bias 和尺度偏移
- `action_6` / `action_13` 这类夹爪或开关维度存在提前、滞后、毛刺、中间值不稳
- raw 图中 `pred` 抖动明显大于 `gt`
- normalized 图中依然存在明显不对齐，说明问题不只是 raw 反归一化显示
- chunked timestep 视图下段间跳变明显

最早的直觉怀疑包括：

- action 维度顺序错误
- norm stats 用错
- obs-action 时间错位
- action absolute / delta 语义搞混
- open-loop eval chunk 展开方式不合理
- checkpoint 太早，模型尚未收敛

后续排查证明，上述怀疑里真正的主因主要集中在：

- benchmark 对 action 的定义错误
- 转换后的 state 信息缺失
- open-loop 旧版 chunk 展示方式放大了抖动

---

## 3. 代码阅读后的关键发现

### 3.1 原始自定义 benchmark 存在“二次 delta”问题

原始自定义 benchmark 使用的动作流水线为：

`PadState -> PadAction -> AddTrajectory(50) -> DeltaAction(enable=True) -> ActionNorm`

而数据转换脚本实际导出的 `action` 已经是：

- `action[0:6]`：左臂 `delta_tcp`
- `action[6]`：左夹爪 / pinch
- `action[7:13]`：右臂 `delta_tcp`
- `action[13]`：右夹爪 / pinch

也就是说，原始连续动作已经是 delta 语义，不应该再做一次 `DeltaAction`。原 benchmark 中这一步会把连续动作 target 再减一遍当前 state，直接污染训练标签。

这是最早 open-loop 不理想的第一主因。

相关代码位置：

- [convert_post_data_01_to_dexdata.py](/home/guoyaokun/dexbotic/data_tools/convert_post_data_01_to_dexdata.py:206)
- [playground/benchmarks/custom/post_data_01_dm0.py](/home/guoyaokun/dexbotic/playground/benchmarks/custom/post_data_01_dm0.py:1)
- 修正版 benchmark：[playground/post_data_01_dm0_deltafix.py](/home/guoyaokun/dexbotic/playground/post_data_01_dm0_deltafix.py:1)

### 3.2 原始 state 丢掉了最关键的 `delta_tcp` 观测

原始 LeRobot 数据中本来就包含：

- `observation.state.left_delta_tcp`
- `observation.state.right_delta_tcp`

但最早的 DexData 转换只保留了：

- `left_tcp(7)`
- `right_tcp(7)`
- `left_pinch(1)`
- `right_pinch(1)`

也就是只有 16 维 state，没有把 12 维 `delta_tcp` 状态带进去。

这会导致模型去预测本来在原始观测里已经存在的控制量，尤其伤害：

- `action_9`
- `action_10`
- `action_11`
- `action_12`

这是最早 open-loop 不理想的第二主因。

为此后续新增了 stateful 转换版本，state 扩展为 28 维：

- `0..6`：left_tcp
- `7..13`：right_tcp
- `14`：left_pinch
- `15`：right_pinch
- `16..21`：left_delta_tcp
- `22..27`：right_delta_tcp

相关代码位置：

- [convert_post_data_01_to_dexdata_stateful.py](/home/guoyaokun/dexbotic/data_tools/convert_post_data_01_to_dexdata_stateful.py:1)
- [post_data_01_stateful.py](/home/guoyaokun/dexbotic/dexbotic/data/data_source/post_data_01_stateful.py:1)

### 3.3 open-loop 旧版 chunk 展示方式会放大段间抖动

DM0 一次推理输出的是一个 action chunk，而不是单步 action。

最早的 `chunked timestep` 图，本质上是把每个 `[H, D]` chunk 直接顺序摊平，中间插入 gap，没有对重叠 timestep 做 temporal ensemble。这会导致：

- chunk 间不一致被直接显示出来
- 段间跳变被视觉放大
- `pred` 看起来比真实执行时更抖

后续在 `eval_openloop.py` 中加入：

- `chunk`
- `first`
- `mean`
- `exp`

四种 `chunk_merge` 模式后，`mean/exp` 的结果显著稳定，说明这一点确实是最早 open-loop “看起来很差”的重要原因之一。

### 3.4 norm 链路不是主因

通过新增的 `check_action_norm_inverse.py` 检查：

- `raw -> normalize -> unnormalize -> raw`

最终误差约为：

- `max_error ≈ 1.66e-7`

说明当前 norm / denorm 链路基本互逆，不是主要问题来源。

### 3.5 当前 DM0 训练链路里，连续 state 张量并没有真正进入模型

这是最后阶段才确认的一个很关键的实现事实。

虽然数据集和 dataloader 会把 `state` 张量带进 batch，但在当前 DM0 的 `forward` 和 `inference_action` 里，`states` 参数实际上没有被用到：

- [dexbotic/model/dm0/dm0_arch.py](/home/guoyaokun/dexbotic/dexbotic/model/dm0/dm0_arch.py:416)
- [dexbotic/model/dm0/dm0_arch.py](/home/guoyaokun/dexbotic/dexbotic/model/dm0/dm0_arch.py:518)

这意味着当前模型主要还是依赖：

- 图像前缀
- 文本 prompt

来预测 action，而不是直接读取 `state` 连续值。

这个发现并不否定 `stateful` 数据修复的价值，但它解释了为什么像 `action_10` 这类本质上更像内部控制量的维度，即使已经被放回 state，仍然不一定立刻学得很好。

---

## 4. 修改迭代记录

## 第一轮修改：补齐 open-loop 诊断链路

### 目标

最初只有图像曲线，难以区分问题来自：

- 模型本身
- target 定义
- norm
- lag
- chunk 展示方式

因此第一轮优先把 open-loop 结果做成“可量化、可保存、可复查”的形式。

### 新增/修改文件

- [eval_openloop.py](/home/guoyaokun/dexbotic/openloop/eval_openloop.py:1)
- [debug_openloop_metrics.py](/home/guoyaokun/dexbotic/openloop/tools/debug_openloop_metrics.py:1)
- [lag_correlation_check.py](/home/guoyaokun/dexbotic/openloop/tools/lag_correlation_check.py:1)
- [check_action_norm_inverse.py](/home/guoyaokun/dexbotic/openloop/tools/check_action_norm_inverse.py:1)
- [compare_openloop_checkpoints.py](/home/guoyaokun/dexbotic/openloop/tools/compare_openloop_checkpoints.py:1)
- [run_dm0_one_episode_overfit.py](/home/guoyaokun/dexbotic/openloop/tools/run_dm0_one_episode_overfit.py:1)

### 主要增强内容

- 保存 `pred_raw.npy / gt_raw.npy / pred_norm.npy / gt_norm.npy`
- 输出 per-dim 指标：
  - `mae`
  - `rmse`
  - `pearson`
  - `bias`
  - `std_gt`
  - `std_pred`
  - `std_ratio`
  - `max_abs_error`
- 输出 lag 搜索结果：
  - `best_lag`
  - `best_corr`
  - `corr_at_lag0`
  - `improvement`
- 输出 raw / normalized 图
- 支持 one-episode overfit 诊断
- 支持多 checkpoint 对比

这一步的意义是把“看起来不好”拆成：

- 哪些维度趋势没学到
- 哪些维度只是幅值偏平
- 哪些维度有固定 lag
- 哪些问题只是画图方式放大出来的

## 第二轮修改：增强 chunk merge 与 temporal ensemble

### 目标

解决 legacy chunk 视图直接摊平 chunk，导致段间抖动被放大的问题。

### 修改内容

在 [eval_openloop.py](/home/guoyaokun/dexbotic/openloop/eval_openloop.py:1) 中新增：

- `--chunk_merge chunk`
- `--chunk_merge first`
- `--chunk_merge mean`
- `--chunk_merge exp`

并修复：

- merge 时间轴长度按 `max(frame_index)+1` 构建
- 稀疏 / 打乱 frame index 不再被截断
- 保存数组与 metrics 更完整

### 结果

这一步证明：

- 不做 temporal ensemble 会明显放大抖动
- `mean` 更适合做默认连续动作分析
- `exp` 更适合夹爪 / 开关动作

## 第三轮修改：修复 benchmark 中的动作定义错误

### 目标

解决原 benchmark 对已是 delta 的动作再次执行 `DeltaAction` 的问题。

### 修改内容

- 新增修正版 benchmark：
  [playground/post_data_01_dm0_deltafix.py](/home/guoyaokun/dexbotic/playground/post_data_01_dm0_deltafix.py:1)
- 新增对齐检查脚本：
  [check_post_data_action_alignment.py](/home/guoyaokun/dexbotic/openloop/tools/check_post_data_action_alignment.py:1)

### 结论

该修复解释了最早许多连续维度的：

- 趋势错位
- 幅值怪异
- lag 明显

因为当时训练 target 本身就是错的。

## 第四轮修改：修复 state 缺失，增加 stateful 数据版本

### 目标

把原始观测中已经存在但在转换时丢掉的 `delta_tcp` 状态重新带回数据。

### 修改内容

- 新增 stateful 转换脚本：
  [convert_post_data_01_to_dexdata_stateful.py](/home/guoyaokun/dexbotic/data_tools/convert_post_data_01_to_dexdata_stateful.py:1)
- 新增数据注册：
  [post_data_01_stateful.py](/home/guoyaokun/dexbotic/dexbotic/data/data_source/post_data_01_stateful.py:1)
- benchmark 支持切换 `dataset_name`

### 结果

state 从 16 维扩到 28 维后，后续训练的 raw open-loop 误差显著下降，说明这个修复是有效的。

## 第五轮修改：训练与推理环境修复

为了让训练和评估稳定复现，还做了若干工程性修复：

- `eval_openloop.py` 增加 `--single-gpu-id`
- `dexbotic/exp/dm0_exp.py` 支持 `device_map / cuda_device`
- `dexbotic/exp/base_exp.py` 兼容两种 `norm_stats.json` 格式
- benchmark 支持：
  - `DEXBOTIC_OUTPUT_DIR`
  - `DEXBOTIC_WARMUP_STEPS`
  - `DEXBOTIC_DEEPSPEED_CONFIG`
  - `DEXBOTIC_TRAIN_BATCH_SIZE`
  - `DEXBOTIC_GRAD_ACCUM`
  - `DEXBOTIC_DATASET_NAME`
  - `DEXBOTIC_BASE_MODEL`
- 新增续训脚本：
  [resume_dm0_and_eval.py](/home/guoyaokun/dexbotic/openloop/tools/resume_dm0_and_eval.py:1)

同时解决了：

- wandb 版本兼容问题
- Docker `/dev/shm` 太小导致的 NCCL 问题
- 单卡 / 多卡训练与推理切换问题
- 多次 OOM 问题

---

## 5. 不同实验阶段的结果演化

这里需要明确区分三条实验线：

### 5.1 旧 baseline：原 benchmark + 原数据

代表 checkpoint：

- [old checkpoint-400](/home/guoyaokun/dexbotic/user_checkpoints/dexbotic/custom_dm0/post_data_01-0416/checkpoint-400)

对应结果：

- [openloop_metrics.json](/home/guoyaokun/dexbotic/openloop/artifacts/retrain_compare/old_ckpt400_mean/openloop_metrics.json:1)

指标：

- `raw_action_mae = 0.0976`
- `normalized_action_mae = 0.2879`

这个版本虽然某些维度 Pearson 看上去不低，但其训练 target 是污染过的，不能作为最终可信模型定义。

### 5.2 中间版本：deltafix，但仍是旧 16 维 state

这一阶段修正了“二次 delta”，但还没有把 `delta_tcp` 放回 state。

结果表现为：

- 比错误 benchmark 明显更合理
- 但 `action_9/10/11` 仍然偏弱
- 说明仅修正 target 还不够

### 5.3 最终版本：deltafix + stateful(28D state)

这是当前最可信的实验线。

代表 checkpoint：

- [checkpoint-300](/home/guoyaokun/dexbotic/user_checkpoints/dexbotic/custom_dm0/post_data_01_stateful_deltafix/checkpoint-300)
- [checkpoint-400](/home/guoyaokun/dexbotic/user_checkpoints/dexbotic/custom_dm0/post_data_01_stateful_deltafix/checkpoint-400)
- [checkpoint-1000](/home/guoyaokun/dexbotic/user_checkpoints/dexbotic/custom_dm0/post_data_01_stateful_deltafix/checkpoint-1000)

对应 open-loop 结果：

- [checkpoint-300](/home/guoyaokun/dexbotic/openloop/artifacts/stateful_compare/checkpoint-300/openloop_metrics.json:1)
- [checkpoint-400](/home/guoyaokun/dexbotic/openloop/artifacts/stateful_compare/checkpoint-400/openloop_metrics.json:1)
- [checkpoint-1000](/home/guoyaokun/dexbotic/openloop/artifacts/stateful_compare/checkpoint-1000/openloop_metrics.json:1)

核心指标：

| checkpoint | raw_action_mae | normalized_action_mae |
|---|---:|---:|
| old-400 | 0.0976 | 0.2879 |
| stateful-300 | 0.0402 | 0.3825 |
| stateful-400 | 0.0344 | 0.3850 |
| stateful-1000 | 0.0239 | 0.3076 |

这里的解读要特别注意：

- raw 空间下，stateful 版本有显著改善
- normalized MAE 不是一开始就最优，但随着连续训练推进逐步接近旧 baseline
- 最终 `checkpoint-1000` 已经是当前最好的 stateful 模型

---

## 6. 最终状态下的 per-dim 结论

当前最佳模型为：

- [checkpoint-1000](/home/guoyaokun/dexbotic/user_checkpoints/dexbotic/custom_dm0/post_data_01_stateful_deltafix/checkpoint-1000)

重点维度指标见：

- [per_dim_metrics_norm.csv](/home/guoyaokun/dexbotic/openloop/artifacts/stateful_compare/checkpoint-1000/per_dim_metrics_norm.csv:1)
- [lag_metrics.csv](/home/guoyaokun/dexbotic/openloop/artifacts/stateful_compare/checkpoint-1000/lag_metrics.csv:1)

关键维度结果：

- `action_6`: `pearson = 0.952`
- `action_13`: `pearson = 0.942`
- `action_8`: `pearson = 0.858`
- `action_12`: `pearson = 0.846`
- `action_11`: `pearson = 0.661`
- `action_9`: `pearson = 0.540`
- `action_10`: `pearson = 0.176`

其中：

- `6 / 13` 已不再是主要问题
- `8 / 12` 已明显学到趋势
- `11` 相比之前也已明显改善
- `10` 仍然是最弱维度

lag 上，`6 / 8 / 11 / 12 / 13` 已基本接近 0 或 1 帧偏移，不再像早期那样存在显著固定 lag。

---

## 7. 最初 open-loop 不行的原因总结

综合整个会话，最早 open-loop 不理想的主因不是单一因素，而是以下几项叠加：

### 原因 1：原 benchmark 对已是 delta 的连续动作又做了一次 DeltaAction

这是最关键、最确定的问题。

它直接导致：

- 连续动作 target 被污染
- 趋势偏移
- 幅值异常
- lag 被放大

### 原因 2：转换后的 state 丢掉了 `left/right_delta_tcp`

模型要预测的许多 action 维，本来就在原始 state 中有同源观测，但转换时被丢掉了。

这尤其伤害了右臂一些较难的 delta 控制维度，如 `9/10/11/12`。

### 原因 3：legacy chunk 展示方式放大了段间抖动

最早 open-loop 图并没有做合理 temporal ensemble，因此 chunk 间跳变被直接画出来，视觉上显得比模型真实能力更差。

### 原因 4：早期 checkpoint 尚未收敛

在修复 target 与 state 定义之前，许多早期 checkpoint 的结果本来就不稳定，不能代表最终能力。

### 原因 5：某些维度本身不是简单 pose 差分

例如 `action_10`，后续排查表明它实际上是：

- `right_delta_tcp[3]`
- 与当前 `state[25]` 完全相同
- 但与右臂 pose 各维相关性较弱

这类维度更像内部控制量，不是单靠视觉和 pose 就容易恢复的信号。

### 原因 6：当前 DM0 实现并未真正使用连续 state 张量

虽然 state 已经进入数据集和 batch，但在当前 DM0 `forward / inference_action` 中并没有实际参与计算。

这也是为什么即使把 `delta_tcp` 放回 state，某些内部控制量维度的提升仍不如夹爪或位置相关维度那样直接。

---

## 8. 当前修改后的效果

经过以上多轮修改，当前 open-loop 评估链路已经具备：

1. 保存 `pred/gt` raw 与 norm 数组
2. 输出 per-dim 指标
3. 检查 norm inverse
4. 搜索每维 lag
5. 对比多个 checkpoint
6. 进行 one-episode overfit
7. 支持 `chunk / first / mean / exp` 多种 merge 模式
8. 支持更稳定的单卡推理控制

当前最重要的结论是：

> DM0 并不是“完全没学到”。最初 open-loop 看起来很差，主要是因为 benchmark 动作定义错误、state 关键信息缺失、legacy chunk 视图放大了抖动，再叠加早期 checkpoint 未收敛。修复这些问题后，模型已经在 raw 空间下取得显著改善，并在多个关键维度上学到稳定趋势。

---

## 9. 对原草稿的修正建议

你当前那份草稿整体结构是对的，但如果直接保留，建议补以下几点：

### 9.1 需要明确区分三个阶段

原草稿里把以下三种状态混在了一起：

- 原 benchmark / 原数据
- `deltafix` 但非 stateful
- `deltafix + stateful`

这会让后续读者误以为 `checkpoint-400` 仍然是当前最终最优结果。实际上现在最可信的是 `stateful checkpoint-1000`。

### 9.2 需要把“二次 delta”写成最高优先级根因

原草稿更强调 chunk merge、lag、checkpoint 早期等因素，但从最终结论看，最关键的第一根因其实是：

- 原 benchmark 对已是 delta 的动作再次执行 `DeltaAction`

### 9.3 需要补充“state 缺失”问题

原草稿没有完整覆盖：

- 原转换后的 state 缺少 `left/right_delta_tcp`

这是后续能把 raw MAE 大幅压下来的关键修复。

### 9.4 需要补充“DM0 当前并未真正使用连续 state”

这点很重要，因为它决定了当前 `action_10` 一类问题不能简单归结为“多训练一点就好”。

### 9.5 checkpoint-200/300/400 那部分要标明属于中间阶段

原草稿中关于 `checkpoint-200/300/400` 的分析大体正确，但它描述的是中间调试阶段，不应再作为“最终版本”的结论部分。

---

## 10. 下一步建议

当前推荐把以下模型作为主基线：

- [checkpoint-1000](/home/guoyaokun/dexbotic/user_checkpoints/dexbotic/custom_dm0/post_data_01_stateful_deltafix/checkpoint-1000)

推荐后续工作：

1. 继续基于该 checkpoint 做 `mean vs exp` 和夹爪后处理对比
2. 若要继续提升 `action_10`，优先考虑让模型真正使用连续 state，而不是单纯继续加训练步数
3. 若准备进入 closed-loop 或真机测试，优先关注：
   - `6 / 13` 的切换稳定性
   - `9 / 10 / 11` 的右臂控制精度
   - `chunk_merge` 策略在部署中的影响

---

## 11. 最终结论

本次开发从“open-loop 图像不理想”出发，逐步把问题从表象拆到了数据定义、state 定义和评估实现三个层面。

最终确认：

- 最早 open-loop 不行，不是因为模型完全没学到
- 主因是原 benchmark 的二次 delta、state 关键信息缺失，以及旧版 chunk 展示方式不合理
- 修复后，当前 `stateful checkpoint-1000` 已在 raw 空间下显著优于最早 baseline
- 当前剩余最弱点集中在 `action_10`
- 对 `action_10` 的进一步优化，不建议只靠继续训练，更值得优先让模型真正使用连续 state

这标志着当前工作已经从“能否跑通 DM0 微调”进入到“如何让 open-loop 与真实控制语义更一致”的阶段。
