# 周报：2026-05-25 至 2026-05-31

## 一、本周概览

本周围绕 DM0 在新增真实采集数据上的训练和部署验证，完成了从数据整理、两阶段全量微调、open-loop 评估，到真实推理服务接入的完整链路建设。

主要进展如下：

- 完成新增 `post_data` 数据的合并转换、视频无 B-frame 转码和 train/test 切分。
- 基于旧数据与新增数据设计并跑通两阶段全量微调流程。
- 完成 Stage 1 混合数据训练至 `checkpoint-6500`，并从该权重 warm-start Stage 2 新数据慢速微调至 `checkpoint-2000`。
- 完成 Stage 2 在旧测试集和新增测试集上的全量 open-loop 评估。
- 接入真实 HTTP 推理服务，补充 UI Agent adapter、XLeRobot 异步 bridge 和在线回放验证工具。

## 二、新增数据整理与预处理

本周将新增 `post_data*.zip` 数据接入 DM0 数据链路。新增工具支持从多份 LeRobot v2 压缩包中读取 parquet、任务描述和三路相机视频，统一生成 DM0 所需的 DexData 格式。

为降低视频随机访问和解码稳定性问题，对旧数据和新增数据的视频统一执行了无 B-frame 转码：

| 数据集 | 转码视频数 | 失败数 | 编码设置 |
| --- | ---: | ---: | --- |
| 旧数据 `post_origin_data_0423` | 1644 | 0 | H.264, `bf=0`, `g=1` |
| 新数据 `post_data_merged` | 831 | 0 | H.264, `bf=0`, `g=1` |

转码后重新按照固定随机种子 `42` 进行 train/test 切分：

| 数据集 | split | episode 数 | 帧数 | 视频数 |
| --- | --- | ---: | ---: | ---: |
| 旧数据 | train | 494 | 284056 | 1482 |
| 旧数据 | test | 54 | 30704 | 162 |
| 新数据 | train | 249 | 277098 | 747 |
| 新数据 | test | 28 | 30949 | 84 |

训练集合计包含 `743` 个 episode、`561154` 帧数据。新数据中包含 `express_pick` 和 `fold_cloth` 两类任务，用于增强模型对新增场景和慢速动作分布的适应能力。

同时补充了训练前预检查脚本，用于检查 episode 连续性、state/action 数值合法性、视频路径、帧数、起始时间和 B-frame 情况，避免数据问题进入长时间训练任务。

## 三、两阶段全量微调

本轮训练采用两阶段 full fine-tune 策略，目标是在保留旧数据能力的同时，提高模型对新增真实数据分布的拟合能力。

### Stage 0：统一 norm stats

首先基于“旧训练集 + 新训练集”计算统一的 action norm stats，并将结果备份到数据盘。Stage 1 和 Stage 2 均固定复用该统计量，避免第二阶段只使用新数据重新计算 norm stats 后造成动作尺度漂移。

### Stage 1：旧数据与新数据混合全量微调

Stage 1 使用旧训练集和新训练集混合训练，从 `DM0-base` 初始化：

| 配置项 | 设置 |
| --- | --- |
| 训练数据 | 旧数据 train + 新数据 train |
| 训练方式 | 全参数微调 |
| 目标 step | 10000 |
| 当前最新 checkpoint | `checkpoint-6500` |
| base learning rate | `2.5e-5` |
| min learning rate | `2.5e-6` |
| warmup steps | `1000` |
| per-device batch size | `2` |
| gradient accumulation | `8` |
| checkpoint-6500 loss | `0.2098` |

### Stage 2：新增慢速数据专项微调

Stage 2 从 Stage 1 最新权重 warm-start，使用新增数据进行更小学习率的全量微调。该阶段是权重初始化，不是 optimizer/scheduler 的断点续训。

| 配置项 | 设置 |
| --- | --- |
| 初始化权重 | Stage 1 `checkpoint-6500` |
| 训练数据 | 新数据 train |
| 训练方式 | 全参数微调 |
| 目标 step | 10000 |
| 当前最新 checkpoint | `checkpoint-2000` |
| base learning rate | `5e-6` |
| min learning rate | `5e-7` |
| warmup steps | `500` |
| per-device batch size | `2` |
| gradient accumulation | `8` |
| checkpoint-2000 loss | `0.1121` |

Stage 2 在保存 `checkpoint-2000` 后继续运行至约 `2026` step，随后任务收到 `SIGTERM` 停止。当前已有可用的 `checkpoint-2000`，可以继续用于评估和后续续训。

## 四、最新 Open-loop 结果

本周对 Stage 2 `checkpoint-2000` 完成了旧测试集和新增测试集的全量 open-loop 评估。评估使用 `stride=1`，即逐帧推理，并保存了整体指标、分 horizon 指标、分维度指标、lag correlation 和预测数组。

| checkpoint | 测试集 | 样本数 | raw MSE | raw MAE | normalized MSE | normalized MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Stage 2 `checkpoint-2000` | 旧数据 test | 30704 | 0.001596 | 0.006955 | 0.071183 | 0.169796 |
| Stage 2 `checkpoint-2000` | 新数据 test | 30949 | 0.000950 | 0.004407 | 0.024597 | 0.089788 |

从结果看，Stage 2 模型在新增数据测试集上的误差明显低于旧数据测试集，说明第二阶段专项微调已经有效适配新增数据分布。分维度 lag correlation 检查中，大部分维度最佳 lag 为 `0`，暂未发现明显的整体时序错位问题。

Stage 1 `checkpoint-6500` 也完成了一次新增数据测试集评估：

| checkpoint | 测试集 | 样本数 | stride | raw MSE | raw MAE | normalized MSE | normalized MAE |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Stage 1 `checkpoint-6500` | 新数据 test | 1000 | 10 | 0.001368 | 0.005395 | 0.035029 | 0.099045 |

需要注意：Stage 1 使用 `1000` 个样本、`stride=10`，Stage 2 使用全量样本、`stride=1`，两组评估口径不同，因此当前只能分别作为阶段性结果记录，不能直接计算严格的同比提升。后续需要用统一评估配置补齐 Stage 1 与 Stage 2 的对照实验。

## 五、真实推理服务接入

本周完成了 DM0 真实推理服务的接口梳理和客户端接入。当前服务通过以下接口提供推理：

```text
POST /process_frame
```

请求包含任务文本、三路相机图像和可选的 `16D state`。模型单次返回 `50 x 14` 的 action chunk，客户端按控制周期逐帧消费动作。

已补充以下接入能力：

- 新增 UI Agent adapter，可适配外部 `SampleAgent.act(obs, task)` 接口，并自动缓存 action chunk。
- 新增 XLeRobot 异步 bridge，支持后台预取、周期性重规划、队列 soft-replace、动作平滑和空队列 fallback。
- 新增模拟在线推理脚本，可从 DexData 逐帧回放观测，调用真实 HTTP 服务并统计延迟、队列状态和动作结果。
- 新增静态 HTML 报告工具，用于可视化在线回放中的动作维度和 action queue 变化。

当前推理服务链路已经具备从真实观测上传、HTTP 推理、动作 chunk 缓存到机器人侧逐帧消费的基本能力。后续重点是结合真实机械臂 rollout 继续验证动作语义、控制频率和异步队列策略。

## 六、工程补充

为支撑本轮训练和评估，还完成了以下工程改动：

- 增加新增数据 zip 合并转换工具和旧数据转换准备脚本。
- 增加视频无 B-frame 转码工具，并生成可追溯 manifest。
- 增加 `post_sort` 数据集注册文件和训练前检查脚本。
- 增加 Stage 0 norm stats、Stage 1 混合训练、Stage 2 新数据慢速训练脚本。
- 增加 Stage 1 / Stage 2 open-loop 评估脚本，支持 old/new split 分别评估。
- 增强 open-loop 工具，支持 stride、最大样本数、最大 episode 数、外部 norm stats、单卡推理和结果数组保存。
- 增加分维度误差和 lag correlation 调试工具，便于定位时序错位及局部动作维度问题。

## 七、下周计划

- 使用统一 `stride=1` 和全量样本配置补齐 Stage 1 / Stage 2 对照评估。
- 基于 `checkpoint-2000` 开展真实机械臂 rollout，验证新增数据微调后的实际控制效果。
- 根据真实推理延迟调优异步 bridge 的重规划周期、预取阈值、soft-replace 和 blend 参数。
- 继续推进 Stage 2 续训，并结合 open-loop 与真实 rollout 结果选择最终 checkpoint。
- 重点观察左右 pinch 维度的误差表现，结合执行器语义确认是否需要单独处理或补充数据。
