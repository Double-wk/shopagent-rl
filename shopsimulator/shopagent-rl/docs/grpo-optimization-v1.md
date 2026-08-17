# GRPO 优化 v1

> 历史诊断文档。2026-08-17 当前主结果已推进到 Paired-C1-hard step 200：Final-200
> strict 40%、completion 90%，说明 v1 的工程/信号问题后来已被绕过；但 price cf 仍为 0%。
> 本文的“下一步”不再是当前执行计划，现行门控见 `constraint-causal-experiment-plan.md`。

诊断对象:`outputs/grpo/v1/logs/grpo_env16_0812_shm_resume50_to200.log` 及同 resume 链前序 log，覆盖 step 1-200（实测 196 步，缺 47-50）。
基线权重:`outputs/grpo/v1/model/checkpoint_step_200/lora_adapter`。
轨迹级统计来自训练 log 内 `[ShopsimAgentLoop] summary` 行,共 1600 条(step 51-200 区间)。

## 结论

这 200 步没有产生学习效果。score 首段(1-25)0.623,末段(176-200)0.612,峰值 0.756 出现在 76-100 段之后回落,整体是噪声波动而非学习曲线。

动作格式层是健康的:每轮非法率 6.6%,turns 均值 4.78,96.6% 轨迹在轮次上限前正常终止。瓶颈不在这里。

瓶颈在**训练信号供给**:全程 51% 的 step 优势为 0、不产生任何梯度,且这个比例从第一段的 20% 单调恶化到末段的 60%。一半算力花在采样上,回传的梯度是零。

v1 只锁一个目标:**把零方差 step 压到 10% 以下**,让 score 曲线第一次出现可辨识的斜率。不追求 strict 指标提升。

## 我现在用的配置

| 项 | 值 |
|---|---|
| 模型 | Qwen3-1.7B-Base + LoRA,起点 `outputs/sft/v1/model/training_output/lora_adapter` |
| 训练数据 | `data/grpo_train.parquet`,dataset len 1000 |
| batch × group | `TRAIN_BATCH=4` × `ROLLOUT_N=4` = 16 rollouts/step |
| env pool | `SHOP_ENV_MAX_NUM=16` |
| 长度 | prompt 512 / response 8192 / max_model_len 10240 |
| 每轮输出上限 | `SHOPSIM_TURN_MAX_TOKENS=160` |
| obs 截断 | `SHOPSIM_OBS_MAX_CHARS=1800` |
| 采样 | temperature 0.7,top_p 1.0,top_k -1 |
| 学习率 | 1e-5 |
| GPU | 单卡 48G,`GPU_MEM_UTIL=0.25` |
| mini/micro batch | 4 / 1 |
| save_freq | 10 |

## 诊断

### 1. 一半的 step 没有梯度,且在恶化

按 25 步分段(零方差率 = `critic/advantages/max` ≈ 0 的 step 占比):

| 段 | score | 零方差率 | resp_len | entropy |
|---|---|---|---|---|
| 1-25 | 0.623 | 20% | 2796 | 0.352 |
| 26-50 | 0.697 | 48% | 3264 | 0.334 |
| 51-75 | 0.732 | 44% | 3276 | 0.318 |
| 76-100 | 0.756 | 64% | 3185 | 0.303 |
| 101-125 | 0.669 | 52% | 3385 | 0.301 |
| 126-150 | 0.733 | 60% | 3257 | 0.302 |
| 151-175 | 0.699 | 56% | 3271 | 0.288 |
| 176-200 | 0.612 | 60% | 3477 | 0.274 |

全程 99/196 步无梯度。GRPO 的优势在组内相对计算,组内同分则 advantage 恒为 0,该 step 的采样与前反向全部作废。

### 2. 奖励近二值,是零方差的根因

1600 条轨迹的 reward 分布:

- `reward = 0`:617 条(38.6%)
- `reward = 1.0`:814 条(50.9%)
- 落在中间:10.5%

组内视角(有 ≥4 条 rollout 的 245 个 task):**192 组组内完全同分,占 78.4%**。其中 108 组全 0(整组失败)、74 组全满分(整组成功)。

近二值奖励 + group size 4,组内同分是高概率事件。这是零方差的直接来源,也是恶化趋势的解释:策略越确定,组内越同分,信号越少。

### 3. 截断轨迹是纯废票,且发假负信号

| 类别 | 条数 | reward 均值 | 满分率 |
|---|---|---|---|
| 撞 8192 response 上限 | 129(8.1%) | 0.000 | 0.0% |
| 未撞上限 | 1471(91.9%) | 0.636 | 55.3% |

撞上限的 129 条 reward 恒为 0,没有一条例外。这不只是浪费:动作序列可能是对的,只是没走完就被判 0,模型收到的是**假负信号**。

长度构成:resp_len 均值约 3300-3560 token,turns 均值 4.78,即每轮约 745 token。模型每轮输出上限是 160,所以每轮约 585 token 是 observation。**大头是 obs 不是模型输出**,`OBS_MAX_CHARS=1800` 是可动的杠杆。

### 4. entropy 单调坍缩

0.386(step 1)→ 0.249(step 195),分段均值 0.352 → 0.274,全程单调下行。score 没涨而 entropy 一直掉,说明策略在收敛到一个平庸解,同时丢失探索能力——这会持续推高零方差率,构成正反馈。

### 5. 训练与评测口径错位

训练 reward 稳定在 0.65-0.7,而 strict 口径的 eval 数字远低于此。训练在优化一个已经半饱和的宽松指标,strict 那一端没有被推动。失败模式分析中 r_option 是第一损失源。

动作构成:click 70.4% / search 29.6%。含非法动作的轨迹占 29.6%(但摊到每轮只有 6.6%),说明非法动作是零散出现而非成片崩坏。

## 硬约束:env pool = 16

`TRAIN_BATCH × ROLLOUT_N ≤ SHOP_ENV_MAX_NUM = 16`。这条不等式决定了哪些优化能立刻做、哪些被阻塞:

- 在 16 rollouts/step 预算内**重新分配** batch 与 group:立刻可做,零额外成本。
- 需要超采样再丢弃(动态过滤)的方案:需要更大的 env pool,而 pack_api 冷启动正在被外部 SIGKILL(见 `run/pack_api.log`,服务成功监听并 serve 后被干净杀掉,非 OOM)。**扩 pool 依赖 env 稳定性先解决**。

## 优化项

### P0-1 组内规模 4 → 8(同预算内重分配)

改 `TRAIN_BATCH=2`、`ROLLOUT_N=8`,仍是 16 rollouts/step,env pool 不变。

对成功率 p 的 task,组内全同分概率是 p^n + (1-p)^n。n 从 4 到 8,p=0.5 时该概率由 0.125 降到 0.008。这是最便宜的一刀,直接检验零方差假设。

代价:每 step 只覆盖 2 个 distinct task,task 采样噪声上升。`PPO_MINI_BATCH` 需能整除 `TRAIN_BATCH`。

### P0-2 奖励稠密化

给未完成购买的轨迹按检索进展/候选商品匹配度发部分分,利用已有的 4 维分量(r_type / r_att / r_option / r_price)。

针对的是 108 个全 0 组——这部分 p ≈ 0,**加大 group size 救不了**,只能靠奖励本身有梯度。P0-1 和 P0-2 打的是零方差的两半,分开做看不出效果,建议一起上。

形式上走 potential-based shaping:不改变最优策略,但在有限算力下能提高实测成功率。

### P0-3 截断轨迹止损

129 条撞上限的轨迹当前恒判 0。两条路选一:排除出训练(不发假信号),或按已完成部分给部分分。配合下调 `OBS_MAX_CHARS`(当前 1800)让同样 8192 装更多轮,比加长上下文便宜——单卡 48G、`GPU_MEM_UTIL=0.25` 下没有加长空间。

### P1-1 止住 entropy 坍缩

格式层已稳(每轮非法率 6.6%),temperature 0.7 可回调到 0.8-0.9,或加 entropy bonus。放在 P0 之后:在零梯度的基础上加探索只是加噪声。

### P1-2 口径对齐

训练侧改用 strict,或按失败模式分析加大 r_option 权重。这条改的是优化目标本身,要等 P0 让曲线动起来之后再动,否则无法归因。

### P2 扩 env pool + 动态过滤

超采样后丢弃零方差组(DAPO 式),把无梯度 step 压到接近 0。**被 pack_api SIGKILL 问题阻塞**,需先查清外部 kill 来源。

## v1 验收标准

| 指标 | 当前 | v1 目标 |
|---|---|---|
| 零方差 step 率 | 51%(末段 60%) | < 10% |
| 组内同分率 | 78.4% | < 30% |
| score 斜率 | 净负(0.623 → 0.612) | 可辨识正斜率 |
| 撞 8192 占比 | 8.1% | < 3% |
| entropy | 单调下行至 0.249 | 不再单调坍缩 |

strict 提升不列入 v1 验收——先要信号,再谈指标。

## 未验证 / 待确认

- `outputs/grpo/v1/evaluation/` 的 Final-200 已完成：strict success 8.5%（17/200）、`r_hard=0.1183`、完成率 32.5%。这证实了上面的结论：该 200-step 训练没有带来评测提升。
- 1600 条轨迹摘要取自 step 51-200 区间的 log 行,非全量 3136 条 rollout,统计为抽样口径。
- pack_api 被外部 SIGKILL 的来源未查清:log 显示服务成功监听并 serve 200 请求后被干净杀掉,cgroup `memory.max` 为 64GiB 但 `oom_kill = 0`,主机 1TB 内存不紧张,死时无 co-tenant 抢内存。排除 cgroup OOM,来源在容器之上,容器内不可见。
