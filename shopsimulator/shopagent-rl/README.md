# shopagent-rl

ShopSimulator 中文购物 Agent 的完整后训练工程：环境适配、teacher 数据、LoRA SFT、veRL/vLLM 多轮 GRPO，以及 Final-200 评测。基座模型是 `Qwen/Qwen3-1.7B-Base`。

## 当前状态

Base、SFT v1、Paired SFT v2、GRPO v1、GRPO v2b、GRPO C1-hard 与 GRPO Paired-C1-hard 的 Final-200 评测均已完成（同一 10 步协议、每轮 512 token）；step20 结果仅作历史归档对照。当前最好是 **GRPO Paired-C1-hard：strict 40%**。

| 模型 | 完成率 | **strict success (Rsucc)** | r_hard | r_loose | 选对商品率 | 报告文件 |
|---|---:|---:|---:|---:|---:|---|
| Base v1 | 0% | **0%** (0/200) | 0 | 0 | 0% | `outputs/base/v1/evaluation/` |
| SFT v1 | 39.5% | **17%** (34/200) | 0.201 | 0.306 | 25.5% | `outputs/sft/v1/evaluation/` |
| **SFT v2 Paired Constraint** | **52.0%** | **25%** (50/200) | **0.278** | **0.392** | 27.5% | `outputs/sft/v2_paired/evaluation/` |
| GRPO v1（env16, step200） | 32.5% | **8.5%** (17/200) | 0.118 | 0.238 | 19.0% | `outputs/grpo/v1/evaluation/` |
| GRPO v2b（env32, n=8, step200） | 68.0% | **18%** (36/200) | 0.215 | 0.435 | 27.5% | `outputs/grpo/v2/evaluation/` |
| GRPO C1-hard（hard 预算惩罚, step200） | 53.5% | **20%** (40/200) | 0.228 | 0.385 | 28.0% | `outputs/grpo/c1_hard/evaluation/` |
| **GRPO Paired-C1-hard（paired init + hard 惩罚, step200）** | **90.0%** | **40%** (80/200) | **0.425** | **0.644** | **45.0%** | `outputs/grpo/paired_c1_hard/evaluation/` |
| GRPO step20（历史归档） | 33.5% | **18%** (36/200) | 0.197 | 0.270 | 24.0% | `archive/grpo_global_step_20_2026-08-10/` |

> ⚠️ **GRPO v2b 的行为指标明显改善，但 headline strict 提升尚未验证**：相对 SFT，完成率
> +28.5pt、r_loose +0.129、r_hard +0.014、选对商品率 +2pt；但 strict 只 +1pt（34→36 题）。
> 环境价格随机化本身会让单任务 strict 跨 run 抖动（见 `INTERVIEW.md` §2 深挖），n=200 上
> 2 题的差落在噪声内。当前结论是「v2b 显著改善完成行为与密集奖励，严格成功率仍需重复评测验证」。

> ⚠️ **C1-hard（超预算购买奖励归零）未修复价格盲视**：反事实探测中 145 对超预算场景
> 144/145 仍照常购买；hard 惩罚只把购买率从 68% 压到 53.5%，strict 20% 与 v2b 的 18%
> 在噪声内。判定与机制分析见 [`docs/counterfactual-eval.md`](docs/counterfactual-eval.md)。

> ✅ **Paired-C1-hard（SFT v2 paired 初始化 + hard 预算惩罚）strict 40%，全线最好**：
> 完成率 90%、r_hard 0.425、选对商品率 45%，较 SFT v2 +15pt（80 vs 50 题，远超单 run
> 抖动）。option-swap paired robust 维持 73.1%（C1-hard 从 base 起训时只有 29.9%）；
> 但价格反事实仍为 0%（145 对超预算场景 143/145 两边同样照买），probe original 侧动作
> 准确率 93.9%——预算内选品很准，只是不把价格比较纳入购买决策。判定见
> [`docs/counterfactual-eval.md`](docs/counterfactual-eval.md)。

| 阶段 | 当前结果 |
|---|---|
| Constraint-causal Gate 2 | **已通过**：Paired SFT v2 的 natural option-swap paired robust `73.1%`（49/67），Final-200 strict `25%`（50/200）；进入 Certified GRPO 前置阶段。价格反事实 paired robust 仍为 `0%`。 |
| GRPO smoke | 已完成 1 step：reward/pg_loss 非零，checkpoint 写入 overlay |
| GRPO 正式训练 | env16 固定配方 `TRAIN_BATCH=4 / ROLLOUT_N=4 / PPO_MINI_BATCH=4` 已完成 **200 steps**；最终可恢复 checkpoint 为 `outputs/grpo/v1/model/checkpoint_step_200/` |
| GRPO v2b | `TRAIN_BATCH=4 / ROLLOUT_N=8 / env32` 已完成 **200 steps**；导出 adapter 为 `/overlay/shopagent_rl_grpo_outputs/grpo/v2/export_step_200/lora_adapter/` |
| GRPO Paired-C1-hard | SFT v2 paired 起训 + hard 预算惩罚（b4/n4/lr1e-5）已完成 **200 steps**；Final-200 strict **40%**（80/200，历史最好），adapter 在 `/overlay/shopagent_rl_grpo_outputs/grpo/paired_c1hard_200_direct/export_step_200/lora_adapter/` |
| 训练期 reward | 早期 reward 波动较大，当前步数仍不足以判断收敛；进入 200/250-step 长跑前先完成 20–30 个连续 step 的显存稳定性验收 |
| OOM 根因 | 旧版 vLLM/ROCm sleep 没有稳定归还物理 VRAM，整卡占用逐 step 增长；fused PPO backward 又为冻结 LM head 无效申请 593.5MiB 梯度。两处均已修复，不需要改变训练参数 |

最终 FSDP checkpoint 与可移植的 LoRA adapter 已归入
[`outputs/grpo/v1/model/checkpoint_step_200/`](outputs/grpo/v1/model/checkpoint_step_200/)；中间 checkpoint 仍保留在 `/overlay`。

## 评测指标口径

三个分数都由官方 [`get_score.py`](../ShopSimulator/get_score.py) 从存盘的 `reward_detail` 重算，
同一批轨迹算出三种严格度。四个维度是 `r_type`（品类，取 1.0 或沾边 0.5）、`r_att`（属性命中比例）、
`r_option`（规格命中比例）、`r_price`（价格 ≤ 预算，0/1）。

| 指标 | 定义 | 严格度 | 用途 |
|---|---|---|---|
| **r_loose** | 环境自己算的 `total_reward`，即 `(属性命中数 + 规格命中数 + r_price) / (属性数 + 规格数 + 1) × r_type`（`goal.py:224`）；评测时直接取存盘的 `reward` 字段 | 最宽容：**加权平均**，买错也能拿零点几分 | RL 训练的密集信号 |
| **r_hard** | `r_type × r_att × r_option × r_price` | 中：**乘法瓶颈**，任一维为 0 则全 0 | = 官方 Rstrict |
| **r_success** | 四维**全部 == 1** 记 1，否则 0（等价于 `total_reward == 1.0`） | 最严：0/1 | = 官方 Rsucc = **「严格成功率」，对外报的就是这个** |

没走到 `done`（没买）的任务，三个分数都是 0，同时拉低完成率。所以 r_loose 下降既可能是"买得少了"，
也可能是"买得同样多但买得更不准"——看它必须和完成率一起看。

## 目录约定

```text
configs/                  唯一配置目录：SFT、teacher、GRPO、AgentLoop 注册
data/                     唯一数据目录：任务划分、teacher raw、SFT jsonl、GRPO parquet
scripts/                  唯一运行入口：采样、构建、SFT、GRPO、评测、vLLM 环境
shop_env/                 ShopSimulator HTTP client/wrapper/observation/reward
experiment/teacher/       teacher 采集与质量校验
experiment/sft/           LoRA SFT 训练
experiment/grpo/          veRL 多轮 AgentLoop
experiment/eval/          Final-200 批量评测与本地 vLLM
experiment/counterfactual_pairs.py  原子规格/价格反事实对生成与校验
outputs/                  当前 SFT 与评测报告；大型 GRPO 输出在 overlay
verl/                     项目内固定版本的 veRL
archive/                  历史配置、旧入口、日志、checkpoint，不参与运行
docs/                     GRPO 配方与工程说明
```

每类文件只有一个工作位置：不要在 `experiment/` 下再创建 configs、scripts、data 或 outputs。

## AMD GPU 开源使用指南

首次使用 AMD GPU、需要从账号注册/实例申请开始配置 ROCm 的用户，请阅读
[`docs/amd-gpu-quickstart.md`](docs/amd-gpu-quickstart.md)。该文档覆盖 AMD 开发者平台注册入口、实例规格选择、ROCm 验收、vLLM HIP 扩展 smoke，以及本项目的环境、评测与 GRPO 启动流程。

## 安装项目内 veRL

`verl/` 是项目固定的 veRL 0.8.0 源码包。它不是一个独立 clone，因此上游仓库根目录的打包文件没有被带入；本项目在
[`pyproject.toml`](pyproject.toml) 提供标准的 editable 安装定义。完成 ROCm/vLLM 与 `requirements.txt` 的安装后，在
`shopagent-rl/` 目录执行一次：

```bash
/overlay/miniconda3/envs/shop-A/bin/python -m pip install --no-deps -e .
```

之后无论当前工作目录是什么，`import verl` 都会解析到 `shopagent-rl/verl/`。运行脚本仍设置 `PYTHONPATH`，但那是为了让 Ray worker
找到 ShopSim 项目模块，以及让 vLLM 找到 ROCm 的 `amdsmi` / `functorch` shim，不再是安装 veRL 的前提。

## 配置与长度

唯一运行配置是 [`configs/grpo.yaml`](configs/grpo.yaml)，唯一 SFT 配置是 [`configs/sft.yaml`](configs/sft.yaml)。GRPO 入口 [`scripts/run_grpo.sh`](scripts/run_grpo.sh) 提供运行时覆盖。

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `TRAIN_BATCH` | 4 | 每步不同 task 数 |
| `ROLLOUT_N` | 4 | 每个 task 的 GRPO group size |
| `MAX_PROMPT_LENGTH` | 512 | 初始 task/system prompt 上限 |
| `SHOPSIM_TURN_MAX_TOKENS` | 160 | 每一轮 assistant 动作生成上限；SFT turn P99 为 109 token，实际边界由 `<|im_end|>` stop 决定 |
| `RESPONSE_LENGTH` | 8192 | 整条多轮 response 区域，包含 observation |
| `MAX_MODEL_LEN` | 10240 | vLLM 单请求上下文硬上限 |
| `GPU_MEM_UTIL` | 0.25 | vLLM KV cache 预留比例；0.20 无法容纳一个 10,240-token 请求 |
| `SHOPSIM_OBS_MAX_CHARS` | 1800 | 每轮 observation 字符上限 |

Actor FSDP 与 vLLM rollout 都固定为 BF16；FSDP 的 `model_dtype=bf16` 避免先以 FP32 物化 1.7B 基座。`response_length` 不能误设为 512：veRL 会把 assistant 和环境 observation 一起放入 response 区域，512 只适合单轮生成。`MAX_MODEL_LEN` 是上限，不会按 16K 为每个实际 3K 轨迹计算推理量，但会影响 KV cache 预留和并发。

> **ROCm/vLLM 0.16.0 长跑前置修复：** 2026-08-12 的同参数训练在 step 47
> 暴露了 ROCm sleep 未归还物理 VRAM 的问题，整卡占用随 step 从约 40.7GB
> 爬升到 50.2GB，而 Actor 峰值保持约 8GB。项目保存了上游最小回移补丁
> [`patches/vllm-0.16.0-rocm-sleep-release.patch`](patches/vllm-0.16.0-rocm-sleep-release.patch)，
> 同时 fused PPO backward 已跳过 LoRA 冻结 LM head 的无效 593.5MiB 梯度。
> 安装、验证与 `global_step_40` 恢复规则见 [`docs/grpo.md`](docs/grpo.md)。这些
> 修复不改变 batch、rollout、长度、学习率或 GRPO 数学语义。

GRPO 数据集只有 1000 行，`dataloader_num_workers=0` 是有意设置：默认的 8 个 StatefulDataLoader 子进程会在 Ray/FSDP 父进程中复制大量状态，已实测造成约 94GB CPU 内存峰值并在退出阶段被系统杀掉；训练的瓶颈是 rollout，不是读取 parquet。

## 运行顺序

```bash
# 0. 【全新 clone 必做】还原压缩入库的大文件（SFT/GRPO adapter 权重、
#    env 商品库 items_eval_train.json、BM25 Lucene 索引）。
#    这些文件超过 GitHub 100MB 硬限或被权重规则排除，仓库里存的是 .gz/分卷。
#    脚本带 sha256 校验，已还原过的会自动跳过。
bash ../../scripts/restore_large_artifacts.sh

# 1. 启动环境服务
bash scripts/start_pack_api.sh

# 2. 构建/检查 SFT 数据（已有数据时可跳过）
bash scripts/build_sft_data.sh

# 3. 训练 SFT adapter
bash scripts/02_sft.sh

# 4. 构建 GRPO parquet（数据已存在时可跳过）
python scripts/build_grpo_data.py

# 5. 1 step smoke
FOREGROUND=1 TOTAL_STEPS=1 SAVE_FREQ=1 \
OUTPUT_DIR=/overlay/shopagent_rl_grpo_outputs/grpo_smoke \
bash scripts/run_grpo.sh

# 6. 第一阶段：50 steps 行为诊断（4 task/step，约 200 个 task draw）
RUN_NAME=grpo_diagnostic OUTPUT_DIR=/overlay/shopagent_rl_grpo_outputs/grpo_diagnostic \
TOTAL_STEPS=50 bash scripts/run_grpo.sh

# 7. 完整覆盖：约 250 steps（1000 task / TRAIN_BATCH=4）
RUN_NAME=grpo_full OUTPUT_DIR=/overlay/shopagent_rl_grpo_outputs/grpo_full \
TOTAL_STEPS=250 bash scripts/run_grpo.sh

# 查看后台训练
tail -f run/grpo_full.log

# 8. Final-200 评测（将 --adapter 替换为训练完成的 adapter 路径）
bash scripts/run_eval.sh --tag GRPO --out outputs/eval_grpo.jsonl \
  --adapter /overlay/shopagent_rl_grpo_outputs/grpo_full/<adapter-path>

# 9. SFT 基线评测（已有结果时可跳过）
bash scripts/run_eval.sh --tag SFT --out outputs/eval_sft.jsonl \
  --adapter outputs/sft/v1/model/training_output/lora_adapter

# 在一张 48GB 卡上顺序运行 SFT 与 GRPO 的可比 Final-200 评测
# （两个 vLLM 引擎不能同时驻留显存）
bash scripts/eval_sft_grpo_serial.sh

# 10. CPU-only：构造 Final-200 原子约束反事实评测对（不启动模型/环境）
python scripts/build_counterfactual_pairs.py
```

反事实对的干预定义、有效性门和第一阶段指标见
[`docs/counterfactual-eval.md`](docs/counterfactual-eval.md)。它是独立诊断集，不修改 Final-200
成功率口径，也不参与当前 GRPO 训练。

## 优化顺序

1. 先保证 rollout 有合法 action、同组 reward 有方差、`actor/pg_loss` 非零，且 `response_length/clip_ratio` 不长期为 1。
2. 再优化 observation：保留指令、当前页面、价格和可点击项，压缩重复页面；必要时将 `SHOPSIM_OBS_MAX_CHARS` 从 1800 分阶段下调。
3. 再处理策略行为：对无 action、重复 click、步数耗尽增加显式诊断，避免用长 response 掩盖停止问题。
4. 最后扩大训练步数：默认 `TRAIN_BATCH=4` 时，50 / 250 steps 分别约对应 200 / 1000 个 task draw。保持已验证配方不变，先确认修复后的 vLLM sleep/wake 回归通过，并观察至少 20–30 个连续 step 的整卡显存不再阶梯式上涨；不要靠缩短 response、降低 `GPU_MEM_UTIL` 或改 PPO batch 来续跑旧 checkpoint。

上述第 1 条在 step 1-200 上的实测诊断、优化项与验收标准见
[`docs/grpo-optimization-v1.md`](docs/grpo-optimization-v1.md)。

数据划分、规模和互斥校验见 [`DATA.md`](DATA.md)；GRPO 验收标准见 [`docs/grpo.md`](docs/grpo.md)。
