# ShopSim GRPO Recipe

执行以 [`configs/grpo.yaml`](../configs/grpo.yaml) 和
[`run_grpo.sh`](../scripts/run_grpo.sh) 为准；本文件只记录决策和验收标准。

训练 log 的诊断结论与优化项见 [`grpo-optimization-v1.md`](grpo-optimization-v1.md)。

## 目标

从 SFT LoRA adapter 继续训练多轮购物 Agent。每个 task 在线与 ShopEnv 交互，
终态 reward 进入 GRPO 组内相对优势，不需要 teacher 轨迹。

## 默认预算

| 参数 | shopagent-rl 默认值 | 说明 |
|---|---:|---|
| task batch | 4 | 每步 4 个不同 task |
| group size `G` | 4 | 每个 task 4 条 rollout；16 条/步，适配 16-env 池 |
| steps | 50 | 先 1 step smoke，再扩大 |
| LoRA | r=32, alpha=64 | 续训 `sft/v1/model/training_output/lora_adapter` |
| lr | 1e-5 | LoRA 续训起点 |
| KL | 0.001 | 先保守稳定，后续可做无 KL 消融 |
| precision | actor / rollout 均 BF16 | actor `model_dtype=bf16`，避免单卡先 FP32 物化基座 |
| vLLM memory | `GPU_MEM_UTIL=0.25` | 0.20 无法为 `max_model_len=10240` 建立最小 KV cache |
| dataloader workers | 0 | 1000 行数据无需 fork；避免 Ray/FSDP 下默认 8 workers 造成宿主内存峰值 |
| output | `/overlay/shopagent_rl_grpo_outputs/grpo` | 避免项目盘写满 |

`TRAIN_BATCH=4` 时，50 steps 只消费约 200 个 task row，适合作为行为诊断；要让
1000 行 GRPO prompt 池完成约一遍覆盖，使用 `TOTAL_STEPS=250`。`ROLLOUT_N=4`
意味着每个 step 生成 16 条 trajectory（4 个 task × 4 条），不是把 task 数改成 8。

## 长度设计

| 参数 | 值 | 作用 |
|---|---:|---|
| `max_prompt_length` | 512 | 初始 task/system prompt 上限；当前样本约 386–419 token |
| `SHOPSIM_TURN_MAX_TOKENS` | 160 | 单次 assistant→env 动作生成上限；SFT turn P99 为 109 token |
| `response_length` | 8192 | 整条多轮 response 区域，含 masked observation |
| `MAX_MODEL_LEN` | 10240 | vLLM 请求硬上下文上限；与 8192 response 及 512/轮相容 |
| `SHOPSIM_OBS_MAX_CHARS` | 1800 | 每轮 observation 压缩，控制历史增长 |

不要把总 `response_length` 改成 512：veRL 将 observation 也放在 response 区域，搜索
后的页面会消耗预算。需要缩短生成时改 `SHOPSIM_TURN_MAX_TOKENS`。

## Reward

终态 shaped reward 使用环境 `reward_detail`：

```text
r_type=.20, r_att=.30, r_option=.30, r_price=.20
```

验收时同时看合法动作、终态 reward 和组内方差；只看 `tool_calls` 不足以判断动作是否执行。

## 验收门槛

1. `TRAIN_BATCH=4`, `ROLLOUT_N=4` 能完成 1 step。
2. 至少一条 rollout 得到非零 reward，且同组 reward 不全相同。
3. `actor/pg_loss` 非零，checkpoint 写入 overlay。
4. `response_length/clip_ratio` 不长期为 1；若持续截断，先修动作停止/observation 压缩。

通过后再运行默认 50 steps；不要先扩大 G、prompt 池或训练步数。

## ROCm 单卡显存修复（vLLM 0.16.0）

2026-08-12 的同参数续跑从 `global_step_30` 到 step 46，step 47 在 actor
backward 发生 HIP OOM。诊断显示 Actor 的 PyTorch allocator 约占 5.9 GiB，
但整卡只余 170 MiB；整卡占用在 step 40/42/46 附近分别约为
40.7/46.3/50.2 GB。Actor 峰值保持在 7.5–8.4 GiB，因此增长来自同卡
vLLM/ROCm sleep 后未归还的物理显存，而不是模型或 batch 随 step 变大。

原始固定环境是 vLLM `0.16.0+rocm721`（上游基线
`89a77b10846fd96273cce78d86d2556ea582d26e`）。当前机器已应用修复并重编译为
`0.16.1.dev1+gd12b7df63`；新机器仍应从下述原始基线按补丁复现。该基线早于上游 ROCm
sleep 修复 `10a1018c127ac34ad0f255ae9fffdc452d0cf4d7`：旧实现执行
`hipMemUnmap` / `hipMemRelease` 后仍保留虚拟地址，物理 VRAM 可能不回到
空闲池。项目将最小回移保存为
[`patches/vllm-0.16.0-rocm-sleep-release.patch`](../patches/vllm-0.16.0-rocm-sleep-release.patch)，
不要依赖机器上 `/overlay/vllm-rocm-src` 的未记录状态。

在固定 vLLM 源码基线上应用并重新编译：

```bash
VLLM_SRC=/overlay/vllm-rocm-src
git -C "$VLLM_SRC" apply --check \
  /workspace/shopsimulator/shopagent-rl/patches/vllm-0.16.0-rocm-sleep-release.patch
git -C "$VLLM_SRC" apply \
  /workspace/shopsimulator/shopagent-rl/patches/vllm-0.16.0-rocm-sleep-release.patch

cd "$VLLM_SRC"
export TRITON_KERNELS_SRC_DIR=/overlay/triton-kernels-src/python/triton_kernels
MAX_JOBS=8 /overlay/miniconda3/envs/shop-A/bin/pip install \
  --no-build-isolation --no-deps -e .
```

若 `git apply --check` 报补丁已经应用，先用 `git log`/`git diff` 确认，不要
重复应用。重新编译后必须做 sleep/wake 显存回归，再恢复正式训练。
`--no-deps` 是必须的：当前 ROCm 环境已锁定验证过的依赖，让 pip 重新
解析 dev 版 vLLM 可能安装 CUDA CuPy 或替换 numpy/numba 等包。
编译时还应设置
`TRITON_KERNELS_SRC_DIR=/overlay/triton-kernels-src/python/triton_kernels`，避免
CMake 在离线/证书受限环境重新拉取 Triton 源码。

物理显存回归验收：

```bash
cd /workspace
source shopsimulator/shopagent-rl/scripts/vllm_env_shopA.sh
"$PY" shopsimulator/shopagent-rl/scripts/vllm_sleep_smoke.py
```

连续三次 sleep/wake 必须以 `SLEEP_WAKE_RESULT=PASS` 结束。

### AMD-SMI 总显存与进程显存为什么对不上

`amd-smi` 顶部的 `Mem-Usage` 是 DRM 驱动报告的整卡物理 VRAM 总量，下面的
进程表不是该总量的可加和分项。Ray/vLLM 使用多进程、HIP IPC 和 VMM 映射时，
同一批物理页可能被多个进程共享映射；进程表可能漏记共享/VMM 页，Linux
`/proc/<pid>/fdinfo` 又可能在多个进程中重复计算这些映射。因此不要用
“整卡总量减进程行之和”直接判断泄漏。

本次修复后 step 41–44 的 actor allocator 峰值均为 7.510 GiB；判断 sleep
修复是否有效，应比较相同阶段跨多个 step 的整卡 `mem_info_vram_used` 是否
阶梯式增长，并结合上述 sleep/wake smoke，而不是只看某一时刻进程表。

另一个 OOM 放大因素位于 veRL 的 fused PPO backward：Qwen3 的冻结 LM head
形状为 `[151936, 2048]`，旧实现仍无条件计算 BF16 `dvocab_weights`，额外申请
593.5 MiB，正好对应 step 47 的 594 MiB 失败申请。当前代码按
`ctx.needs_input_grad` 跳过冻结权重梯度；这不改变 LoRA/GRPO 数学结果。

恢复训练必须使用最后一个完整 checkpoint。step 41–46 没有 checkpoint，因此
从 `global_step_40` 恢复；batch、group size、mini/micro batch、长度、学习率和
其他 GRPO 参数全部保持不变。进入 200/250-step 长跑前，至少验证 20–30 个
连续 step 的 actor update 前整卡显存不再阶梯式上涨。

Hydra 自动生成的 `outputs/YYYY-MM-DD/HH-MM-SS/` 只保存最终展开的配置和主日志，
不能用于续训。可续训状态由 `trainer.default_local_dir` 指定；当前为
`/overlay/shopagent_rl_grpo_outputs/grpo/global_step_*`，包含模型、优化器、随机数
和调度器状态。换电脑时除了 Git 仓库，还必须单独复制所需的完整 checkpoint；
PID 文件仅供审计，在新机器上不能复用。

当前 40→50 显存修复验收结束后，使用
[`scripts/resume_grpo_50_to_200.sh`](../scripts/resume_grpo_50_to_200.sh) 自动校验
`global_step_50` 的模型、优化器和训练状态文件，再同参数续跑到 step 200。
脚本仅改变 `trainer.total_training_steps`，固定保持 batch/group、mini/micro batch、
长度、显存预算和 `1e-5` 学习率不变；当前 scheduler 为 constant，因此延长终点
不会改变 step 1–50 已执行的学习率轨迹。

40→50 的真实训练进一步暴露了另一条 ROCm 特有路径：naive checkpoint engine
每步新建 `update_weights_bucket_megabytes=2048` 的 HIP IPC 通信桶。即使 sender
和 receiver 都调用 `ipc_collect()`，跨进程 VMM 映射仍可能保留到进程退出，
表现为整卡 VRAM 约每步增加 2–3 GiB；进程退出后会立即降回约 26 MiB。
这与 Actor allocator（各步稳定为 7.510 GiB）无关。

当前 `run_grpo.sh` 在 ROCm 上默认令 `VERL_VLLM_WEIGHT_TRANSFER=auto` 选择 POSIX
shared memory 传输，绕过 HIP IPC/VMM；同步的 tensor 数值和训练参数不变，只可能
增加少量权重传输时间。`/dev/shm` 必须至少容纳 2 GiB 通信桶。诊断时可显式设置
`VERL_VLLM_WEIGHT_TRANSFER=ipc` 恢复旧路径，但不应用于长跑。进入 200-step
长跑前，先从 step 50 到 60 验证十个 step 的整卡显存不再阶梯式增长。
