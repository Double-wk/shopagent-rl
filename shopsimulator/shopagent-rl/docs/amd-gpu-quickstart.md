# AMD GPU 平台与本项目快速开始

本指南面向首次使用 AMD GPU 的开源用户，说明如何注册 AMD 开发者平台、获得一台可 SSH 的 AMD GPU Linux 实例，并配置到能运行本项目的程度。

> 本项目验证过的环境为：单卡 48GB、`gfx1100`、ROCm 7.2.1、Ubuntu Linux、Python 3.12。不同平台提供的 GPU、镜像、配额和申请入口会变化；请以平台控制台显示的实际资源为准。

## 0. 你需要准备什么

- 一个可接收验证码的邮箱或手机号；
- 一台带 AMD GPU、可通过 SSH 登录的 Linux 实例，建议至少 **48GB VRAM、64GB RAM、100GB 可用磁盘**；
- 本项目及其依赖模型的下载网络；
- 若运行完整 GRPO：建议预留 12 小时以上的独占单卡时间。当前安全配方约 9–10 分钟/step，50 step 不是短任务。

AMD GPU 并不等于任意 AMD 显卡都能跑 ROCm。下单/申请前务必先查 AMD 官方的 [ROCm Linux 支持矩阵](https://rocm.docs.amd.com/projects/install-on-linux/en/develop/reference/system-requirements.html)。例如 Radeon RX 7900 XTX/XT/GRE 属于 `gfx1100`，在当前 ROCm 文档中受支持。

## 1. 注册 AMD 开发者平台并获得实例

1. 打开 [AMD 开发者平台登录/注册页](https://developer.amd.com.cn/login?source=8EHcpUKwn)。
2. 选择注册，按页面要求完成邮箱或手机号验证、账号信息与必要的实名认证/协议确认。
3. 登录后，在平台控制台查找 GPU 云实例、开发者资源、活动申请或算力权益入口；按当期页面提示提交申请或创建实例。
4. 获得实例后，记录以下信息：公网 IP/域名、SSH 用户名、SSH 私钥或密码、操作系统版本、GPU 型号、显存大小、磁盘容量、是否具有 `sudo` 权限。

注册账号**不保证**自动获得 GPU 配额或免费实例；具体资格、库存、价格和审批规则由 AMD 平台当期政策决定。本项目不收集你的平台账号、验证码、SSH 私钥或 API token。

## 2. 创建实例时的选择建议

优先选择：

| 项目 | 建议 | 原因 |
|---|---|---|
| GPU | 单卡 48GB、ROCm 官方支持型号 | 当前 GRPO 同时驻留 vLLM rollout 与 BF16 FSDP actor |
| 系统 | 平台提供的 ROCm 预装 Ubuntu 镜像 | 少走内核/驱动匹配问题 |
| RAM | ≥64GB，推荐 128GB | Ray、环境服务和 Python 进程需要较大主存 |
| 磁盘 | ≥100GB | Conda、ROCm、模型与 checkpoint 均占空间 |
| 网络 | 可访问 GitHub、Hugging Face、PyPI | 初次构建/下载需要；生产环境可改为镜像或离线缓存 |

若只有 24GB 显存，可做数据、SFT、小规模评测或缩短 response 的调试；不要直接照搬本项目的 48GB 单卡 GRPO 配方。

## 3. 首次 SSH 登录与硬件验收

从本地登录（将占位符替换为控制台给出的信息）：

```bash
ssh -i ~/.ssh/amd_gpu_key <user>@<host>
```

在实例上执行：

```bash
uname -a
df -h
rocm-smi || rocm-smi --showproductname
rocminfo | rg 'Name:|gfx'
```

验收标准：`rocm-smi` 能列出 GPU 与显存；`rocminfo` 能看到一个 `gfx*` 架构名。若其中任一命令不存在或无设备，请先在控制台确认你选的是 AMD GPU/ROCm 镜像，再按 AMD 官方 ROCm 安装文档处理驱动与运行时；不要先安装 CUDA 包。

多卡主机如出现 RCCL 的 `NUMA auto balancing enabled` 警告，需要在**宿主机**由管理员处理：

```bash
sudo sysctl -w kernel.numa_balancing=0
echo 'kernel.numa_balancing = 0' | sudo tee /etc/sysctl.d/99-rocm-rccl-numa.conf
sudo sysctl --system
```

容器内若报 `Read-only file system`，这不是项目错误，说明该设置只能由云实例宿主机管理员完成。

## 4. 获取代码与创建 Python 环境

以下示例假定你在工作目录下 clone 本仓库。若 AMD 平台已提供 Conda/ROCm 环境，优先复用；否则按仓库根目录的 [`服务器迁移指南.md`](../../../服务器迁移指南.md) 完成完整环境构建。

```bash
git clone <你的仓库地址> /workspace
cd /workspace

# 首先读取迁移指南；它锁定了本项目实际验证过的 ROCm、vLLM 和 Python 依赖版本。
sed -n '1,220p' 服务器迁移指南.md
```

当前已验证环境变量脚本不只是在设置 PATH：它还提供 ROCm 平台探测需要的兼容 shim。因此每次运行 vLLM、评测或 GRPO 前都要 source：

```bash
cd /workspace/shopsimulator/shopagent-rl
source scripts/vllm_env_shopA.sh
```

安装项目内固定的 veRL 源码包：

```bash
"$PY" -m pip install --no-deps -e .
"$PY" -c "import torch, verl; print(torch.cuda.is_available(), verl.__version__, verl.__file__)"
```

预期 `torch.cuda.is_available()` 为 `True`，而 `verl.__file__` 指向当前 clone 的 `shopagent-rl/verl/`。editable 安装的临时元数据会写到 `archive/build/`，已被 Git 忽略。

## 5. 验证 vLLM 的 HIP 扩展，而不仅是 import

`import vllm` 成功并不代表 HIP 扩展可用。先检查扩展，再做真实引擎 smoke：

```bash
cd /workspace
source shopsimulator/shopagent-rl/scripts/vllm_env_shopA.sh
"$PY" -c "import torch, vllm; print(torch.__version__, torch.cuda.is_available(), vllm.__version__)"
"$PY" -c "import vllm._C, vllm._rocm_C, vllm._moe_C; print('HIP extensions OK')"
"$PY" scripts/vllm_smoke.py
```

最后一条应以 `RESULT=PASS` 结束。若失败，先保存完整日志，再对照迁移指南的 vLLM commit、ROCm 版本和 `TRITON_KERNELS_SRC_DIR` 说明排查；不要混用 CUDA vLLM wheel 与 ROCm PyTorch。

## 6. 启动环境、评测与训练

环境服务必须先启动：

```bash
cd /workspace/shopsimulator/shopagent-rl
bash scripts/start_pack_api.sh
tail -f run/pack_api.log
```

看到 `Running on http://127.0.0.1:5000` 后，另开一个 shell 执行评测或训练。先做低风险评测：

```bash
source scripts/vllm_env_shopA.sh
bash scripts/run_eval.sh --tag SFT --out outputs/eval_sft.jsonl \
  --adapter outputs/sft/v1/model/training_output/lora_adapter
```

确认通过后再开始 GRPO：

```bash
RUN_NAME=grpo_diagnostic \
OUTPUT_DIR=/overlay/shopagent_rl_grpo_outputs/grpo_diagnostic \
TOTAL_STEPS=50 \
GPU_MEM_UTIL=0.25 \
bash scripts/run_grpo.sh
```

日志写入 `run/<RUN_NAME>.log`，大 checkpoint 应写到实例的大盘（示例为 `/overlay/...`），不要写满代码所在的小系统盘。

如果复现当前 Certified corrective 路线，先运行 SFT，再运行串行评测门；同一张卡上
不要同时启动 vLLM 评测、SFT 和 GRPO：

```bash
bash scripts/run_certified_corrective_sft.sh
bash scripts/chain_certified_corrective_eval.sh

# 仅在 natural heldout-v2 price >= 30% 且 Final-200 strict >= 16% 后人工执行
bash scripts/run_certified_grpo.sh
```

评测链不会自动启动 GRPO。`run/certified_corrective_*.log` 是运行日志，不应提交；
可复现数据、配置、指标 JSON 和构建脚本才是版本化对象。

## 7. 单卡显存的关键限制

当前推荐配方固定为 BF16：vLLM 的 KV cache 与 FSDP actor 的反向传播峰值会同时占用显存。请保持以下默认值，除非先完成一轮 smoke：

```bash
TRAIN_BATCH=4
ROLLOUT_N=4
PPO_MINI_BATCH=4
GPU_MEM_UTIL=0.25
MAX_MODEL_LEN=10240
RESPONSE_LENGTH=8192
SHOPSIM_TURN_MAX_TOKENS=160
```

- `GPU_MEM_UTIL=0.25` 控制 vLLM 初始化时模型与 KV cache 的预算，但不能限制旧版 ROCm sleep 未正确归还的 VMM 物理页或临时 workspace。vLLM 0.16.0 必须应用 [`patches/vllm-0.16.0-rocm-sleep-release.patch`](../patches/vllm-0.16.0-rocm-sleep-release.patch) 并重新编译；详见 [`docs/grpo.md`](grpo.md)。
- 若 actor 反向传播时 HIP OOM，先区分 Actor allocator 与整卡占用。Actor 仅占约 6–8 GiB 而整卡接近 48 GiB 时，应修 vLLM/ROCm 物理显存释放，不要通过修改 `PPO_MINI_BATCH`、response length 或其他训练参数续跑旧 checkpoint。
- `response_length=8192` 包含多轮 assistant 输出和环境 observation；每轮生成上限是 `SHOPSIM_TURN_MAX_TOKENS=160`，两者不是同一个参数。正常 turn 会在 `<|im_end|>` 处更早结束。

## 8. 提交 issue 时请附上这些信息

请勿提交密码、私钥、平台 token 或完整环境变量。最有用的信息是：

```bash
rocm-smi
rocminfo | rg 'Name:|gfx'
"$PY" -c "import torch,vllm,verl; print(torch.__version__, torch.cuda.is_available(), vllm.__version__, verl.__version__)"
tail -n 120 run/<你的任务名>.log
```

并写明 GPU 型号、显存、Linux 版本、ROCm 版本、是否容器，以及你执行的命令。
