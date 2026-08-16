# AMD 开发环境初始化指南

本文说明如何在 AMD 云开发环境中配置 SSH、网络代理、`cc-switch`、Claude Code 和 Codex CLI。按阶段执行即可；每一段命令都标明了用途、完成标志和注意事项。

## 目录说明

| 路径 | 用途 |
| --- | --- |
| `mihomo/` | mihomo 二进制、配置和运行日志 |
| `cc-switch/` | cc-switch CLI、包装脚本及项目独立配置 |
| `miniconda3/` | 项目内安装的 Miniconda 与 Conda base 环境 |
| `notebooks/` | Jupyter Notebook 工作文件 |

开始前请进入仓库根目录：

```bash
cd /workspace
```

> 安全提示：`mihomo/config/config.yaml` 和 `cc-switch/.cc-switch/` 可能含有代理凭据、订阅或 API Key。请只保存在本地，不要提交到 Git 或粘贴到公开文档。

### GitHub 上传认证（PAT）

后续可以使用 GitHub Personal Access Token（PAT）通过 Git 或 GitHub API 上传本项目。PAT 只应保存在当前 shell 的环境变量中，**不要写入 README、脚本、Git remote URL、命令历史或提交内容**。

创建 fine-grained PAT 时，仅授予目标仓库的 `Contents: Read and write` 权限；只有需要操作 Pull Request 时才额外授予 Pull requests 权限。建议设置较短的过期时间，并在任务完成后撤销或轮换 token。

```bash
export GITHUB_TOKEN='github_pat_<your-PAT-here>'   # 仅在当前 shell 设置；切勿写入本文件或提交

# GitHub API 示例
curl -fsS \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  -H 'Accept: application/vnd.github+json' \
  https://api.github.com/repos/Double-wk/amd

# Git 上传时不要把 token 写进 URL；使用 Git credential helper 或交互式认证。
git push origin main
```

本对话中曾粘贴过一个 PAT，该 token 已视为泄露，不能继续使用；请先在 GitHub 设置中撤销它并重新生成。

## 阶段 0：AMD 入口与配置来源

在执行任何命令前，先使用以下两个入口：

| 入口 | 用途 |
| --- | --- |
| [AMD 配置文件（飞书）](https://my.feishu.cn/wiki/WZjewQhHBi5qvwk0U1Sca8Qenfg?from=from_copylink) | 获取本环境的 AMD 配置说明、镜像或实例相关信息。访问可能需要飞书权限。 |
| [AMD 连接控制台](https://radeon-global.anruicloud.com/) | 登录云平台、进入或启动对应的开发环境，并取得终端、Jupyter 或 SSH 连接信息。 |

建议顺序：先在控制台确认实例已启动并进入终端，再参照飞书配置文件确认镜像和环境要求，最后继续本文的 SSH、代理和 CLI 配置。

## 阶段 1：启用 SSH 远程连接

**目的：** 某些基础镜像没有启动 SSH 服务。以下命令安装并在当前会话启动它，使终端可通过 SSH 连接。

```bash
sudo apt update
sudo apt install -y openssh-server
sudo service ssh start
```

| 命令 | 作用 |
| --- | --- |
| `apt update` | 刷新可安装软件包的索引。 |
| `apt install` | 安装 SSH 服务端。 |
| `service ssh start` | 在当前容器中启动 SSH 服务。 |

验证服务状态：

```bash
sudo service ssh status
```

若仍无法连接，还需检查云平台是否暴露了 SSH 端口，以及安全组规则是否允许访问。

### 1.1 为 AMD Cloud 实例追加登录公钥

AMD Cloud 服务器用 `~/.ssh/authorized_keys` 中的**公钥**验证本地电脑保存的私钥（例如 `~/.ssh/id_ed25519`）。

在**本地 MacBook** 查看需要添加的公钥：

```bash
cat ~/.ssh/id_ed25519.pub
```

在**AMD Cloud 服务器终端** 查看当前已授权的公钥：

```bash
cat ~/.ssh/authorized_keys
```

追加一把新的公钥时，可将本地 MacBook 执行 `cat ~/.ssh/id_ed25519.pub` 输出的完整一行写入服务器：

```bash
echo 'ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIKgDKGSu8fX9Q3lKwyp15isWhxTw4YDT3NxnjuXn2mx6 3315439213@example.com' >> ~/.ssh/authorized_keys
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
```

`>>` 是追加，不会影响已有 Key；不要使用单个 `>`，否则会覆盖 `authorized_keys` 并使原有登录密钥失效。一个 Key 占一行，可同时保留多把 Key。


## 阶段 2：安装并启动 mihomo 代理

**目的：** mihomo 在本机提供 HTTP 代理。本项目配置默认使用 `127.0.0.1:7890`，用于安装依赖和访问外部服务。

### 2.1 安装 mihomo

```bash
mkdir -p mihomo
wget --no-check-certificate -c --tries=20 --timeout=30 --waitretry=3 \
  -O mihomo-linux-amd64-v1.19.28.gz \
  https://gh-proxy.com/https://github.com/MetaCubeX/mihomo/releases/download/v1.19.28/mihomo-linux-amd64-v1.19.28.gz
gzip -d mihomo/mihomo-linux-amd64-v1.19.28.gz
mv mihomo/mihomo-linux-amd64-v1.19.28 mihomo/mihomo
chmod +x mihomo/mihomo
```

- `curl -L` 会跟随下载跳转；`-k` 会跳过 TLS 证书校验，只应在确有必要时使用。
- `gzip -d` 解压下载的发布包。
- `chmod +x` 赋予二进制执行权限。

### 2.2 准备配置并启动

将个人代理配置保存为 `mihomo/config/config.yaml`，然后执行：

```bash
cd mihomo
nohup ./mihomo -d ./config > mihomo.log 2>&1 &
```

`-d ./config` 指定配置目录；`nohup ... &` 使进程在终端关闭后继续运行；日志会写入 `mihomo.log`。

验证控制接口：

```bash
curl --noproxy '*' http://127.0.0.1:9090/version
```

`--noproxy '*'` 确保请求直接发送到本地控制端口，不会被代理环境变量再次转发。

### 2.3 为命令行程序设置代理

```bash
cat >> ~/.bashrc <<'EOF_BASHRC'
export http_proxy=http://127.0.0.1:7890
export https_proxy=http://127.0.0.1:7890
export HTTP_PROXY=http://127.0.0.1:7890
export HTTPS_PROXY=http://127.0.0.1:7890
EOF_BASHRC
source ~/.bashrc
```

小写和大写变量分别兼容不同命令行工具与运行时。`source ~/.bashrc` 让配置立即在当前终端生效；之后打开的新终端会自动加载。

### 2.4 切换代理节点（可选）

将 `<代理组>` 和 `<节点名>` 替换为 `config.yaml` 中的真实名称：

```bash
curl --noproxy '*' -sS -X PUT \
  'http://127.0.0.1:9090/proxies/%E8%8A%82%E7%82%B9%E9%80%89%E6%8B%A9' \
  -H 'Content-Type: application/json' \
  --data-raw '{"name":"新加坡SG-HY2"}'
```

这会通过 mihomo 控制接口，将指定策略组切换到对应节点。

## 阶段 3：安装 Miniconda（项目内）

**目的：** 在仓库内创建独立的 Conda Python 环境，避免依赖系统 Python 或其他项目的环境。当前安装路径为 `/overlay/miniconda3`（大容量 overlay 盘；3 个 conda 环境共约 21G，不适合放 `/workspace`）。

```bash
# 安装到大容量 overlay 盘（conda 环境约 21G，别放 /workspace）
wget --no-check-certificate \
  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /overlay/miniconda3
rm -f Miniconda3-latest-Linux-x86_64.sh
```

- `-b` 表示非交互式安装；`-p` 指定安装目录。
- `--no-check-certificate` 会跳过 TLS 证书校验，只应在当前代理/证书环境确有需要时使用。

加载 Conda：

```bash
source /overlay/miniconda3/etc/profile.d/conda.sh
conda --version
```

初始化 Bash 并在当前 shell 生效：

```bash
/overlay/miniconda3/bin/conda init bash
source ~/.bashrc
conda --version
```

`conda init bash` 会修改 `~/.bashrc`，后续新开的 Bash 会自动加载 Conda。成功标志是终端提示符出现 `(base)`，且 `conda --version` 能输出版本号。

### 3.1 配置 Conda 镜像（可选）

若默认源访问缓慢，可改用清华镜像。先查看当前来源：

```bash
conda config --show channels
conda config --show-sources
```

确认不再需要现有自定义配置后，重新写入镜像配置：

```bash
rm -f /overlay/miniconda3/.condarc
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
conda config --set show_channel_urls yes
conda config --set channel_priority flexible
conda config --show-sources
```

这会覆盖该 Conda 安装现有的频道配置；如有项目依赖特定私有源，请先备份 `.condarc` 并合并所需频道。


## 阶段 3.5：AMD GPU 训练环境（ROCm 7.2.1 + conda 多环境）

**目的：** 配置 AMD ROCm GPU 训练环境。本机单卡 gfx1100（RDNA3，48GB），ROCm 7.2.1（`/opt/rocm-7.2.1`）。conda 在 `/overlay/miniconda3`，按项目分环境（overlay 不支持硬链接，环境间是复制不是共享）。

### 当前版本（2026-08-12 实测）

依赖以 `shopsimulator/shopagent-rl/requirements.txt` 为准（247 个包全部 `==` 钉死）。

| 包 | 版本 |
| --- | --- |
| python | 3.12.13 |
| torch / torchvision / torchaudio | 2.9.1+rocm7.2.1 / 0.24.0+rocm7.2.1 / 2.9.0+rocm7.2.1（本地 whl，在 `/` 根目录） |
| triton | 3.5.1+rocm7.2.1.gita272dfa8 |
| vllm | 0.16.0，源码编译，commit `89a77b108`（源码 `/overlay/vllm-rocm-src`，脚本 `build_vllm_rocm.sh`） |
| verl | **0.8.0**，源码 `shopsimulator/shopagent-rl/verl`，以 `shop-a-verl` editable 安装提供 |
| transformers / huggingface_hub | 4.57.6 / 0.36.2 |
| peft / numpy / datasets / accelerate | 0.20.0 / 2.5.1 / 5.0.1 / 1.14.0 |
| tensordict | 0.10.0（verl 0.8.0 要求 ≥0.10.0） |
| py-cpuinfo | 9.0.0（`from vllm import LLM` 需要，`requirements.txt` 里没有，单独装） |
| pyserini + JDK | 2.3.0 + OpenJDK 21.0.11（BM25 检索用） |

> `requirements.txt` 里的 `vllm==0.16.1.dev0` 一行不用管：vLLM 由源码编译提供，装出来是 `0.16.0`（commit `89a77b108` 正好落在 `v0.16.0` tag 上，setuptools-scm 因此不加 `.dev` 后缀）。代码以 commit 为准。

### 环境布局（`/overlay/miniconda3/envs/`，实占）

| 环境 | 大小 | 包数 | 用途 |
| --- | --- | --- | --- |
| `rocm-base` | 6.4G | 255 | GPU 全栈模板：完整依赖 + vLLM 0.16.0 + verl 0.8.0。建新环境直接 `conda create --clone rocm-base -n <新名>` |
| `shop-A` | 6.3G | 256 | shopsim/shop_A（SFT/GRPO/eval/pack API）。**脚本里硬编码 `PY=/overlay/miniconda3/envs/shop-A/bin/python`，环境名必须叫 `shop-A`** |
| `opd-rocm` | 6.3G | 261 | ctv-opd；在模板基础上加 ctv-opd editable |

三个环境的 torch / vLLM / verl / transformers 版本一致，真实引擎 smoke 均已通过。

另占：`/overlay/vllm-rocm-src` 542M（含编译产物）、`/overlay/triton-kernels-src` 29M、模型缓存 `/root/.cache/huggingface` 28G。

> overlay 不支持硬链接，`conda create --clone` 是实复制（约 6.3G × 环境数）。只跑单个项目直接用 `rocm-base`。

### 建环境步骤

先把 `rocm-base` 建成 GPU 全栈模板，再从它 clone 出 `shop-A` 和 `opd-rocm`。vLLM 要排在 requirements 之前装，因为 `requirements.txt` 里的 `vllm==0.16.1.dev0` 在 PyPI 上取不到，必须由源码编译先占位。

```bash
source /overlay/miniconda3/etc/profile.d/conda.sh
BASE_PY=/overlay/miniconda3/envs/rocm-base/bin/python

# 1. GPU 全栈模板环境
conda create -n rocm-base python=3.12.13 -y
$BASE_PY -m pip install --no-deps /torch-*rocm*.whl /torchvision-*rocm*.whl \
                                  /torchaudio-*rocm*.whl /triton-*rocm*.whl
# torch 的 import 期依赖（--no-deps 不会自动装）
$BASE_PY -m pip install --no-deps typing_extensions==4.16.0 filelock==3.32.2 sympy==1.14.0 \
  mpmath==1.3.0 networkx==3.6.1 Jinja2==3.1.6 MarkupSafe==3.0.3 fsspec==2026.6.0 \
  numpy==2.5.1 setuptools==83.0.0 wheel==0.47.0 packaging==25.0
$BASE_PY -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # → True

# 2. 源码编译 vLLM 到 rocm-base（约 30 分钟；本机 128 核全并行）
#    脚本默认 commit 89a77b108，装出 0.16.0
#    CMake 会自行 clone triton-lang/triton@v3.5.0，直连 github 会被代理证书拦住，
#    所以先用 gh-proxy 取到本地，再用 TRITON_KERNELS_SRC_DIR 指过去
git clone https://gh-proxy.com/https://github.com/vllm-project/vllm.git /overlay/vllm-rocm-src
git -C /overlay/vllm-rocm-src checkout 89a77b108
git clone --branch v3.5.0 --depth 1 \
  https://gh-proxy.com/https://github.com/triton-lang/triton.git /overlay/triton-kernels-src
env TRITON_KERNELS_SRC_DIR=/overlay/triton-kernels-src/python/triton_kernels/triton_kernels \
    bash build_vllm_rocm.sh rocm-base
$BASE_PY -c "import vllm; print(vllm.__version__)"   # → 0.16.0

# 3. 完整依赖
$BASE_PY -m pip install --no-deps -r shopsimulator/shopagent-rl/requirements.txt
$BASE_PY -m pip install --no-deps shopsimulator/envs_config/zh_core_web_sm-3.8.0-py3-none-any.whl
$BASE_PY -m pip install --no-deps py-cpuinfo==9.0.0

# 4. verl 0.8.0：项目内 vendored 源码的标准 editable 安装。
#    pyproject.toml 在 shop_A 根目录，verl/ 是实际的 Python 包目录。
$BASE_PY -m pip install --no-deps -e /workspace/shopsimulator/shopagent-rl
$BASE_PY -c "import verl; print(verl.__version__)"   # → 0.8.0

# 5. Java 21（pyserini/pyjnius 要 libjvm.so）+ 还原大文件产物
apt-get install -y openjdk-21-jre-headless
bash scripts/restore_large_artifacts.sh

# 6. 从模板复制项目环境（overlay 是实复制，各约 6.3G）
#    rocm-base 里的 verl editable 安装随 clone 一起继承（finder 写的是绝对路径
#    /workspace/shopsimulator/shopagent-rl/verl），所以 shopagent-rl / opd-rocm 不用再装 verl；
#    只有 opd-rocm 额外加 ctv-opd。
conda create --clone rocm-base -n shop-A -y
conda create --clone rocm-base -n opd-rocm -y
/overlay/miniconda3/envs/opd-rocm/bin/python -m pip install --no-deps -e /workspace/opd/ctv-opd

# 7. 模型权重（缓存到 /root/.cache/huggingface/hub，共约 28G）
export PATH=/overlay/miniconda3/envs/rocm-base/bin:$PATH
hf download Qwen/Qwen3-4B
hf download Qwen/Qwen3-4B-Base
hf download Qwen/Qwen3-1.7B-Base
hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
hf download hbx/JustRL-DeepSeek-1.5B
hf download sshleifer/tiny-gpt2
hf download distilgpt2
```

`hf download` 一次只接受一个仓库，多个写一行会被当成「仓库 + 文件名」。

> **veRL 使用 editable 安装**：`shopagent-rl/verl` 是源码包，上游仓库根的打包定义在 vendoring 时没有带入；本项目已在 `shopagent-rl/pyproject.toml` 补上标准定义。执行 `pip install --no-deps -e /workspace/shopsimulator/shopagent-rl` 后，裸 Python 也会解析到项目内的 `verl` 源码。GRPO 脚本仍将 `shopagent-rl` 放入 `PYTHONPATH`，但这是为了 Ray worker 的项目模块、Hydra 的本地配置路径和 ROCm shim，不再是安装 veRL 的前提。

> 项目脚本照常 source 各自的 shim，它负责 amdsmi、functorch、spawn 和离线模型缓存。

> **ROCm NUMA 提示**：若 GRPO/vLLM 日志出现 RCCL 的 `NUMA auto balancing enabled`，应在宿主机执行 `sysctl -w kernel.numa_balancing=0`，并用 `/etc/sysctl.d/99-rocm-rccl-numa.conf` 持久化。训练容器若报 `/proc/sys` 只读，表示该设置必须由宿主机管理员完成；这是一项性能稳定性优化，不影响计算正确性。

### 验证

```bash
cd /workspace
source scripts/vllm_env_rocm_base.sh          # 或 shop-A / opd-rocm 各自的 shim

"$PY" -c "import torch,vllm,verl,transformers,peft; \
  print(torch.__version__, torch.cuda.is_available(), vllm.__version__, verl.__version__)"
# 2.9.1+rocm7.2.1.gitff65f5bc True 0.16.0 0.8.0

# HIP 扩展（编译产物，import vllm 成功不代表这几个在）
"$PY" -c "import vllm._C, vllm._rocm_C, vllm._moe_C; print('HIP ext OK')"

# 起引擎需要的 LLM 入口（比 import vllm 多一层依赖）
"$PY" -c "from vllm import LLM; print('LLM OK')"

# 真实引擎
"$PY" scripts/vllm_smoke.py                   # 末行 RESULT=PASS

# 检索链（路径就是 indexes/ 本身，它自己即索引目录）
env JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64 "$PY" -c "
from pyserini.search.lucene import LuceneSearcher
s = LuceneSearcher('/workspace/shopsimulator/ShopSimulator/shop_env/search_engine/indexes')
print('docs:', s.num_docs, '| hits:', len(s.search('phone case', k=3)))"
# docs: 23421 | hits: 1

# ctv-opd（opd-rocm 环境）
/overlay/miniconda3/envs/opd-rocm/bin/python -m pytest opd/ctv-opd/tests -q
# 50 passed
```

### vLLM 运行时（关键：`import vllm` 成功 ≠ 引擎能起）

gfx1100 非 gfx90a/gfx942，起引擎前必须先 source 对应环境的 shim 脚本：

```bash
source /workspace/scripts/vllm_env_rocm_base.sh                   # rocm-base
source /workspace/shopsimulator/shopagent-rl/scripts/vllm_env_shopA.sh   # shopagent-rl
source /workspace/opd/ctv-opd/runs/vllm_env.sh                     # opd-rocm
```

三个脚本都做 amdsmi 符号链接 + functorch shim（重启后 `/tmp` 清空，每次都要重新 source）+ spawn 设置 + 离线模型缓存路径。不 source 直接起引擎会报 `RuntimeError: Device string must not be empty`——那是 vLLM 找不到 amdsmi、平台探测为空，不是设备问题。

三个环境的真实引擎都已跑通：`Qwen/Qwen3-1.7B-Base`、Triton Attention backend、权重 3.31 GiB、KV cache 23.76 GiB、生成 34 tokens、输出含 `17 + 28 = 45`、末行 `RESULT=PASS`。复跑用 `scripts/vllm_smoke.py`。

模型缓存在 `/root/.cache/huggingface/hub`（7 个模型，28G）。`HF_HUB_CACHE` 必须指到 `hub` 这一层，指到父目录会在离线模式下报 `LocalEntryNotFoundError`。其他运行时注意点（`free_engine` 杀子进程、`LLM()` 必须在 `if __name__=="__main__":` 内、greedy 用 `temperature=0.01`、不要设 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`）见 `服务器迁移指南.md` Part B.8。

### 代理与 GitHub clone

本机 mihomo 对 github/docker/huggingface 做 TLS 中间人（自签 `CN=Proxy Certificate for GitHub/Docker/HuggingFace`），直连 `git clone https://github.com/...` 必然报 `server certificate verification failed`。加 `gh-proxy.com` 前缀即可（它证书是正常的 Let's Encrypt），不要用 `GIT_SSL_NO_VERIFY`：

```bash
git clone https://gh-proxy.com/https://github.com/<org>/<repo>.git
```

PyPI 不受影响。另：`apt-get update` 经代理拉 AMD 内网源 `compute-artifactory.amd.com` 会 502，不影响装公共包。

## 阶段 4：安装 cc-switch

**目的：** `cc-switch` 统一保存 Claude、Codex 等命令行客户端的服务商配置，并负责在服务商之间切换。

```bash
cd /workspace
mkdir -p cc-switch/bin
curl -k -L -o cc-switch/cc-switch-cli-linux-x64-musl.tar.gz \
  https://gh-proxy.com/https://github.com/saladday/cc-switch-cli/releases/latest/download/cc-switch-cli-linux-x64-musl.tar.gz
tar -xzf cc-switch/cc-switch-cli-linux-x64-musl.tar.gz -C cc-switch/bin
chmod +x cc-switch/bin/cc-switch
rm -f cc-switch/cc-switch-cli-linux-x64-musl.tar.gz
```

| 命令 | 作用 |
| --- | --- |
| `mkdir -p cc-switch/bin` | 建立固定安装目录。 |
| `tar -xzf ... -C` | 解压二进制到 `cc-switch/bin/`。 |
| `chmod +x` | 允许直接运行 cc-switch。 |
| `rm -f` | 删除不再需要的临时压缩包。 |

验证：

```bash
./cc-switch/bin/cc-switch --version
```

## 阶段 5：通过包装脚本初始化 cc-switch

### cc-switch 日常命令速查

在 `cc-switch/` 目录中执行以下常用命令：

| 想干嘛 | 命令 |
| --- | --- |
| 启动终端界面 | `./init-cc-switch.sh ui` |
| 开机恢复 Claude 和 Codex 的上次配置 | `./init-cc-switch.sh restore` |
| 查看有哪些 Provider | `./bin/cc-switch -a claude provider list`（Codex 将 `-a claude` 改为 `-a codex`） |
| 切换 Claude 的 Provider | `./init-cc-switch.sh use <id>` |
| 切换 Codex 的 Provider | `./init-cc-switch.sh -a codex use <id>` |
| 查看当前使用的 Provider | `./bin/cc-switch -a claude provider current`（Codex 将 `-a claude` 改为 `-a codex`） |
| 修改 Provider（名称、Token、模型） | `./bin/cc-switch -a claude provider edit <id>`（Codex 将 `-a claude` 改为 `-a codex`） |

切换或恢复后，请重启对应的 Claude Code 或 Codex 客户端，使新配置生效。

**目的：** 始终将 cc-switch 数据保存在项目内的 `cc-switch/.cc-switch/`，避免依赖或污染 `/root/.cc-switch`；并在切换服务商前准备 Claude/Codex 所需的最小配置文件。

仓库中的 [`cc-switch/init-cc-switch.sh`](cc-switch/init-cc-switch.sh) 已实现以下流程：

1. 根据脚本位置计算根目录，将 `CC_SWITCH_CONFIG_DIR` 指向 `cc-switch/.cc-switch/`。
2. 若项目配置尚不存在，则复制 `/root/.cc-switch/` 中的已有配置，便于从旧环境迁移。
3. 确认 `cc-switch/bin/cc-switch` 存在且可执行；否则立即报错。
4. 在 `use`、`start` 和 `provider switch` 操作前，创建 `~/.claude/settings.json` 与 `~/.codex/config.toml`（若尚不存在），并将配置目录权限设为仅当前用户可读写。
5. 最后将参数原样转发给真正的 cc-switch 二进制。

赋予脚本执行权限并查看服务商：

```bash
chmod +x cc-switch/init-cc-switch.sh
./cc-switch/init-cc-switch.sh provider list -a claude
./cc-switch/init-cc-switch.sh provider list -a codex
```

### 切换与检查服务商

先列出服务商，确认实际 ID；再按 ID 切换：

```bash
./cc-switch/init-cc-switch.sh provider switch -a claude anyrouter
./cc-switch/init-cc-switch.sh provider switch -a codex hhhl
```

`anyrouter` 和 `hhhl` 是当前环境的示例 ID。服务商显示名称与 ID 不一定相同，切换命令必须使用列表第一列的 **ID**。

验证当前选择：

```bash
./cc-switch/init-cc-switch.sh provider current -a claude
./cc-switch/init-cc-switch.sh provider current -a codex
```

## 阶段 6：安装 Claude Code 和 Codex CLI

**目的：** 安装实际执行 AI 编程任务的命令行客户端；cc-switch 只负责管理它们的服务商配置。

```bash
npm install -g @anthropic-ai/claude-code@latest
npm install -g @openai/codex@latest
```

验证安装结果：

```bash
claude --version
codex --version
```

若 npm 下载失败，优先确认阶段 2 的代理服务和环境变量是否正常：

```bash
curl -I https://registry.npmjs.org/
```

### Codex 登录：设备码方式

在没有可直接打开浏览器的远程服务器上，可用设备码完成 ChatGPT 登录：

```bash
codex login --device-auth
```

命令会输出验证地址和一次性代码；在本地浏览器打开该地址，登录并输入代码后，保持服务器终端运行至显示登录成功。


### Codex 启动方式

明确限制 Codex 只能写入当前工作区：

```bash
codex --sandbox workspace-write
```

放开沙箱访问范围，使 Codex 可以按当前系统用户的权限访问更多目录，但执行敏感命令前仍需人工确认。适用于项目需要访问多个目录、同时希望保留审批确认的场景：

```bash
codex --sandbox danger-full-access
```

同时跳过人工审批并禁用沙箱。仅适用于可以随意破坏、且与重要数据和系统隔离的环境：

```bash
codex --dangerously-bypass-approvals-and-sandbox
```

## 阶段 7：配置 GitHub 账号级 SSH Key 并提交项目

**目的：** 让这台 AMD 服务器使用 GitHub 账号级 Authentication Key，通过 SSH 访问该账号有权限的多个仓库，不需要在终端保存或粘贴 Personal Access Token（PAT）。这与只授权单个仓库的 Deploy Key 不同。

### 7.1 在服务器生成账号级 SSH 密钥

建议不要复用仓库专用密钥，单独生成一把账号级 Ed25519 密钥。私钥只保留在服务器，公钥才需要添加到 GitHub。

```bash
mkdir -p -m 700 ~/.ssh
ssh-keygen -t ed25519 \
  -f ~/.ssh/github_all \
  -C "amd-server-github-account"
cat ~/.ssh/github_all.pub
```

生成时如果提示 `Enter passphrase`：无人值守任务可以按安全规范留空；需要更高安全性时设置 passphrase，并在使用前通过 `ssh-agent` 解锁：

```bash
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/github_all
```

只复制 `.pub` 文件的完整内容（以 `ssh-ed25519` 开头），不要复制或上传 `~/.ssh/github_all` 私钥。

### 7.2 将公钥添加到 GitHub 账号

登录目标 GitHub 账号，打开 [Settings → SSH and GPG keys](https://github.com/settings/keys)，点击 **New SSH key**：

| 字段 | 值 |
| --- | --- |
| Title | `amd-server-account` |
| Key type | `Authentication Key` |
| Key | 粘贴 `~/.ssh/github_all.pub` 的完整内容 |

这里必须使用个人账号的 `Settings → SSH and GPG keys`，不要使用仓库的 `Settings → Deploy keys`。账号级 Key 可访问该账号有权限的多个仓库；如果仓库属于启用了 SSO 的组织，还需要按组织要求授权该 Key。

### 7.3 配置服务器使用这把密钥

以下配置同时覆盖 `github.com` 和显式使用 `ssh.github.com` 的地址，并通过 SSH-over-443 避开可能受限的 22 端口。如果 `~/.ssh/config` 已存在，请先合并这个 Host 区块，不要覆盖其他主机配置。

```bash
mkdir -p -m 700 ~/.ssh
cat >> ~/.ssh/config <<'EOF_SSH_CONFIG'

Host github.com ssh.github.com
    HostName ssh.github.com
    Port 443
    User git
    IdentityFile ~/.ssh/github_all
    IdentitiesOnly yes
    StrictHostKeyChecking accept-new
EOF_SSH_CONFIG

chmod 600 ~/.ssh/config
chmod 600 ~/.ssh/github_all
chmod 644 ~/.ssh/github_all.pub
```

### 7.4 验证 GitHub 账号身份

```bash
ssh -T git@github.com
```

成功时会看到类似下面的提示：

```text
Hi Double-wk! You've successfully authenticated, but GitHub does not provide shell access.
```

其中的账号名必须是添加这把 Key 的 GitHub 账号。若显示了其他账号，检查 SSH 实际采用的配置和密钥：

```bash
ssh -G github.com | grep -E 'hostname|port|user|identityfile|identitiesonly'
```

### 7.5 配置项目远程地址并测试权限

进入要提交的项目目录，先查看现有远程地址：

```bash
cd /workspace/mihomo
git remote -v
```

将 `Double-wk/amd` 替换为目标 GitHub 账号和仓库：

```bash
git remote set-url origin git@github.com:Double-wk/amd.git
git remote -v
git ls-remote --heads origin
```

`git ls-remote` 能列出远程分支和提交哈希，表示网络、账号身份和仓库读取权限都正常。若必须使用完整的 SSH-over-443 URL，也可以使用：

```bash
git remote set-url origin ssh://git@ssh.github.com:443/Double-wk/amd.git
```

### 7.6 提交并推送项目

提交前先检查变更，避免把运行日志、代理配置或密钥纳入版本控制：

```bash
git status --short
git add <要提交的文件或目录>
git commit -m "简明说明本次改动"
git branch --show-current
git push -u origin main
```

如果当前分支不是 `main`，将最后一条命令中的 `main` 换成 `git branch --show-current` 输出的分支名。遇到非快进（non-fast-forward）错误时，先执行 `git fetch origin` 检查远端变更；不要直接使用强制推送覆盖他人的提交。

### 7.7 浅克隆与目录选择

在云服务器上只需要最新代码时，使用浅克隆可减少下载时间和磁盘占用：

```bash
cd /workspace
git clone --depth 1 git@github.com:Double-wk/amd.git
cd amd
```

`--depth 1` 只获取最新提交；需要完整提交历史时再执行：

```bash
git fetch --unshallow
```

默认会创建与仓库同名的 `amd/` 目录。指定目录名可使用：

```bash
git clone --depth 1 git@github.com:Double-wk/amd.git my_project
```

不要在非空目录中以 `.` 作为克隆目标，否则会出现 `destination path '.' already exists and is not an empty directory`。若已有同名目录，优先改用新的目标目录，避免删除其中的文件。


### Deploy Key 与账号级 Authentication Key

| 类型 | 添加位置 | 权限范围 |
| --- | --- | --- |
| Deploy Key | 单个仓库的 `Settings → Deploy keys` | 通常只访问一个仓库 |
| Authentication Key | 个人账号的 `Settings → SSH and GPG keys` | 访问该账号有权限的多个仓库 |

本阶段使用的是账号级 `Authentication Key`，不是仓库级 `Deploy Key`。

## 日常使用顺序

1. 启动 mihomo，并确认 `http://127.0.0.1:9090/version` 可访问。
2. 打开新终端，确认代理环境变量已加载。
3. 使用 `provider list` 和 `provider current` 检查 cc-switch 配置。
4. 根据需要切换 Claude 或 Codex 的服务商。
5. 启动 `claude` 或 `codex`。

## 常见问题

### SSH 无法连接

执行 `sudo service ssh status`。如果服务未启动，执行 `sudo service ssh start`；同时检查端口映射与安全组规则。

### mihomo 已启动但无法联网

先查看日志：

```bash
tail -n 100 /workspace/mihomo/mihomo.log
```

然后确认代理变量指向 `127.0.0.1:7890`，并检查配置中的节点是否仍可用。

### cc-switch 找不到服务商或切换不生效

始终通过 `./cc-switch/init-cc-switch.sh` 调用，避免混用全局配置目录。然后执行：

```bash
./cc-switch/init-cc-switch.sh provider list -a claude
./cc-switch/init-cc-switch.sh provider current -a claude
```

若列表正常但客户端仍使用旧配置，重新执行对应的 `provider switch`，再重启 Claude Code 或 Codex。
