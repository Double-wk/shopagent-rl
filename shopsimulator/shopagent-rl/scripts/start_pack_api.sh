#!/usr/bin/env bash
# 启动 pack_api (ShopSimulator env 服务 :5000) — 修正版
# (旧脚本 scripts/00_start_env.sh 指向已失效的 /workspace/persistent/..., 本机不可用)
#
# CPU-only (flask+pyserini), 与 SFT/GRPO 的 GPU 抢占无关, 可与训练并行运行。
# nohup 后台: 关终端不影响 (PPID=1, 忽略 SIGHUP)。
# 重启容器/机器后需手动重跑本脚本 (非 systemd, 不自启)。
set -u
PY=/overlay/miniconda3/envs/shop-A/bin/python
cd /workspace/shopsimulator/ShopSimulator/shop_env/shop_env   # pack_api.py 的 sys.path.append("../") 依赖此 cwd
export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64           # pyserini/pyjnius 要 libjvm.so
export JVM_PATH="$JAVA_HOME/lib/server/libjvm.so"
# This container sees the host's 1-TiB RAM but is cgroup-limited to 64 GiB.
# Cap Lucene's JVM explicitly so its ergonomics cannot size a heap from host RAM.
: "${JAVA_TOOL_OPTIONS:=-Xms256m -Xmx2g}"
export JAVA_TOOL_OPTIONS
# env16 is the standard profile for this 64-GiB container.  Override only
# when the caller has intentionally provisioned and validated another size.
: "${SHOP_ENV_MAX_NUM:=16}"
export SHOP_ENV_MAX_NUM

# 防重复启动: 若已在跑, 提示并退出
# 注意实际 cmdline 是 "python -u pack_api.py"（带 -u），匹配 pack_api.py 即可；
# [p]ack 的写法避免 pgrep 匹配到含本模式字面量的调用方自身 shell。
if pgrep -f "[p]ack_api.py" >/dev/null 2>&1; then
  echo "pack_api 已在运行 (PID $(pgrep -f '[p]ack_api.py'))"; exit 0
fi

# A separate session prevents managed shells from forwarding their teardown
# signal to the long-running local service.  Unbuffered output preserves
# Python/JVM startup failures in the persistent log.
setsid nohup "$PY" -u pack_api.py >> /workspace/shopsimulator/shopagent-rl/run/pack_api.log 2>&1 < /dev/null &
echo $! > /workspace/shopsimulator/shopagent-rl/run/pack_api.pid
echo "pack_api 启动 PID: $!"
echo "  日志: tail -f /workspace/shopsimulator/shopagent-rl/run/pack_api.log"
echo "  就绪标志(冷启~2min): log 出现 'Running on http://127.0.0.1:5000'"
