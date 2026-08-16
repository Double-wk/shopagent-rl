#!/usr/bin/env bash
# Watchdog：周期巡检 collect 是否存活；挂了 + pack_api(:5000) 在服务就自动续跑（w=36）。
#
# 为什么需要它：collect 常被外部 SIGKILL 干掉（见 RUN_collect_terra.md），挂了就没人写 jsonl。
# 注：teacher 采集已采满 5000（3793 strict 入训），本脚本现在触发即「jsonl≥5000 → 退出」，仅留作未来补采兜底。
#
# 为什么不用 ss：本环境 `ss -ltn` 返回空（失效），老的 auto_resume_collect.sh 用 ss 探活
#   永远判 DOWN → 永远不拉 collect（今天已踩坑，jsonl 卡 1250 空转 20min）。
#   这里改用 python socket connect_ex 探 :5000（跟 collect 实际连 pack_api 同理）。
#
# 为什么用 comm=python：`pgrep/ps -f teacher.collect` 会匹配本脚本自身的命令行（含该字样）
#   → 误判存活。这里 awk 先按 comm=python 过滤，排除 bash 本身（见 memory avoid-pkill-self-match）。
#
# 幂等：collect.py 启动读 jsonl 的 done_ids 去重，重跑安全。
# 用法：setsid nohup bash watchdog_collect.sh </dev/null >/dev/null 2>&1 &
# 状态：run/watchdog.state ；进度：data/trajectories_raw/gpt-5.6-terra/collect.log
set -u
SHOP=/workspace/shopsimulator/shop_A
PY=/overlay/miniconda3/envs/shop-A/bin/python
LOG=$SHOP/data/trajectories_raw/gpt-5.6-terra/collect.log
JSONL=$SHOP/data/trajectories_raw/gpt-5.6-terra/trajectories_raw.jsonl
STATE=$SHOP/run/watchdog.state
TARGET=5000

log(){ echo "$(date '+%F %T') $*" >> "$STATE"; }

port_up(){           # 返回 UP/DOWN：python socket 探 :5000（ss 失效）
  "$PY" -c "import socket;s=socket.socket();s.settimeout(3);print('UP' if s.connect_ex(('127.0.0.1',5000))==0 else 'DOWN')" 2>/dev/null
}
collect_alive(){     # 返回存活的 collect python 进程数（comm=python + -m teacher.collect）
  ps -eo comm,args | awk '$1 ~ /python/ && $0 ~ /-m teacher[.]collect/ {c++} END{print c+0}'
}

log "watchdog 启动（每 5min 巡检；collect 挂 + :5000 UP 即续跑 w=36；jsonl≥$TARGET 退出）"
while true; do
  sleep 300
  LINES=$(wc -l < "$JSONL" 2>/dev/null || echo 0)
  if [ "${LINES:-0}" -ge "$TARGET" ]; then
    log "DONE：jsonl=$LINES >= $TARGET，采集完成，watchdog 退出"; exit 0
  fi
  if [ "$(collect_alive)" -gt 0 ]; then
    log "OK：collect 活着（jsonl=$LINES/$TARGET）"; continue
  fi
  log "WARN：collect 不在，探 pack_api :5000 ..."
  if [ "$(port_up)" != "UP" ]; then
    log "BLOCK：pack_api :5000 DOWN — 不续跑（collect 无 pack_api 跑不了；需人工重启 pack_api，冷启 ~8-10min，重启后本 watchdog 会自动续跑 collect）"
    continue
  fi
  log "pack_api UP，30s 后续跑 collect（jsonl 当前 $LINES）"
  sleep 30
  cd "$SHOP" || { log "ERROR：cd $SHOP 失败"; continue; }
  setsid "$PY" -m teacher.collect \
    --config configs/teacher_gpt-5.6-terra.yaml \
    --targets data/sft_collect_targets_5000.json \
    --out data/trajectories_raw/gpt-5.6-terra \
    --workers 36 \
    >> "$LOG" 2>&1 </dev/null &
  log "续跑 collect pid=$!（w=36，jsonl 起始 $LINES）"
done
