#!/usr/bin/env bash
# 自动续跑：等 pack_api(:5000) 冷启完成，就自动以 w=36 拉起 collect。
# 幂等：collect.py 会按 jsonl 去重，重跑安全。
# 用法：setsid nohup bash auto_resume_collect.sh </dev/null >/dev/null 2>&1 &
# 状态看：run/auto_resume.state ；进度看 collect.log。
set -u
SHOP=/workspace/shopsimulator/shop_A
PY=/overlay/miniconda3/envs/shop-A/bin/python
LOG=$SHOP/data/trajectories_raw/gpt-5.6-terra/collect.log
STATE=$SHOP/run/auto_resume.state

log(){ echo "$(date '+%F %T') $*" >> "$STATE"; }
log "auto_resume: 开始等 pack_api :5000（冷启 ~8-10min）"

for i in $(seq 1 60); do            # 最多等 ~20min
  if ss -ltn 2>/dev/null | grep -q ':5000'; then
    log "pack_api READY（:5000 up，等了 ${i}x20s）；5s 后拉起 collect"
    sleep 5
    cd "$SHOP" || { log "ERROR: cd $SHOP 失败"; exit 1; }
    nohup "$PY" -m teacher.collect \
      --config configs/teacher_gpt-5.6-terra.yaml \
      --targets data/sft_collect_targets_5000.json \
      --out data/trajectories_raw/gpt-5.6-terra \
      --workers 36 \
      >> "$LOG" 2>&1 &
    log "collect 已拉起 pid=$!（w=36）；auto_resume 完成"
    exit 0
  fi
  sleep 20
done
log "TIMEOUT：pack_api :5000 在 ~20min 内没就绪——collect 未启动，请检查 pack_api"
exit 1
