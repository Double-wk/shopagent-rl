# gpt-5.6-terra 采集 — 命令留档（✅ 采集已完成）

> ✅ **采集已完成（2026-08-08）**：全 train `[1459,23421)` 随机 5000 目标已采满，
> `teacher.validate` 实过 **3793 条 strict 入训**（75.9% pass，reward 均值 0.897）。
> 数据现状以 [`../DATA.md`](../DATA.md) 为准。本文保留**采集命令 + 运维经验**，供未来补采/重采时直接复用。
>
> 断点续传：`collect.py` 启动时读 jsonl 的 `done_ids` 自动去重，**已采集的不会重做**，命令幂等。

## 命令（采集时用，现仅留档）

```bash
cd /workspace/shopsimulator/shop_A
nohup /overlay/miniconda3/envs/shop-A/bin/python -m teacher.collect \
  --config configs/teacher_gpt-5.6-terra.yaml \
  --targets data/sft_collect_targets_5000.json \
  --out data/trajectories_raw/gpt-5.6-terra \
  --workers 24 \
  >> data/trajectories_raw/gpt-5.6-terra/collect.log 2>&1 &
```

## 关键参数（别乱改）

- **`--out data/trajectories_raw/gpt-5.6-terra`**：**canonical 输出目录**，与 `build_sft_data.sh`、
  `watchdog_collect.sh`、[`DATA.md`](../DATA.md) 一致。早先因污染曾用过 `_v2` 临时名、把脏的 v1 挪到
  `_trainonly/`；**最终干净版落在 `gpt-5.6-terra/`**（5000 行），`_v2` 名已废弃、磁盘上不存在。
  漏掉 `--out` 会按 model 默认派生同名目录（本例恰好相同），但**显式带上最稳**。
- `--targets data/sft_collect_targets_5000.json`：固定 5000 个 task id，与 GRPO/eval/Ablation 集不重叠。
- `--config configs/teacher_gpt-5.6-terra.yaml`：`model=gpt-5.6-terra`，多 (mcgrox) key round-robin。
- `--workers 24`：并发（config `num_workers`）。越大越吃 key 额度/env 池；满屏 429/502 就降。
  （watchdog 兜底脚本里仍硬编码 36，采集已完不影响。）
- conda env 是**大写 `shop-A`**（`ps` 里 `pack_api` 显示的小写 `shop-a` 是改名前的残留 inode）。

## key 池（随充值/余额变化，采集已完不展开）

- mcgrox 渠道多 key round-robin；余额耗尽返 401/402/403 自动剔除（见 memory `tokenrhythm-402-payment-required`、`mcgrox-newkey-smoke-unreliable`、`mcgrox-model-aliasing`）。
- 具体 key 的活/死状态是**易变**的（充值即恢复），本文不固化数字。补采前用 `teacher_*-c7d5.yaml` 单 key 探针确认 strict≈80% 再开全量。
- ⚠️ mcgrox 有把 `gpt-5.6-terra` 别名到 `gpt-5.5`（更贵计费）的前科——新 key 务必跑一轮别名探针。

## 运维经验（踩过的坑）

- 进程常被 **SIGKILL** 干掉（无 traceback、非 OOM，924GB 内存空闲）→ 多半外部 kill，非代码崩溃。用上面命令重启即可，续传安全。
  ⚠️ 重启 collect 用 **SIGTERM**，别 `kill -9`：硬杀会泄漏 env（~900s 才回收）→ 打空 40-env 池 → 0 写入空烧 mcgrox（见 memory `collect-sigkill-leaks-env-pool`、`pack-api-env-pool-leak`）。
- 前置依赖：env 服务 `pack_api.py` 必须在 `:5000` 跑着（collect 经 HTTP 调它）。
  - ⚠️ **本环境 `ss -ltn` 返回空（失效），别用 `ss`/`netstat` 判 :5000**（永远判 DOWN）。用 python socket：
    `python -c "import socket;s=socket.socket();s.settimeout(3);print(s.connect_ex(('127.0.0.1',5000)))"` → `0`=在服务。
  - pack_api 冷启 ~8-10min；曾出现 env 池泄漏被杀，需重启。

## 监控（采集进行时用）

```bash
tail -f /workspace/shopsimulator/shop_A/data/trajectories_raw/gpt-5.6-terra/collect.log
wc -l  /workspace/shopsimulator/shop_A/data/trajectories_raw/gpt-5.6-terra/trajectories_raw.jsonl   # 到 5000 即完成
pgrep -af teacher.collect
```

## 自动续跑（watchdog，人不在时靠它）

`scripts/watchdog_collect.sh`：每 5min 巡检；collect 挂了 **且** `:5000` python-socket 探活为 UP，就自动续跑；jsonl≥5000 自动退出（**采集已完，当前不会触发**）。
- 探活用 python socket（不用 ss），防自匹配用 `comm=python`（不用 `pgrep -f`，见 memory `avoid-pkill-self-match`）。
- 注意：若 **pack_api 自己挂了**，watchdog **不会**重启 pack_api（冷启有风险，留给人），只在 state 里记 `BLOCK: pack_api DOWN`。
- 启动：`setsid bash scripts/watchdog_collect.sh </dev/null >/dev/null 2>&1 &` ；状态：`cat run/watchdog.state`。
- ⚠️ 旧脚本 `scripts/auto_resume_collect.sh` 用 `ss` 探活**已失效**，弃用（保留作反面教材）。
