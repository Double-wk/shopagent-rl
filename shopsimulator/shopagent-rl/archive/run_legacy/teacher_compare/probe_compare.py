#!/usr/bin/env python
"""多 teacher 同条件对比探针：在同一批 30 个 train task_id 上跑多个 teacher，
统计 strict / reward / steps / token(prompt+reasoning) / token·成功。

同口径保证：
  - task_id 取自 run/teacher_compare/gpt-5.6-sol（与 gpt-5.5 / 5.6-sol / 5.6-terra 探针完全相同的 30 条，已校验三探针 set 相同）
  - system_prompt 用各 config 自带的官方 PROMPT_TEMPLATE_zh（与 gpt 逐字一致）
  - temperature=0.0, max_tokens=512, max_turns=30
  - token 经 OpenAITeacherClient._log_usage 记账（4 列含 reasoning），与 gpt 表同源

统计 + 落盘完整轨迹到本目录(trajectories_probe.jsonl，非正式采集目录)，不污染正在跑的 gpt-5.6-sol 全量(data/trajectories_raw/gpt-5.6-sol)。
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# 必须在 import teacher.* 之前设 TEACHER_USAGE_LOG（client_openai 在 import 时读取该环境变量）
USAGE_LOG = "/workspace/shopsimulator/shop_A/run/teacher_compare/probe_compare_usage.tsv"
os.environ["TEACHER_USAGE_LOG"] = USAGE_LOG          # 强制用独立文件，与全量采集的 usage 日志隔离
if os.path.exists(USAGE_LOG):
    os.remove(USAGE_LOG)                              # 清掉旧数据，保证 token 汇总干净

# 完整轨迹落盘路径（结构与 data/trajectories_raw/trajectories_raw.jsonl 一致，额外带 model 字段）
OUT_JSONL = "/workspace/shopsimulator/shop_A/run/teacher_compare/trajectories_probe.jsonl"
if os.path.exists(OUT_JSONL):
    os.remove(OUT_JSONL)

sys.path.insert(0, "/workspace/shopsimulator/shop_A")

import yaml  # noqa: E402
from teacher.collect import run_one  # noqa: E402
from teacher.client_openai import AllKeysExhausted  # noqa: E402

BASE_URL = os.environ.get("SHOP_ENV_BASE_URL", "http://127.0.0.1:5000")
WORKERS = 6

# 与 gpt 三探针完全相同的 30 条 task_id（run/teacher_compare/gpt-5.6-sol/trajectories_raw.jsonl）
TASK_IDS = [2785, 3875, 4566, 4695, 4758, 6022, 6038, 6274, 8616, 9667,
            9943, 10694, 11397, 11621, 12278, 13052, 13191, 14082, 14727, 15240,
            16930, 17076, 17381, 17996, 18212, 18910, 19802, 20575, 21182, 21721]

# 顺序：deepseek(tokenrhythm) 在前，glm(智谱 Coding Plan) 在后
CONFIGS = [
    "/workspace/shopsimulator/shop_A/configs/teacher_deepseek-v4-flash-0731.yaml",
    "/workspace/shopsimulator/shop_A/configs/teacher_glm-5.2.yaml",
]


def run_model(cfg_path, ids):
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
    # ✅ 修复：system_prompt 是顶层 key，必须合并进 teacher_cfg（run_one 从 teacher_cfg["system_prompt"] 读它）
    tc = {**cfg["teacher"], "system_prompt": cfg.get("system_prompt", "")}
    model = tc.get("model")
    rs = []
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(run_one, tid, tc, BASE_URL, False): tid for tid in ids}
        try:
            for fut in as_completed(futs):
                tid = futs[fut]
                try:
                    rec = fut.result()
                except AllKeysExhausted as e:
                    print(f"  [{model}] ⚠️ {e} —— 取消剩余任务", flush=True)
                    for f in futs:
                        f.cancel()
                    break
                except Exception as e:  # noqa: BLE001
                    print(f"  [{model}] task {tid}: 异常 {e!r}", flush=True)
                    continue
                rs.append(rec)
                line = (f"  [{model}] task {tid}: strict={rec.get('strict_success')} "
                        f"reward={rec.get('reward')} steps={rec.get('n_steps')}")
                if not rec.get("ok"):
                    line += f" err={str(rec.get('error', ''))[:70]}"
                print(line, flush=True)
        finally:
            pass
    ok = [r for r in rs if r.get("ok")]
    strict = [r for r in ok if r.get("strict_success")]
    rew = [r.get("reward", 0) for r in ok]
    steps = [r.get("n_steps", 0) for r in ok]
    print(f"\n>>> [{model}] strict {len(strict)}/{len(ok)} "
          f"({100*len(strict)/max(1,len(ok)):.0f}%) | reward {sum(rew)/max(1,len(rew)):.3f} "
          f"| steps {sum(steps)/max(1,len(steps)):.1f} | {time.time()-t0:.0f}s\n", flush=True)
    # 落盘完整轨迹（含失败行 ok=False），供后续分析；字段对齐 gpt 的 trajectories_raw.jsonl，额外加 model
    with open(OUT_JSONL, "a", encoding="utf-8") as f:
        for r in rs:
            r2 = dict(r)
            r2["model"] = model
            r2.setdefault("split", "train" if r.get("task_id", 99999) >= 1459 else "eval")
            f.write(json.dumps(r2, ensure_ascii=False) + "\n")
    nok = sum(1 for r in rs if r.get("ok"))
    print(f"  [{model}] 落盘 {len(rs)} 条(ok={nok}) → {OUT_JSONL}", flush=True)
    return model, rs


def usage_by_model(path):
    agg = {}
    if not os.path.exists(path):
        return agg
    for ln in open(path):
        p = ln.split("\t")
        if len(p) < 4:
            continue
        try:
            m, pr, co, rs = p[0], int(p[1]), int(p[2]), int(p[3].strip())
        except ValueError:
            continue
        d = agg.setdefault(m, {"prompt": 0, "completion": 0, "reasoning": 0, "calls": 0})
        d["prompt"] += pr
        d["completion"] += co
        d["reasoning"] += rs
        d["calls"] += 1
    return agg


def main():
    print(f"对比探针: {len(TASK_IDS)} 个同 task_id | workers={WORKERS} | usage_log={USAGE_LOG}", flush=True)
    print(f"对照基准(gpt 既有探针): 5.5=70%/0.889 | 5.6-sol=87%/0.971 | 5.6-terra=80%/0.961\n", flush=True)
    allrs = {}
    for cfg_path in CONFIGS:
        model, rs = run_model(cfg_path, TASK_IDS)
        allrs[model] = rs

    usage = usage_by_model(USAGE_LOG)
    print("=" * 78, flush=True)
    print("对比汇总（30 同 task_id | 官方 PROMPT_TEMPLATE_zh | temperature=0.0 | max_tokens=512）", flush=True)
    hdr = f"{'model':<26}{'strict':>9}{'reward':>9}{'steps':>7}{'prompt_tok':>12}{'reason_tok':>12}{'tok/成功':>11}"
    print(hdr, flush=True)
    for model, rs in allrs.items():
        ok = [r for r in rs if r.get("ok")]
        sc = sum(1 for r in ok if r.get("strict_success"))
        rew = sum(r.get("reward", 0) for r in ok) / max(1, len(ok))
        stp = sum(r.get("n_steps", 0) for r in ok) / max(1, len(ok))
        u = usage.get(model, {"prompt": 0, "reasoning": 0})
        tps = u["prompt"] / sc if sc else float("inf")
        tps_str = f"{tps:.0f}" if tps != float("inf") else "inf"
        print(f"{model:<26}{sc:>4}/{len(ok):<3}{rew:>9.3f}{stp:>7.1f}{u['prompt']:>12}{u['reasoning']:>12}{tps_str:>11}",
              flush=True)
    print("=" * 78, flush=True)
    print("对照(gpt 既有): gpt-5.5 21/30 70% 0.889 | gpt-5.6-sol 26/30 87% 0.971 | gpt-5.6-terra 24/30 80% 0.961",
          flush=True)
    print("注: gpt-5.6-luna 暂无既有探针数据(mcgrox key 与全量采集争用)，本次未纳入。", flush=True)


if __name__ == "__main__":
    main()
