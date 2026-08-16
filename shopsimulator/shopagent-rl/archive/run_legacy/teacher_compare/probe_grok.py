#!/usr/bin/env python
"""grok-4.5 单模型探针：在与 gpt/deepseek/glm 完全相同的 30 个 train task_id 上跑，
落盘到 run/teacher_compare/grok-4.5/（与其他 teacher 子目录同构），不污染全量采集、
不重跑已 402 的 deepseek/glm。

同口径（与 probe_compare.py 一致）：
  - task_id: 同一批 30 条（取自 gpt-5.6-sol 探针）
  - system_prompt: 官方 PROMPT_TEMPLATE_zh（config 自带）
  - temperature=0.0, max_tokens=512, max_turns=30
  - token 经 OpenAITeacherClient._log_usage 记账（含 reasoning）
"""
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

OUT_DIR = "/workspace/shopsimulator/shop_A/run/teacher_compare/grok-4.5"
os.makedirs(OUT_DIR, exist_ok=True)

# ⚠️ 必须在 import teacher.* 之前设 TEACHER_USAGE_LOG（client_openai 在 import 时读取）
USAGE_LOG = os.path.join(OUT_DIR, "usage.tsv")
os.environ["TEACHER_USAGE_LOG"] = USAGE_LOG
if os.path.exists(USAGE_LOG):
    os.remove(USAGE_LOG)

OUT_JSONL = os.path.join(OUT_DIR, "trajectories_raw.jsonl")
if os.path.exists(OUT_JSONL):
    os.remove(OUT_JSONL)

sys.path.insert(0, "/workspace/shopsimulator/shop_A")

import yaml  # noqa: E402
from teacher.collect import run_one  # noqa: E402
from teacher.client_openai import AllKeysExhausted  # noqa: E402

BASE_URL = os.environ.get("SHOP_ENV_BASE_URL", "http://127.0.0.1:5000")
WORKERS = 6
CFG = "/workspace/shopsimulator/shop_A/configs/teacher_grok-4.5.yaml"

# 与 gpt/deepseek/glm 三探针完全相同的 30 条 task_id
TASK_IDS = [2785, 3875, 4566, 4695, 4758, 6022, 6038, 6274, 8616, 9667,
            9943, 10694, 11397, 11621, 12278, 13052, 13191, 14082, 14727, 15240,
            16930, 17076, 17381, 17996, 18212, 18910, 19802, 20575, 21182, 21721]


def run_model(cfg_path, ids):
    cfg = yaml.safe_load(open(cfg_path, encoding="utf-8"))
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
    # 落盘完整轨迹（含失败行），字段对齐其他 teacher 的 trajectories_raw.jsonl，额外带 model
    with open(OUT_JSONL, "a", encoding="utf-8") as f:
        for r in rs:
            r2 = dict(r)
            r2["model"] = model
            r2.setdefault("split", "train" if r.get("task_id", 99999) >= 1459 else "eval")
            f.write(json.dumps(r2, ensure_ascii=False) + "\n")
    nok = sum(1 for r in rs if r.get("ok"))
    print(f"  [{model}] 落盘 {len(rs)} 条(ok={nok}) → {OUT_JSONL}", flush=True)
    return model, rs


def usage_sum(path):
    d = {"prompt": 0, "completion": 0, "reasoning": 0, "calls": 0}
    if not os.path.exists(path):
        return d
    for ln in open(path):
        p = ln.split("\t")
        if len(p) < 4:
            continue
        try:
            _m, pr, co, rs = p[0], int(p[1]), int(p[2]), int(p[3].strip())
        except ValueError:
            continue
        d["prompt"] += pr
        d["completion"] += co
        d["reasoning"] += rs
        d["calls"] += 1
    return d


def main():
    print(f"grok-4.5 探针: {len(TASK_IDS)} 个同 task_id | workers={WORKERS} | out={OUT_DIR}", flush=True)
    print(f"对照基准(gpt/deepseek/glm 既有): "
          f"5.6-sol=87%/0.971 | deepseek=73%/0.809 | glm=70%/0.792 | 5.5=70%/0.889\n", flush=True)
    model, rs = run_model(CFG, TASK_IDS)

    ok = [r for r in rs if r.get("ok")]
    sc = sum(1 for r in ok if r.get("strict_success"))
    rew = sum(r.get("reward", 0) for r in ok) / max(1, len(ok))
    stp = sum(r.get("n_steps", 0) for r in ok) / max(1, len(ok))
    u = usage_sum(USAGE_LOG)
    tps = u["prompt"] / sc if sc else float("inf")
    tps_str = f"{tps:.0f}" if tps != float("inf") else "inf"
    print("=" * 78, flush=True)
    print("grok-4.5 汇总（30 同 task_id | 官方 PROMPT_TEMPLATE_zh | temperature=0.0 | max_tokens=512）", flush=True)
    hdr = f"{'model':<12}{'strict':>9}{'reward':>9}{'steps':>7}{'prompt_tok':>12}{'reason_tok':>12}{'tok/成功':>11}"
    print(hdr, flush=True)
    print(f"{model:<12}{sc:>4}/{len(ok):<3}{rew:>9.3f}{stp:>7.1f}"
          f"{u['prompt']:>12}{u['reasoning']:>12}{tps_str:>11}", flush=True)
    print("=" * 78, flush=True)
    print("对照(gpt/deepseek/glm 既有):", flush=True)
    print("  gpt-5.6-sol      26/30 87% 0.971  steps5.9  prompt698k  reason17k  tok/成功26,848", flush=True)
    print("  deepseek-v4      22/30 73% 0.809  steps14.2 prompt3,996k reason153k tok/成功181,620", flush=True)
    print("  glm-5.2          21/30 70% 0.792  steps11.0 prompt4,244k reason151k tok/成功202,131", flush=True)
    print("  gpt-5.5          21/30 70% 0.889  steps5.9  prompt847k  reason19k  tok/成功40,339", flush=True)
    print("注: env reset 随机化价格 → r_price 跨 run 波动；strict 趋势稳定。", flush=True)


if __name__ == "__main__":
    main()
