"""Verify intent_log_probs on real parquet rows with the real Qwen3-1.7B policy.

Checks, per (intervention_type, side) bucket:
  1. normalization  - mass over the legal set sums to 1 (or <1 iff inert buttons)
  2. availability   - -inf appears exactly for intents with no legal action
  3. reachability   - every expected_action_intent is finite
  4. gradient flow  - d(log pi)/d(theta) != 0 through the aggregation
"""
import json
import math
import sys
from collections import defaultdict

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace/shopsimulator/shopagent-rl")
from experiment.grpo.intent_policy_scoring import (  # noqa: E402
    intent_log_probs,
    intent_of_action,
    parse_legal_actions,
)
from experiment.grpo.preference_margin import CANONICAL_INTENTS  # noqa: E402

MODEL = "/root/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B-Base/snapshots"
PARQUET = "data/grpo_certified_paper_v1_800_pairblocked.parquet"


def resolve_model_path() -> str:
    import glob
    hits = glob.glob(MODEL + "/*/config.json")
    if not hits:
        hits = glob.glob("/root/.cache/huggingface/**/Qwen3-1.7B-Base*/**/config.json",
                         recursive=True)
    if not hits:
        raise SystemExit("model not found")
    return hits[0].rsplit("/", 1)[0]


def prompt_text(prompt) -> str:
    if isinstance(prompt, str):
        return prompt
    parts = []
    for m in prompt:
        parts.append(m["content"] if isinstance(m, dict) else str(m))
    return "\n".join(parts)


def main() -> int:
    path = resolve_model_path()
    tok = AutoTokenizer.from_pretrained(path)
    # bf16 to match training dtype and to leave room for the backward pass: the
    # scored candidates share a ~1.3k-token prefix, so the fp32 activation graph
    # does not fit in 48 GiB alongside the weights.
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda().eval()

    df = pd.read_parquet(PARQUET)
    buckets = defaultdict(lambda: {"n": 0, "norm_ok": 0, "avail_ok": 0, "reach_ok": 0})
    failures = []
    # Full 800 rows at fp32 would be slow; stratify to cover every bucket.
    df["_bucket"] = df["intervention_type"].astype(str) + "/" + df["side"].astype(str)
    sample = df.groupby("_bucket", group_keys=False).head(6)
    print(f"buckets: {sorted(df['_bucket'].unique())}")
    print(f"sampled {len(sample)} of {len(df)} rows")

    for _, row in sample.iterrows():
        state = prompt_text(row["prompt"])
        legal = parse_legal_actions(state)
        mapped = [a for a in legal if intent_of_action(a) is not None]
        b = buckets[row["_bucket"]]
        b["n"] += 1

        with torch.no_grad():
            lp, info = intent_log_probs(model, tok, state)

        # 1. normalization: exact 1.0 iff every legal action maps to an intent.
        # Rows with no rendered button list (sample_mode='environment') are
        # legitimately unscorable and must carry zero mass, not 1.0.
        mass = lp.exp().sum().item()
        if not legal:
            ok_mass = info["scorable"] is False and mass == 0.0
        elif len(mapped) == len(legal):
            ok_mass = abs(mass - 1.0) < 1e-4
        else:
            ok_mass = 0.0 < mass < 1.0 + 1e-4
        if ok_mass:
            b["norm_ok"] += 1
        else:
            failures.append(f"norm {row['_bucket']} mass={mass:.6f} "
                            f"legal={len(legal)} mapped={len(mapped)}")

        # 2. availability
        present = {intent_of_action(a) for a in mapped}
        ok = True
        for i, name in enumerate(CANONICAL_INTENTS):
            if (name in present) != bool(torch.isfinite(lp[i])):
                ok = False
                failures.append(f"avail {row['_bucket']} {name} present={name in present} "
                                f"finite={bool(torch.isfinite(lp[i]))}")
        b["avail_ok"] += int(ok)

        # 3. reachability of the expected intents
        exp = row["expected_action_intents"]
        exp = json.loads(exp) if isinstance(exp, str) else list(exp)
        reach = all(torch.isfinite(lp[CANONICAL_INTENTS.index(e)])
                    for e in exp if e in CANONICAL_INTENTS)
        b["reach_ok"] += int(reach)
        if not reach:
            failures.append(f"reach {row['_bucket']} expected={exp} lp={lp.tolist()}")

    print()
    hdr = f"{'bucket':<34}{'n':>4}{'norm':>7}{'avail':>7}{'reach':>7}"
    print(hdr)
    all_ok = True
    for k in sorted(buckets):
        v = buckets[k]
        print(f"{k:<34}{v['n']:>4}{v['norm_ok']:>7}{v['avail_ok']:>7}{v['reach_ok']:>7}")
        all_ok &= v["norm_ok"] == v["avail_ok"] == v["reach_ok"] == v["n"]

    # 4. gradient flow through the aggregation, on a real *scorable* row
    scorable = [r for _, r in sample.iterrows() if parse_legal_actions(prompt_text(r["prompt"]))]
    if not scorable:
        print("no scorable row sampled")
        return 1
    # Worst case first: the widest legal set is the one that has to fit.
    scorable.sort(key=lambda r: -len(parse_legal_actions(prompt_text(r["prompt"]))))
    state = prompt_text(scorable[0]["prompt"])
    n_cand = len(parse_legal_actions(state))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    # transformers gates checkpointing on self.training, so eval() silently makes
    # gradient_checkpointing_enable() a no-op. Without it the backward pass over
    # ~29 candidates x ~1.2k tokens needs >48 GiB and OOMs.
    model.gradient_checkpointing_enable()
    model.train()
    lp, _ = intent_log_probs(model, tok, state)
    finite = lp[torch.isfinite(lp)]
    model.zero_grad(set_to_none=True)
    finite.sum().backward()
    gnorm = sum(p.grad.float().norm().item() ** 2 for p in model.parameters()
                if p.grad is not None) ** 0.5
    peak = torch.cuda.max_memory_allocated() / 2 ** 30
    model.eval()
    model.gradient_checkpointing_disable()
    print(f"grad probe: n_candidates={n_cand} peak={peak:.2f}GiB")
    print(f"\ngrad norm through aggregation: {gnorm:.6e}")
    grad_ok = math.isfinite(gnorm) and gnorm > 0

    for f in failures[:12]:
        print("FAIL", f)
    print(f"\nRESULT={'PASS' if (all_ok and grad_ok and not failures) else 'FAIL'}")
    return 0 if (all_ok and grad_ok and not failures) else 1


if __name__ == "__main__":
    raise SystemExit(main())
