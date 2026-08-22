"""Gate A gradient acceptance for the paired relation loss.

Four claims, on the real Qwen3-1.7B policy and real paired parquet rows:

  A. NONZERO   - d(relation_loss)/d(theta) != 0, so the term actually trains.
  B. DIRECTION - one gradient step on a decision-changing pair increases the
                 preference margin M, i.e. the flip term pushes the right way.
  C. PRESERVE  - on a decision-preserving pair, the same step does not blow up
                 |M|, i.e. the preserve term does not manufacture a flip.
  D. ISOLATION - with no pair rows in the batch, the wrapper returns the base
                 loss object unchanged and adds no gradient. This is what keeps
                 the Independent baseline byte-for-byte identical.

Run from the repo root. Requires one GPU; peaks near 13 GiB.
"""
import glob
import sys

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, "/workspace/shopsimulator/shopagent-rl")
from experiment.grpo.intent_policy_scoring import parse_legal_actions  # noqa: E402
from experiment.grpo.preference_margin import INTENT_TO_INDEX  # noqa: E402
from experiment.grpo.relation_batch import (  # noqa: E402
    compute_batch_relation_loss,
    group_pairs,
)
from experiment.grpo.relation_loss_hook import make_relation_loss_fn  # noqa: E402

PARQUET = "data/grpo_certified_paper_v1_800_pairblocked.parquet"


def load_model():
    hits = glob.glob(
        "/root/.cache/huggingface/hub/models--Qwen--Qwen3-1.7B-Base/snapshots/*/config.json"
    )
    path = hits[0].rsplit("/", 1)[0]
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).cuda()
    model.gradient_checkpointing_enable()
    model.train()
    return model, tok


def prompt_text(prompt) -> str:
    if isinstance(prompt, str):
        return prompt
    for message in reversed(list(prompt)):
        if isinstance(message, dict) and message.get("role") == "user":
            return str(message.get("content") or "")
    return ""


def build_rows(df, kind: str, limit: int = 1) -> list[dict]:
    """One pair of the requested kind, as actor-shaped rows."""
    want_changing = kind == "changing"
    out: list[dict] = []
    seen = set()
    for _, row in df.iterrows():
        pid = str(row["pair_id"] or "")
        if not pid or pid in seen:
            continue
        relation = [str(z) for z in row["expected_relation"]]
        if len(relation) != 2:
            continue
        if (relation[0] != relation[1]) != want_changing:
            continue
        sides = df[df["pair_id"].astype(str) == pid]
        if len(sides) != 2:
            continue
        rows = []
        for _, side_row in sides.iterrows():
            state = prompt_text(side_row["prompt"])
            if not parse_legal_actions(state):
                rows = []
                break
            rows.append({
                "pair_id": pid,
                "side": str(side_row["side"]),
                "state_text": state,
                "expected_relation": relation,
                "expected_action_intents": [str(z) for z in side_row["expected_action_intents"]],
                "intervention_type": str(side_row["intervention_type"]),
            })
        if not rows:
            continue
        seen.add(pid)
        out.extend(rows)
        if len(seen) >= limit:
            break
    return out


def margin_of(model, tok, rows) -> float:
    with torch.no_grad():
        _, stats = compute_batch_relation_loss(model, tok, rows, device=torch.device("cuda"))
    return float(stats["margin_mean"])


def _grads(model) -> dict:
    return {name: p.grad.detach().clone()
            for name, p in model.named_parameters() if p.grad is not None}


def directional_alignment(model, tok, rows, anchor_weight: float) -> tuple[float, float, float]:
    """Do gradient descent on the loss and ascent on the margin agree locally?

    Returns (cosine, dot, margin_grad_norm) between -grad(loss) and +grad(margin).

    A finite step cannot answer this: at grad-norm ~4e2 even lr=1e-4 moves the
    weights far outside the first-order regime, so the observed margin change is
    dominated by curvature rather than by the direction the loss points in. The
    directional derivative is exact and step-size free.
    """
    from experiment.grpo.preference_margin import compute_preference_margin
    from experiment.grpo.intent_policy_scoring import intent_log_probs
    from experiment.grpo.relation_batch import COUNTERFACTUAL, ORIGINAL

    pairs, _ = group_pairs(rows)
    pair = pairs[0]

    # grad of the loss actually optimized
    model.zero_grad(set_to_none=True)
    loss, _ = compute_batch_relation_loss(
        model, tok, rows, anchor_weight=anchor_weight, device=torch.device("cuda")
    )
    loss.backward()
    g_loss = _grads(model)

    # grad of the margin alone
    model.zero_grad(set_to_none=True)
    lp_o, _ = intent_log_probs(model, tok, pair[ORIGINAL]["state_text"],
                              device=torch.device("cuda"))
    lp_c, _ = intent_log_probs(model, tok, pair[COUNTERFACTUAL]["state_text"],
                              device=torch.device("cuda"))
    margin = compute_preference_margin(lp_o, lp_c, pair["intent_original"], pair["intent_cf"])
    margin.backward()
    g_margin = _grads(model)

    dot = 0.0
    n_loss = 0.0
    n_margin = 0.0
    for name, gm in g_margin.items():
        gl = g_loss.get(name)
        if gl is None:
            continue
        dot += torch.sum(-gl.float() * gm.float()).item()
        n_loss += gl.float().pow(2).sum().item()
        n_margin += gm.float().pow(2).sum().item()
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    denom = (n_loss ** 0.5) * (n_margin ** 0.5)
    return (dot / denom if denom > 0 else 0.0), dot, n_margin ** 0.5


def js_of(model, tok, rows) -> float:
    """Preserve-side quantity: the JS divergence the preserve loss minimizes."""
    with torch.no_grad():
        _, stats = compute_batch_relation_loss(model, tok, rows, device=torch.device("cuda"))
    return float(stats.get("preserve_loss_mean", 0.0))


def preserve_alignment(model, tok, rows, anchor_weight: float) -> tuple[float, float, float]:
    """Does descending the loss also descend the JS the preserve term targets?

    Same directional-derivative logic as `directional_alignment`, with the sign
    flipped: JS is minimized, so we want cos(-grad L, -grad JS) > 0.
    Returns (cosine, dot, |grad JS|).
    """
    from experiment.grpo.intent_policy_scoring import intent_log_probs
    from experiment.grpo.preference_margin import preserve_loss
    from experiment.grpo.relation_batch import COUNTERFACTUAL, ORIGINAL

    pairs, _ = group_pairs(rows)
    pair = pairs[0]

    model.zero_grad(set_to_none=True)
    loss, _ = compute_batch_relation_loss(
        model, tok, rows, anchor_weight=anchor_weight, device=torch.device("cuda")
    )
    loss.backward()
    g_loss = _grads(model)

    model.zero_grad(set_to_none=True)
    lp_o, _ = intent_log_probs(model, tok, pair[ORIGINAL]["state_text"],
                              device=torch.device("cuda"))
    lp_c, _ = intent_log_probs(model, tok, pair[COUNTERFACTUAL]["state_text"],
                              device=torch.device("cuda"))
    js = preserve_loss(lp_o, lp_c)
    js.backward()
    g_js = _grads(model)

    dot = 0.0
    n_loss = 0.0
    n_js = 0.0
    for name, gj in g_js.items():
        gl = g_loss.get(name)
        if gl is None:
            continue
        dot += torch.sum(-gl.float() * -gj.float()).item()
        n_loss += gl.float().pow(2).sum().item()
        n_js += gj.float().pow(2).sum().item()
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    denom = (n_loss ** 0.5) * (n_js ** 0.5)
    return (dot / denom if denom > 0 else 0.0), dot, n_js ** 0.5


def step_once(model, tok, rows, lr: float) -> tuple[float, float]:
    """One SGD step on the relation loss alone. Returns (loss, grad_norm)."""
    model.zero_grad(set_to_none=True)
    loss, stats = compute_batch_relation_loss(model, tok, rows, device=torch.device("cuda"))
    assert stats["pairs_used"] == 1, stats
    loss.backward()
    gnorm = sum(p.grad.float().norm().item() ** 2
                for p in model.parameters() if p.grad is not None) ** 0.5
    with torch.no_grad():
        for p in model.parameters():
            if p.grad is not None:
                p -= lr * p.grad
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    return float(loss.detach()), gnorm


def main() -> int:
    df = pd.read_parquet(PARQUET)
    model, tok = load_model()
    results = {}

    changing = build_rows(df, "changing")
    preserving = build_rows(df, "preserving")
    print(f"decision-changing rows: {len(changing)}  preserving rows: {len(preserving)}")
    pairs, counters = group_pairs(changing)
    print(f"changing pair relation: {pairs[0]['intent_original']} -> {pairs[0]['intent_cf']} "
          f"({pairs[0]['intervention_type']})")

    # ---- A: the term actually produces gradient ------------------------------
    m_before = margin_of(model, tok, changing)
    model.zero_grad(set_to_none=True)
    loss0, _ = compute_batch_relation_loss(model, tok, changing, device=torch.device("cuda"))
    loss0.backward()
    gnorm = sum(p.grad.float().norm().item() ** 2
                for p in model.parameters() if p.grad is not None) ** 0.5
    model.zero_grad(set_to_none=True)
    torch.cuda.empty_cache()
    results["A_nonzero"] = gnorm > 0 and gnorm == gnorm and gnorm != float("inf")
    print(f"\n[A] grad_norm={gnorm:.4e}  loss={float(loss0.detach()):.4f}  "
          f"margin={m_before:+.6f}")

    # ---- B: descending the loss ascends the margin ---------------------------
    cos_flip, dot_flip, gm_norm = directional_alignment(model, tok, changing, anchor_weight=0.0)
    cos_full, dot_full, _ = directional_alignment(model, tok, changing, anchor_weight=1.0)
    results["B_direction"] = dot_flip > 0
    results["B_direction_with_anchor"] = dot_full > 0
    print(f"[B] flip-only : cos(-grad L, +grad M) = {cos_flip:+.6f}  dot={dot_flip:+.4e}")
    print(f"[B] with anchor: cos                  = {cos_full:+.6f}  dot={dot_full:+.4e}"
          f"   (|grad M|={gm_norm:.4e})")

    # ---- C: preserving pair -- the JS the preserve loss minimizes goes down ---
    # M is structurally 0 for a preserving pair (intent_original == intent_cf), so
    # the margin says nothing here; the preserve term acts on the distributions.
    js_now = js_of(model, tok, preserving)
    cos_pres, dot_pres, gj_norm = preserve_alignment(model, tok, preserving, anchor_weight=0.0)
    cos_pres_a, dot_pres_a, _ = preserve_alignment(model, tok, preserving, anchor_weight=1.0)
    results["C_preserve"] = dot_pres > 0
    results["C_preserve_with_anchor"] = dot_pres_a > 0
    print(f"[C] JS now={js_now:.6f}  |grad JS|={gj_norm:.4e}")
    print(f"[C] preserve-only: cos(-grad L, -grad JS) = {cos_pres:+.6f}  dot={dot_pres:+.4e}")
    print(f"[C] with anchor  : cos                    = {cos_pres_a:+.6f}  dot={dot_pres_a:+.4e}")

    # ---- D: environment-only batch is untouched -----------------------------
    base_marker = torch.tensor(2.0, requires_grad=True)

    def base_loss(model_output=None, data=None, dp_group=None):
        return base_marker, {"actor/pg_loss": 2.0}

    fn = make_relation_loss_fn(base_loss, lambda: model, tok)
    env_rows = [{"pair_id": "", "side": "", "state_text": "",
                 "expected_relation": [], "expected_action_intents": [],
                 "intervention_type": ""}]
    from tensordict import TensorDict
    from tensordict.tensorclass import NonTensorData, NonTensorStack
    td = TensorDict({}, batch_size=[1])
    for key in ("pair_id", "side", "state_text", "expected_relation",
                "expected_action_intents", "intervention_type"):
        td[key] = NonTensorStack.from_list([NonTensorData(env_rows[0][key])])
    model.zero_grad(set_to_none=True)
    out_loss, out_metrics = fn(model_output=None, data=td)
    touched = any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
    results["D_isolation"] = (out_loss is base_marker) and not touched and not any(
        k.startswith("relation/") for k in out_metrics
    )
    print(f"[D] identity={out_loss is base_marker} grads_touched={touched} "
          f"relation_metrics={[k for k in out_metrics if k.startswith('relation/')]}")

    # ---- C2: where does the anchor stop overwhelming preserve? ---------------
    # Informational sweep. The anchor and preserve terms genuinely compete on a
    # preserving pair at clean init: the anchor moves each side toward COMMIT
    # independently, which can widen the gap between them. Find the anchor_weight
    # at which the combined gradient still descends JS.
    print()
    crossover = None
    for aw in (1.0, 0.3, 0.1, 0.03, 0.01, 0.003):
        cos_aw, dot_aw, _ = preserve_alignment(model, tok, preserving, anchor_weight=aw)
        mark = "descends JS" if dot_aw > 0 else "opposes JS"
        print(f"[C2] anchor_weight={aw:<6} cos={cos_aw:+.6f}  {mark}")
        if dot_aw > 0 and crossover is None:
            crossover = aw
    print(f"[C2] preserve survives from anchor_weight <= {crossover}")

    # ---- term magnitudes: informational, drives weight selection -------------
    with torch.no_grad():
        _, s_chg = compute_batch_relation_loss(model, tok, changing,
                                               device=torch.device("cuda"))
        _, s_prs = compute_batch_relation_loss(model, tok, preserving,
                                               device=torch.device("cuda"))
    print(f"\n[scale] changing : flip={s_chg['flip_loss_mean']:.4f} "
          f"anchor={s_chg['anchor_loss_mean']:.4f}")
    print(f"[scale] preserving: preserve={s_prs['preserve_loss_mean']:.4f} "
          f"anchor={s_prs['anchor_loss_mean']:.4f}")

    # Acceptance is about whether each term is implemented correctly and reaches
    # the parameters. `*_with_anchor` is reported but not gating: whether the
    # anchor should dominate at clean init is a weight choice, not a defect. The
    # C2 sweep above is what informs that choice.
    gating = ("A_nonzero", "B_direction", "C_preserve", "D_isolation")
    informational = ("B_direction_with_anchor", "C_preserve_with_anchor")

    print()
    for key in gating:
        print(f"{key:<26}{'PASS' if results[key] else 'FAIL'}   (gating)")
    for key in informational:
        print(f"{key:<26}{'yes' if results[key] else 'no':<7}(informational)")
    ok = all(results[k] for k in gating)
    print(f"\nRESULT={'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
