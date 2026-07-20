#!/usr/bin/env python
"""Phase D4: evaluate the monitor against two baselines on held-out policies.

Baselines (must be beaten / measured before trusting the aggregator):
  1. max-over-turns : run the frozen single-turn classifier per turn, take the
                      running max P(violation). If the aggregator can't beat
                      this, the attention earns nothing.
  2. full-concat    : whole transcript through the 12B as one input_block.
                      Likely quality ceiling and cost strawman; measures how big
                      the compositional gap even is.

Reports, all at a tau set for a target PER-CONVERSATION benign FPR on val:
  - conversation F1 / AUROC on held-out policies
  - detection turn-lag on violating conversations
  - benign FPR as a function of conversation length (the multi-turn failure mode)

Usage
  python scripts/eval_monitor.py --data outputs/data --split test \
      --agg outputs/agg_r1/aggregator.pt --out outputs/eval_test.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from mtlib.aggregator import ConversationAggregator  # noqa: E402
from mtlib.common import dump_json, setup_logging  # noqa: E402
from mtlib.encoder import TurnEncoder  # noqa: E402
from mtlib.metrics import (  # noqa: E402
    benign_fpr_by_length,
    conversation_metrics,
    detection_stats,
    tau_for_conversation_fpr,
)
from mtlib.monitor import MultiTurnMonitor  # noqa: E402
from mtlib.schema import (  # noqa: E402
    COMPOSITIONAL_SOURCES,
    Conversation,
    read_conversations,
    transcript,
    turn_block,
)


def running_max(xs: list[float]) -> list[float]:
    out, cur = [], float("-inf")
    for x in xs:
        cur = max(cur, x)
        out.append(cur)
    return out


def baseline_max_over_turns(encoder: TurnEncoder, conv: Conversation) -> list[float]:
    """Per-turn running-max single-turn score."""
    scores = encoder.score_blocks(
        [conv.policy_prompt] * len(conv.turns), [turn_block(t) for t in conv.turns]
    ).tolist()
    return running_max(scores)


def baseline_full_concat(encoder: TurnEncoder, conv: Conversation) -> list[float]:
    """Per-turn score of the growing transcript prefix (O(T^2) — the cost strawman)."""
    prefixes = [transcript(conv.turns[: t + 1]) for t in range(len(conv.turns))]
    return encoder.score_blocks([conv.policy_prompt] * len(prefixes), prefixes).tolist()


def load_latent_index(latents_dir: str | None) -> dict[str, dict]:
    """conv_id -> precomputed shard item; lets max_over_turns and the aggregator
    run without touching the 12B (only full_concat still needs it)."""
    if not latents_dir:
        return {}
    from mtlib.dataset import read_shard

    index: dict[str, dict] = {}
    for shard in sorted(Path(latents_dir).glob("*.pt")):
        for item in read_shard(shard):
            index[item["conv_id"]] = item
    return index


def cached_max_over_turns(item: dict, positive_id: int) -> list[float]:
    import torch

    probs = torch.softmax(item["turn_logits"].float(), dim=-1)[:, positive_id].tolist()
    return running_max(probs)


def cached_aggregator(item: dict, model, positive_id: int) -> list[float]:
    import torch

    with torch.no_grad():
        logits = model(
            turn_latents=item["turn_latents"].float().unsqueeze(0),
            turn_logits=item["turn_logits"].float().unsqueeze(0),
            role_ids=item["role_ids"].long().unsqueeze(0),
            attention_mask=torch.ones(1, item["turn_latents"].shape[0], dtype=torch.long),
            policy_latent=item["policy_latent"].float().unsqueeze(0),
        )
        return torch.softmax(logits[0], dim=-1)[:, positive_id].tolist()


def cached_llm_aggregator(item: dict, policy: str, model, positive_id: int) -> list[float]:
    import torch

    device = next(model.lm.parameters()).device
    with torch.no_grad():
        logits = model(
            policies=[policy],
            turn_latents=item["turn_latents"].float().unsqueeze(0).to(device),
            turn_logits=item["turn_logits"].float().unsqueeze(0).to(device),
            role_ids=item["role_ids"].long().unsqueeze(0).to(device),
            attention_mask=torch.ones(1, item["turn_latents"].shape[0], dtype=torch.long, device=device),
        )
        return torch.softmax(logits[0].float(), dim=-1)[:, positive_id].tolist()


def collect_scores(name: str, per_turn_fn, convs: list[Conversation]) -> dict:
    """Run a per-turn scorer over all conversations; return raw per-conversation traces."""
    traces = []
    for conv in convs:
        scores = per_turn_fn(conv)
        traces.append(
            {
                "conv_id": conv.conv_id,
                "guardrail_id": conv.guardrail_id,
                "label": int(conv.label == "True"),
                "onset": conv.first_violation_turn,
                "source": conv.source,
                "per_turn_scores": scores,
                "conv_score": max(scores) if scores else 0.0,
            }
        )
    return {"name": name, "traces": traces}


def score_report(traces: list[dict], tau: float, target_fpr: float = 0.02) -> dict:
    labels = [t["label"] for t in traces]
    conv_scores = [t["conv_score"] for t in traces]
    violating = [(t["onset"], t["per_turn_scores"]) for t in traces if t["label"] == 1 and t["onset"] is not None]
    benign = [t["per_turn_scores"] for t in traces if t["label"] == 0]
    report = {
        "conversation": conversation_metrics(labels, conv_scores, tau),
        "detection": detection_stats(violating, tau),
        "benign_fpr_by_length": benign_fpr_by_length(benign, tau),
    }
    # ORACLE calibration: tau set on the TEST benigns themselves. Not deployable —
    # isolates ranking quality from val->test tau drift when comparing methods.
    if benign:
        oracle_tau = tau_for_conversation_fpr(benign, target_fpr)
        report["detection_oracle_calibrated"] = detection_stats(violating, oracle_tau)

    # Compositional slice: the ONLY place the aggregator can beat max-over-turns.
    # Positives = compositional conversations (no single turn violates by
    # construction); negatives = all benign conversations. Same tau.
    comp_traces = [t for t in traces if t["source"] in COMPOSITIONAL_SOURCES or t["label"] == 0]
    comp_pos = [t for t in comp_traces if t["label"] == 1]
    if comp_pos:
        comp_violating = [(t["onset"], t["per_turn_scores"]) for t in comp_pos if t["onset"] is not None]
        report["compositional"] = {
            "n_positives": len(comp_pos),
            "conversation": conversation_metrics(
                [t["label"] for t in comp_traces], [t["conv_score"] for t in comp_traces], tau
            ),
            "detection": detection_stats(comp_violating, tau),
        }
        if benign:
            report["compositional"]["detection_oracle_calibrated"] = detection_stats(
                comp_violating, tau_for_conversation_fpr(benign, target_fpr)
            )
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir with {split}.jsonl")
    ap.add_argument("--split", default="test")
    ap.add_argument("--val-split", default="val", help="split used to set tau")
    ap.add_argument("--agg", default=None, help="aggregator checkpoint; omit to run baselines only")
    ap.add_argument("--llm-agg", default=None, help="LLM latent-aggregator checkpoint (method: llm_aggregator)")
    ap.add_argument("--encoder-ckpt", default=None)
    ap.add_argument("--target-fpr", type=float, default=0.02)
    ap.add_argument("--methods", default="aggregator,max_over_turns,full_concat")
    ap.add_argument("--test-latents", default=None,
                    help="precomputed shard dir for the test split; lets aggregator + max_over_turns skip the 12B")
    ap.add_argument("--val-latents", default=None, help="precomputed shard dir for the val split")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    logger = setup_logging("eval_monitor")
    data_dir = Path(args.data)
    test_convs = list(read_conversations(data_dir / f"{args.split}.jsonl"))
    val_convs = list(read_conversations(data_dir / f"{args.val_split}.jsonl"))
    if args.limit:
        test_convs = test_convs[: args.limit]
        val_convs = val_convs[: args.limit]
    logger.info("eval on %d test / %d val conversations", len(test_convs), len(val_convs))

    methods = [m for m in args.methods.split(",") if m]
    latents = {**load_latent_index(args.val_latents), **load_latent_index(args.test_latents)}
    have_latents = bool(latents) and all(
        c.conv_id in latents for c in test_convs + val_convs
    )
    if latents and not have_latents:
        logger.warning("latent index incomplete; falling back to live encoding")

    # The 12B is only needed for full_concat, or when no latent shards were given.
    encoder = None
    if "full_concat" in methods or not have_latents:
        encoder = TurnEncoder(model_dir=args.encoder_ckpt or None)

    scorers: dict[str, callable] = {}
    positive_id = 1  # id2label order is fixed: {0: False, 1: True}
    if "max_over_turns" in methods:
        if have_latents:
            scorers["max_over_turns"] = lambda c: cached_max_over_turns(latents[c.conv_id], positive_id)
        else:
            scorers["max_over_turns"] = lambda c: baseline_max_over_turns(encoder, c)
    if "full_concat" in methods:
        scorers["full_concat"] = lambda c: baseline_full_concat(encoder, c)
    if "aggregator" in methods and args.agg:
        model, extra = ConversationAggregator.load(args.agg)
        if have_latents:
            model = model.eval()
            scorers["aggregator"] = lambda c: cached_aggregator(latents[c.conv_id], model, positive_id)
        else:
            monitor = MultiTurnMonitor(encoder, model, tau=extra.get("tau", 0.5))
            scorers["aggregator"] = lambda c: monitor.score_conversation(
                c.policy_prompt, c.turns
            )["per_turn_conversation_probs"]
    if "llm_aggregator" in methods and args.llm_agg:
        if not have_latents:
            raise SystemExit("llm_aggregator needs --test-latents/--val-latents (cached latents)")
        import torch

        from mtlib.llm_aggregator import LlmLatentAggregator

        llm_model, _ = LlmLatentAggregator.load(args.llm_agg)
        if torch.cuda.is_available():
            llm_model = llm_model.to("cuda")
        scorers["llm_aggregator"] = lambda c: cached_llm_aggregator(
            latents[c.conv_id], c.policy_prompt, llm_model, positive_id
        )

    results = {}
    for name, fn in scorers.items():
        logger.info("scoring method=%s", name)
        val_traces = collect_scores(name, fn, val_convs)["traces"]
        test_traces = collect_scores(name, fn, test_convs)["traces"]
        benign_val = [t["per_turn_scores"] for t in val_traces if t["label"] == 0]
        tau = tau_for_conversation_fpr(benign_val, args.target_fpr) if benign_val else 0.5
        report = score_report(test_traces, tau, target_fpr=args.target_fpr)
        report["tau"] = tau
        report["tau_source"] = f"val@{args.target_fpr}_conversation_fpr"
        results[name] = report
        comp = report.get("compositional")
        logger.info(
            "method=%s: ALL conv AUROC=%.4f F1=%.4f det=%.3f fpr@20=%.3f | "
            "COMPOSITIONAL(n=%s) AUROC=%.4f det=%.3f (tau=%.4f)",
            name,
            report["conversation"]["auroc"],
            report["conversation"]["f1"],
            report["detection"]["detection_rate"],
            report["benign_fpr_by_length"].get("fpr_at_20_turns", float("nan")),
            comp["n_positives"] if comp else 0,
            comp["conversation"]["auroc"] if comp else float("nan"),
            comp["detection"]["detection_rate"] if comp else float("nan"),
            tau,
        )

    dump_json(
        {"split": args.split, "n_test": len(test_convs), "target_fpr": args.target_fpr, "methods": results},
        args.out,
    )
    logger.info("wrote %s", args.out)
    # Headline verdict: compositional detection, aggregator vs max-over-turns.
    if "aggregator" in results and "max_over_turns" in results:
        agg_c = results["aggregator"].get("compositional")
        base_c = results["max_over_turns"].get("compositional")
        if agg_c and base_c:
            logger.info(
                "VERDICT compositional detection @matched-FPR: aggregator=%.3f vs max_over_turns=%.3f",
                agg_c["detection"]["detection_rate"], base_c["detection"]["detection_rate"],
            )


if __name__ == "__main__":
    main()
