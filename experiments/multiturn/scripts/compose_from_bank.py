#!/usr/bin/env python
"""Phase v5-2: recombine bank latents into a ~20x training set (no 12B forwards).

Plans a large set of conversations by recombining compositional/violating CORES
with benign padding (mtlib.datagen.plan_recombinations), dedups, gathers latents
from the bank, teacher-verifies cross-policy swaps via bank logits, and writes
training shards directly. Eval splits stay the REAL v4 val/test — this only
grows TRAIN.

Usage
  python scripts/compose_from_bank.py \
      --convs outputs/data_v4/train.jsonl outputs/data_fam/train.jsonl \
      --bank /raid/.../bank_v5.pt --samples ../../output/.../accepted_samples.jsonl \
      --out /raid/frontiers_ashoka/BARRED-multiturn/latents_v5/train --target-convs 175000
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from mtlib.common import NEGATIVE_LABEL, POSITIVE_LABEL, SAMPLES_PATH, dump_json, setup_logging  # noqa: E402
from mtlib.datagen import (  # noqa: E402
    RecombineConfig,
    dedup_conversations,
    load_seed_rows,
    plan_recombinations,
)
from mtlib.dataset import shard_item_from_bank, write_shard  # noqa: E402
from mtlib.latent_bank import LatentBank, pair_key  # noqa: E402
from mtlib.schema import COMPOSITIONAL_SOURCES as COMP  # noqa: E402
from mtlib.schema import Turn, read_conversations, split_transcript, turn_block  # noqa: E402


def build_sources(conv_paths, samples_path, held_out, comp_only=False):
    """cores, benign_pool_by_policy, policy_of_guardrail from source convs+seeds.

    ``comp_only`` restricts the recontextualized cores to genuinely COMPOSITIONAL
    ones (natural_compositional / decompose). Single-turn violations (compose)
    still donate their benign prefix turns to the pool but are NOT recombined —
    flooding training with single-turn positives teaches the aggregator to rely
    on individual high-scoring turns (anti-compositional; measured to drop OOD
    comp AUROC 0.90 -> 0.61)."""
    cores = []
    benign_pool: dict[str, list[Turn]] = defaultdict(list)
    policy_gid: dict[str, str] = {}
    for path in conv_paths:
        for conv in read_conversations(path):
            policy_gid[conv.policy_prompt] = conv.guardrail_id
            if conv.label == POSITIVE_LABEL:
                is_comp = conv.source in COMP
                if is_comp or not comp_only:
                    cores.append(conv)
                # pre-onset turns of any violating core are benign under its policy.
                onset = conv.first_violation_turn or 0
                benign_pool[conv.policy_prompt].extend(conv.turns[:onset])
            else:
                benign_pool[conv.policy_prompt].extend(conv.turns)
    if samples_path:
        for r in load_seed_rows(Path(samples_path), held_out):
            policy_gid.setdefault(r["policy_prompt"], r["guardrail_id"])
            if r["label"] == NEGATIVE_LABEL:
                benign_pool[r["policy_prompt"]].extend(split_transcript(r["input_block"]))
    # de-dup benign turns per policy, keep those with real content
    for policy, turns in list(benign_pool.items()):
        uniq = {t.text.strip(): t for t in turns if len(t.text.strip()) >= 20}
        benign_pool[policy] = list(uniq.values())
    return cores, dict(benign_pool), policy_gid


def swap_is_benign(conv, bank: LatentBank, positive_id: int, thresh: float = 0.5) -> bool:
    """Cross-policy swap must be genuinely benign under the target policy: every
    turn's bank logit-prob below thresh (free — bank stores the logits)."""
    for turn in conv.turns:
        key = pair_key(conv.policy_prompt, turn_block(turn))
        if key not in bank.logits:
            return False
        p = torch.softmax(bank.logits[key].float(), dim=-1)[positive_id].item()
        if p >= thresh:
            return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--convs", nargs="+", required=True)
    ap.add_argument("--bank", required=True)
    ap.add_argument("--samples", default=str(SAMPLES_PATH))
    ap.add_argument("--held-out", default="")
    ap.add_argument("--out", required=True, help="output shard dir (train)")
    ap.add_argument("--target-convs", type=int, default=175000)
    ap.add_argument("--pos-frac", type=float, default=0.45, help="target positive fraction")
    ap.add_argument("--shard-size", type=int, default=2000)
    ap.add_argument("--jaccard", type=float, default=0.9)
    ap.add_argument("--comp-only", action="store_true",
                    help="recontextualize only compositional cores (skip single-turn violations)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    logger = setup_logging("compose_from_bank")
    import random

    rng = random.Random(args.seed)
    held_out = {x for x in args.held_out.split(",") if x}
    bank = LatentBank.load(args.bank)
    positive_id = bank.num_labels - 1

    cores, benign_pool, policy_gid = build_sources(
        args.convs, args.samples or None, held_out, comp_only=args.comp_only
    )
    n_comp = sum(c.source in COMP for c in cores)
    logger.info("sources: %d cores (%d compositional), %d policies with benign pools",
                len(cores), n_comp, len(benign_pool))

    # Size the knobs to hit target counts.
    target_pos = int(args.target_convs * args.pos_frac)
    target_neg = args.target_convs - target_pos
    cfg = RecombineConfig(
        recontext_per_core=max(1, target_pos // max(1, len(cores))),
        benign_per_policy=max(1, target_neg // max(1, len(benign_pool))),
        # Recomb swaps would need core turns encoded under other policies (absent
        # from the bank); real v4 policy-swap negatives cover this signal.
        swap_targets_per_core=0,
    )
    planned = plan_recombinations(cores, benign_pool, policy_gid, cfg, rng)
    planned, dstats = dedup_conversations(planned, jaccard_threshold=args.jaccard)

    # Assemble shards from the bank.
    out_dir = Path(args.out)
    shard, shard_idx = [], 0
    kept = skipped_missing = skipped_swap = 0
    counts = defaultdict(int)
    for conv in planned:
        if conv.source == "recomb_swap" and not swap_is_benign(conv, bank, positive_id):
            skipped_swap += 1
            continue
        item = shard_item_from_bank(conv, bank)
        if item is None:
            skipped_missing += 1
            continue
        shard.append(item)
        counts[conv.source] += 1
        kept += 1
        if len(shard) >= args.shard_size:
            write_shard(shard, out_dir / f"recomb_{shard_idx:04d}.pt")
            shard, shard_idx = [], shard_idx + 1
    if shard:
        write_shard(shard, out_dir / f"recomb_{shard_idx:04d}.pt")

    pos = sum(v for k, v in counts.items() if k.startswith("recomb_") and "benign" not in k and "swap" not in k)
    summary = {
        "kept": kept, "positives_est": pos, "by_source": dict(counts),
        "dedup": dstats, "skipped_missing_latents": skipped_missing,
        "skipped_nonbenign_swaps": skipped_swap, "config": vars(cfg),
    }
    dump_json(summary, out_dir / "recomb_summary.json")
    logger.info("wrote %d recombined convs -> %s | %s", kept, out_dir,
                {k: summary[k] for k in ("positives_est", "skipped_missing_latents", "skipped_nonbenign_swaps")})


if __name__ == "__main__":
    main()
