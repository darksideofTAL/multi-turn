#!/usr/bin/env python
"""Phase v5-1: build a latent bank from source conversations + seed rows.

Encodes every unique (policy, turn_block) pair and each policy latent ONCE with
the frozen 12B, so downstream recombination (compose_from_bank.py) can build
unlimited conversations with no further 12B forwards.

Sources: any number of conversation JSONLs (mtlib.schema) plus optional
accepted-samples seed rows (their turns become the benign padding vocabulary).

Usage
  CUDA_VISIBLE_DEVICES=2 python scripts/build_latent_bank.py \
      --convs outputs/data_v4/train.jsonl outputs/data_fam/train.jsonl \
      --samples ../../output/generation/policies_v4/accepted_samples.jsonl \
      --out /raid/frontiers_ashoka/BARRED-multiturn/bank_v5.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from mtlib.common import SAMPLES_PATH, setup_logging  # noqa: E402
from mtlib.datagen import load_seed_rows  # noqa: E402
from mtlib.encoder import TurnEncoder  # noqa: E402
from mtlib.latent_bank import LatentBank  # noqa: E402
from mtlib.schema import read_conversations, split_transcript, turn_block  # noqa: E402


def collect_pairs(conv_paths, samples_path, held_out):
    """Unique (policy, block) pairs + policies from all sources."""
    pairs: list[tuple[str, str]] = []
    policies: set[str] = set()
    seen: set[tuple[str, str]] = set()

    def add(policy, block):
        k = (policy.strip(), block.strip())
        if k not in seen:
            seen.add(k)
            pairs.append((policy, block))

    for path in conv_paths:
        for conv in read_conversations(path):
            policies.add(conv.policy_prompt.strip())
            for t in conv.turns:
                add(conv.policy_prompt, turn_block(t))
    if samples_path:
        for r in load_seed_rows(Path(samples_path), held_out):
            policies.add(r["policy_prompt"].strip())
            for t in split_transcript(r["input_block"]):
                add(r["policy_prompt"], turn_block(t))
    return pairs, policies


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--convs", nargs="+", required=True, help="conversation JSONLs")
    ap.add_argument("--samples", default=str(SAMPLES_PATH), help="accepted_samples for benign vocab (or '')")
    ap.add_argument("--held-out", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--encoder-ckpt", default=None)
    args = ap.parse_args()

    logger = setup_logging("build_latent_bank")
    held_out = {x for x in args.held_out.split(",") if x}
    pairs, policies = collect_pairs(args.convs, args.samples or None, held_out)
    logger.info("collected %d unique pairs, %d policies", len(pairs), len(policies))

    encoder = TurnEncoder(model_dir=args.encoder_ckpt or None)
    bank = LatentBank(encoder.hidden_size, encoder.num_labels)
    stats = bank.fill(encoder, pairs, policies)
    bank.save(args.out)
    logger.info("bank saved %s: %s", args.out, stats)


if __name__ == "__main__":
    main()
