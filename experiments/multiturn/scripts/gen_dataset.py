#!/usr/bin/env python
"""Phase D1: build the multi-turn conversation dataset (all three buckets).

  1. compose        -- no generation (fast); pure recombination of seed rows.
  2. decompose      -- generator splits violations into distributed turns;
                       frozen classifier teacher-filters to the compositional gap.
  3. hard_negative  -- benign decoys (generated) + policy swaps (relabel).

Writes JSONL of mtlib.schema.Conversation rows, split into train/val/test by
guardrail so eval policies are unseen (mirrors the token-supervision OOD story).

Usage
  python scripts/gen_dataset.py --out outputs/data \
      --generator Qwen/Qwen3.5-9B --gpus 0,1,2,3 \
      --n-decomp 2 --n-decoy 1 --turns 6
Skip generation (compose-only smoke run):
  python scripts/gen_dataset.py --out outputs/data --no-generate
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # experiments/multiturn

logger = logging.getLogger("gen_dataset")

from mtlib.common import SAMPLES_PATH, dump_json, setup_logging  # noqa: E402
from mtlib.datagen import (  # noqa: E402
    DatagenConfig,
    accept_decompositions,
    build_compose_bucket,
    build_generation_specs,
    build_policy_swaps,
    load_seed_rows,
    sample_seeds_across_guardrails,
    scan_natural_compositional,
    validate_all,
)
from mtlib.encoder import TurnEncoder  # noqa: E402
from mtlib.schema import COMPOSITIONAL_SOURCES, Conversation, write_conversations  # noqa: E402


def split_by_guardrail(
    convs: list[Conversation], val_frac: float, test_frac: float, seed: int
) -> dict[str, list[Conversation]]:
    """Partition guardrail ids into train/val/test so no policy leaks across
    splits (the OOD story). Guardrails are stratified by whether they carry any
    compositional positives, so every split gets its share of the (scarce)
    compositional guardrails instead of leaving val/test without the one case
    that matters. Falls back to a per-conversation split when there are too few
    guardrails; the fallback is logged since it breaks policy-disjointness."""
    rng = random.Random(seed)
    comp_gids = {c.guardrail_id for c in convs if c.source in COMPOSITIONAL_SOURCES}
    gids = sorted({c.guardrail_id for c in convs})
    n_test = max(1, int(len(gids) * test_frac))
    n_val = max(1, int(len(gids) * val_frac))
    if len(gids) >= 3 and n_test + n_val < len(gids):
        test_ids: set[str] = set()
        val_ids: set[str] = set()
        # Deal each stratum round-robin in test/val/train proportion.
        for stratum in (sorted(comp_gids), sorted(set(gids) - comp_gids)):
            stratum = list(stratum)
            rng.shuffle(stratum)
            k_test = max(1, round(len(stratum) * test_frac)) if stratum else 0
            k_val = max(1, round(len(stratum) * val_frac)) if stratum else 0
            test_ids |= set(stratum[:k_test])
            val_ids |= set(stratum[k_test : k_test + k_val])
        splits: dict[str, list[Conversation]] = {"train": [], "val": [], "test": []}
        for conv in convs:
            if conv.guardrail_id in test_ids:
                splits["test"].append(conv)
            elif conv.guardrail_id in val_ids:
                splits["val"].append(conv)
            else:
                splits["train"].append(conv)
        return splits

    logger.warning(
        "only %d guardrails — falling back to a per-conversation split "
        "(policies are NOT disjoint across splits; eval is not OOD)",
        len(gids),
    )
    shuffled = list(convs)
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    return {
        "test": shuffled[:n_test],
        "val": shuffled[n_test : n_test + n_val],
        "train": shuffled[n_test + n_val :],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output dir for {train,val,test}.jsonl")
    ap.add_argument("--samples", default=str(SAMPLES_PATH))
    ap.add_argument("--generator", default="Qwen/Qwen3.5-9B")
    ap.add_argument("--gpus", default="0")
    ap.add_argument("--encoder-ckpt", default=None, help="turn-encoder checkpoint (default: common.CHECKPOINT)")
    ap.add_argument("--held-out", default="", help="comma-separated guardrail ids excluded from seeding")
    ap.add_argument("--n-decomp", type=int, default=2)
    ap.add_argument("--n-decoy", type=int, default=1)
    ap.add_argument("--turns", type=int, default=6)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--turn-max", type=float, default=0.5)
    ap.add_argument("--concat-min", type=float, default=0.5)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-generate", action="store_true", help="skip the LLM decompose/decoy buckets")
    ap.add_argument("--scan-natural", action="store_true",
                    help="harvest naturally-compositional violating seeds (generation-free)")
    ap.add_argument("--scan-seeds", type=int, default=0,
                    help="violating-seed pool to scan for natural compositional (0=all)")
    ap.add_argument("--limit-seeds", type=int, default=0, help="cap seed rows for a quick run (0=all)")
    args = ap.parse_args()

    logger = setup_logging("gen_dataset")
    rng = random.Random(args.seed)
    cfg = DatagenConfig(
        turn_max=args.turn_max,
        concat_min=args.concat_min,
        decoy_max=args.turn_max,
        turns_per_decomposition=args.turns,
        n_decompositions_per_seed=args.n_decomp,
        n_decoys_per_seed=args.n_decoy,
        seed=args.seed,
    )

    held_out = {x for x in args.held_out.split(",") if x}
    all_seed_rows = load_seed_rows(Path(args.samples), held_out)
    seed_rows = (
        sample_seeds_across_guardrails(all_seed_rows, args.limit_seeds, rng)
        if args.limit_seeds else all_seed_rows
    )
    n_gids = len({r["guardrail_id"] for r in seed_rows})
    logger.info(
        "loaded %d seed rows across %d guardrails (%d held-out gids)",
        len(seed_rows), n_gids, len(held_out),
    )

    all_convs: list[Conversation] = []

    # Bucket 1: compose (no model needed).
    all_convs += build_compose_bucket(seed_rows, cfg, rng)

    need_encoder = not args.no_generate or args.scan_natural
    encoder = TurnEncoder(model_dir=args.encoder_ckpt or None) if need_encoder else None

    # Bucket 0: naturally-compositional seeds (generation-free). Scanned over a
    # large violating-seed pool since it's cheap and compositional positives are
    # rare; these need no generator.
    if args.scan_natural:
        scan_pool = (
            sample_seeds_across_guardrails(all_seed_rows, args.scan_seeds, rng)
            if args.scan_seeds else all_seed_rows
        )
        natural, nat_stats = scan_natural_compositional(scan_pool, encoder, cfg)
        dump_json(nat_stats, Path(args.out) / "natural_scan_stats.json")
        all_convs += natural

    # Buckets 2 & 3 need the frozen classifier (encoder + teacher) + generation.
    if not args.no_generate:
        from rl.hf_generate import generate_distributed, load_tokenizer

        tokenizer = load_tokenizer(args.generator)
        specs = build_generation_specs(seed_rows, cfg, tokenizer, rng)
        gpus = [int(g) for g in args.gpus.split(",") if g != ""]
        raw = generate_distributed(
            args.generator,
            [s.prompt for s in specs],
            gpus=gpus,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            do_sample=args.temperature > 0,
            seed=args.seed,
            batch_size=args.batch_size,
        )
        generated, diag = accept_decompositions(specs, raw, encoder, cfg)
        dump_json(diag, Path(args.out) / "teacher_filter_diagnostics.json")
        all_convs += generated

    # Bucket 3b: policy swaps from every compositional positive (generated + natural).
    if encoder is not None:
        positives = [c for c in all_convs if c.source in COMPOSITIONAL_SOURCES]
        swaps, swap_diag = build_policy_swaps(positives, seed_rows, encoder, cfg, rng)
        dump_json(swap_diag, Path(args.out) / "policy_swap_diagnostics.json")
        all_convs += swaps

    all_convs = validate_all(all_convs)
    splits = split_by_guardrail(all_convs, args.val_frac, args.test_frac, args.seed)

    out_dir = Path(args.out)
    summary = {"config": vars(args), "counts": {}, "by_source": {}}
    for name, convs in splits.items():
        write_conversations(convs, out_dir / f"{name}.jsonl")
        summary["counts"][name] = len(convs)
        by_src: dict[str, int] = {}
        for conv in convs:
            by_src[conv.source] = by_src.get(conv.source, 0) + 1
        summary["by_source"][name] = by_src
    dump_json(summary, out_dir / "dataset_summary.json")
    logger.info("wrote splits to %s: %s", out_dir, summary["counts"])


if __name__ == "__main__":
    main()
