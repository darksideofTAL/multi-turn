#!/usr/bin/env python
"""Build a LARGE guardrail-disjoint compositional eval from the pooled cores.

The v4 test had only 27 compositional positives — too few to distinguish
detection rates (Wilson CIs overlap 26%..59%). Now that ~864 cores exist across
~94 guardrails, hold out whole guardrails for val/test to get 100+ positives.

Latents for every core turn are already in the latent bank, so val/test shards
are GATHERED from the bank (no 12B). Also writes the TRAIN core JSONLs for two
variants — "small" (v4+v6) and "all" (v4+v6+v7) — restricted to train
guardrails, so compose_from_bank can build matched light-recomb train sets and
the only difference is core count.

Usage
  python scripts/build_core_eval.py --bank /raid/.../bank_v7.pt \
      --out-data outputs/data_cscale --out-lat /raid/.../cscale
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from mtlib.common import dump_json, setup_logging  # noqa: E402
from mtlib.datagen import RecombineConfig, plan_recombinations  # noqa: E402
from mtlib.dataset import shard_item_from_bank, write_shard  # noqa: E402
from mtlib.latent_bank import LatentBank  # noqa: E402
from mtlib.schema import COMPOSITIONAL_SOURCES as COMP  # noqa: E402
from mtlib.schema import read_conversations, write_conversations  # noqa: E402

# (tag, files) — v4 cores live mixed inside data_v4 splits; v6/v7 in *_cores.
SOURCES = {
    "v4": ["outputs/data_v4/train.jsonl", "outputs/data_v4/val.jsonl", "outputs/data_v4/test.jsonl"],
    "v6": ["outputs/data_v6_cores/train.jsonl", "outputs/data_v6_cores/val.jsonl", "outputs/data_v6_cores/test.jsonl"],
    "v7": ["outputs/data_v7_cores/train.jsonl", "outputs/data_v7_cores/val.jsonl", "outputs/data_v7_cores/test.jsonl"],
}


def load_cores():
    """All compositional cores tagged by origin, de-duplicated by conv_id."""
    cores, seen = [], set()
    for tag, files in SOURCES.items():
        for f in files:
            if not Path(f).exists():
                continue
            for c in read_conversations(f):
                if c.source in COMP and c.conv_id not in seen:
                    seen.add(c.conv_id)
                    c.meta = {**c.meta, "origin": tag}
                    cores.append(c)
    return cores


def benign_pool_by_policy(cores):
    """Pre-onset turns of cores serve as benign padding under each policy."""
    pool = defaultdict(list)
    for c in cores:
        onset = c.first_violation_turn or 0
        pool[c.policy_prompt].extend(c.turns[:onset])
    for p, turns in list(pool.items()):
        uniq = {t.text.strip(): t for t in turns if len(t.text.strip()) >= 20}
        pool[p] = list(uniq.values())
    return dict(pool)


def build_split_shards(cores, gids, bank, policy_gid, rng, benign_mult=2):
    """Real compositional positives (gathered from bank, no recomb) + benign
    negatives (light recomb of the same gids' benign pools) for a held-out split."""
    split_cores = [c for c in cores if c.guardrail_id in gids]
    pool = benign_pool_by_policy(split_cores)
    convs, items = [], []
    for c in split_cores:
        it = shard_item_from_bank(c, bank)
        if it is not None:
            items.append(it)
            convs.append(c)
    # benign negatives from these gids
    cfg = RecombineConfig(benign_per_policy=benign_mult * max(1, len(split_cores) // max(1, len(pool))),
                          recontext_per_core=0, swap_targets_per_core=0)
    benign = [c for c in plan_recombinations([], pool, policy_gid, cfg, rng) if c.label == "False"]
    for c in benign:
        it = shard_item_from_bank(c, bank)
        if it is not None:
            items.append(it)
            convs.append(c)
    return convs, items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bank", required=True)
    ap.add_argument("--out-data", required=True)
    ap.add_argument("--out-lat", required=True)
    ap.add_argument("--val-gid-frac", type=float, default=0.2)
    ap.add_argument("--test-gid-frac", type=float, default=0.2)
    ap.add_argument("--min-cores-per-gid", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    logger = setup_logging("build_core_eval")
    rng = random.Random(args.seed)
    bank = LatentBank.load(args.bank)
    cores = load_cores()
    policy_gid = {c.policy_prompt: c.guardrail_id for c in cores}

    by_gid = defaultdict(list)
    for c in cores:
        by_gid[c.guardrail_id].append(c)
    # only hold out guardrails with enough cores to matter for eval
    eval_gids = sorted(g for g, cs in by_gid.items() if len(cs) >= args.min_cores_per_gid)
    rng.shuffle(eval_gids)
    n_test = max(1, int(len(eval_gids) * args.test_gid_frac))
    n_val = max(1, int(len(eval_gids) * args.val_gid_frac))
    test_gids = set(eval_gids[:n_test])
    val_gids = set(eval_gids[n_test:n_test + n_val])
    train_gids = {c.guardrail_id for c in cores} - test_gids - val_gids
    logger.info("guardrail split: %d train, %d val, %d test", len(train_gids), len(val_gids), len(test_gids))

    out_data, out_lat = Path(args.out_data), Path(args.out_lat)
    summary = {}
    for name, gids in (("val", val_gids), ("test", test_gids)):
        convs, items = build_split_shards(cores, gids, bank, policy_gid, rng)
        write_conversations(convs, out_data / f"{name}.jsonl")
        write_shard(items, out_lat / name / "shard_0000.pt")
        pos = sum(c.label == "True" for c in convs)
        summary[name] = {"convs": len(convs), "positives": pos, "guardrails": len(gids)}

    # TRAIN core JSONLs (train guardrails only), two variants.
    train_cores = [c for c in cores if c.guardrail_id in train_gids]
    small = [c for c in train_cores if c.meta.get("origin") in ("v4", "v6")]
    write_conversations(train_cores, out_data / "cores_all.jsonl")
    write_conversations(small, out_data / "cores_small.jsonl")
    summary["train_cores"] = {"all": len(train_cores), "small": len(small),
                              "all_gids": len({c.guardrail_id for c in train_cores})}
    dump_json(summary, out_data / "core_eval_summary.json")
    logger.info("summary: %s", summary)


if __name__ == "__main__":
    main()
