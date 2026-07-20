#!/usr/bin/env python
"""Phase D2: freeze the backbone once, dump per-conversation latents to shards.

The 12B is frozen, so every aggregator training run reads these shards instead
of re-encoding. Latents are fp16 (~8 KB/turn). Big shards go under /raid.

Usage
  python scripts/precompute_latents.py \
      --data outputs/data --split train \
      --out /raid/frontiers_ashoka/BARRED-multiturn/latents/train \
      --shard-size 512
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from mtlib.common import RAID_OUT, setup_logging  # noqa: E402
from mtlib.dataset import shard_item, write_shard  # noqa: E402
from mtlib.encoder import TurnEncoder  # noqa: E402
from mtlib.schema import read_conversations  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir with {split}.jsonl, or a single .jsonl")
    ap.add_argument("--split", default="train")
    ap.add_argument("--out", default=None, help="shard output dir (default: RAID/latents/{split})")
    ap.add_argument("--encoder-ckpt", default=None)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--shard-size", type=int, default=512, help="conversations per shard")
    ap.add_argument("--limit", type=int, default=0, help="cap conversations (0=all)")
    args = ap.parse_args()

    logger = setup_logging("precompute_latents")
    data_path = Path(args.data)
    conv_path = data_path if data_path.suffix == ".jsonl" else data_path / f"{args.split}.jsonl"
    out_dir = Path(args.out) if args.out else RAID_OUT / "latents" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    encoder = TurnEncoder(model_dir=args.encoder_ckpt or None, dtype=args.dtype)
    convs = list(read_conversations(conv_path))
    if args.limit:
        convs = convs[: args.limit]
    logger.info("encoding %d conversations from %s", len(convs), conv_path)

    shard: list = []
    shard_idx = 0
    n_turns = 0
    for i, conv in enumerate(convs):
        latents = encoder.encode_conversation(conv.policy_prompt, conv.turns)
        shard.append(shard_item(conv, latents))
        n_turns += len(conv.turns)
        if len(shard) >= args.shard_size:
            path = out_dir / f"shard_{shard_idx:04d}.pt"
            write_shard(shard, path)
            logger.info("wrote %s (%d convs); %d/%d done", path.name, len(shard), i + 1, len(convs))
            shard, shard_idx = [], shard_idx + 1
    if shard:
        path = out_dir / f"shard_{shard_idx:04d}.pt"
        write_shard(shard, path)
        logger.info("wrote %s (%d convs)", path.name, len(shard))

    logger.info(
        "done: %d conversations, %d turns, hidden=%d, %d shards -> %s",
        len(convs), n_turns, encoder.hidden_size, shard_idx + 1, out_dir,
    )


if __name__ == "__main__":
    main()
