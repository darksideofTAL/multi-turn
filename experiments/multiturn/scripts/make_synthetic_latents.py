#!/usr/bin/env python
"""Fabricate synthetic latent shards with a KNOWN compositional signal.

For smoke-testing the aggregator train/eval loop without the 12B backbone. A
conversation violates iff the running sum of a hidden "evidence" coordinate
crosses a threshold — a genuinely CUMULATIVE signal no single turn reveals, so a
working aggregator must beat max-over-turns. Latents are otherwise random.

Usage
  python scripts/make_synthetic_latents.py --out /tmp/mt_synth --n 400 --hidden 64
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from mtlib.dataset import write_shard  # noqa: E402


def make_split(n: int, hidden: int, num_labels: int, gen: torch.Generator) -> list[dict]:
    items = []
    for i in range(n):
        n_turns = int(torch.randint(3, 10, (1,), generator=gen).item())
        latents = torch.randn(n_turns, hidden, generator=gen)
        # Evidence coordinate 0: small per-turn increments; violation when the
        # cumulative sum first exceeds threshold. No single turn is decisive.
        evidence = torch.rand(n_turns, generator=gen) * 0.6
        latents[:, 0] = evidence
        cumulative = torch.cumsum(evidence, dim=0)
        crossed = (cumulative >= 1.5).nonzero()
        onset = int(crossed[0].item()) if crossed.numel() else None
        label = "True" if onset is not None else "False"
        per_turn = [int(onset is not None and t >= onset) for t in range(n_turns)]
        logits = torch.zeros(n_turns, num_labels)
        logits[:, 1] = latents[:, 0] - 0.3  # weak per-turn signal (max-over-turns baseline)
        role_ids = torch.tensor([1 if t % 2 == 0 else 2 for t in range(n_turns)], dtype=torch.int8)
        items.append(
            {
                "conv_id": f"synth-{i}",
                "guardrail_id": f"G{i % 8}",
                "label": label,
                "first_violation_turn": onset,
                "source": "synthetic",
                "policy_latent": torch.randn(hidden, generator=gen).to(torch.float16),
                "turn_latents": latents.to(torch.float16),
                "turn_logits": logits.to(torch.float16),
                "role_ids": role_ids,
                "per_turn_labels": torch.tensor(per_turn, dtype=torch.int8),
            }
        )
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--num-labels", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.out)
    gen = torch.Generator().manual_seed(args.seed)
    for split, n in (("train", args.n), ("val", args.n // 4), ("test", args.n // 4)):
        items = make_split(n, args.hidden, args.num_labels, gen)
        write_shard(items, out / split / "shard_0000.pt")
        pos = sum(it["label"] == "True" for it in items)
        print(f"{split}: {n} convs ({pos} positive) -> {out / split}")


if __name__ == "__main__":
    main()
