"""Latent bank: encode each unique (policy, turn_block) pair ONCE, reuse forever.

Turn latents depend only on the (policy, turn_text) pair, and composed
conversations reuse the same turns in many arrangements. So instead of encoding
every conversation, we encode the unique pairs and recombine cached latents into
unlimited conversations with NO extra 12B forwards — the near-free 20× data
multiplier.

Keyed by a stable hash of (policy, block). Stores fp16 latent + logits per pair,
plus one policy latent per policy (block = the empty sentinel the encoder uses).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable

import torch

from mtlib.encoder import POLICY_ONLY_BLOCK, TurnEncoder


def pair_key(policy: str, block: str) -> str:
    h = hashlib.sha1()
    h.update(policy.strip().encode())
    h.update(b"\x00")
    h.update(block.strip().encode())
    return h.hexdigest()


class LatentBank:
    """(policy, block) -> (latent fp16 [H], logits fp16 [num_labels])."""

    def __init__(self, hidden_size: int, num_labels: int) -> None:
        self.hidden_size = hidden_size
        self.num_labels = num_labels
        self.latents: dict[str, torch.Tensor] = {}
        self.logits: dict[str, torch.Tensor] = {}
        self.policy_latents: dict[str, torch.Tensor] = {}  # policy text -> [H] fp16

    def __contains__(self, key: str) -> bool:
        return key in self.latents

    def has(self, policy: str, block: str) -> bool:
        return pair_key(policy, block) in self.latents

    def get(self, policy: str, block: str) -> tuple[torch.Tensor, torch.Tensor]:
        key = pair_key(policy, block)
        return self.latents[key], self.logits[key]

    def get_policy(self, policy: str) -> torch.Tensor:
        return self.policy_latents[policy.strip()]

    def missing(self, pairs: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
        out, seen = [], set()
        for policy, block in pairs:
            key = pair_key(policy, block)
            if key not in self.latents and key not in seen:
                seen.add(key)
                out.append((policy, block))
        return out

    def fill(
        self,
        encoder: TurnEncoder,
        pairs: Iterable[tuple[str, str]],
        policies: Iterable[str] = (),
        batch: int = 256,
    ) -> dict[str, int]:
        """Encode any not-yet-present (policy, block) pairs and policy latents."""
        todo = self.missing(pairs)
        added = 0
        for i in range(0, len(todo), batch):
            chunk = todo[i : i + batch]
            latents, logits = encoder.encode([p for p, _ in chunk], [b for _, b in chunk])
            for (policy, block), lat, log in zip(chunk, latents, logits):
                key = pair_key(policy, block)
                self.latents[key] = lat.to(torch.float16)
                self.logits[key] = log.to(torch.float16)
                added += 1

        pol_todo = [p.strip() for p in policies if p.strip() not in self.policy_latents]
        pol_todo = list(dict.fromkeys(pol_todo))
        for i in range(0, len(pol_todo), batch):
            chunk = pol_todo[i : i + batch]
            latents, _ = encoder.encode(chunk, [POLICY_ONLY_BLOCK] * len(chunk))
            for policy, lat in zip(chunk, latents):
                self.policy_latents[policy] = lat.to(torch.float16)

        return {"pairs_added": added, "policies_added": len(pol_todo), "total_pairs": len(self.latents)}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "hidden_size": self.hidden_size,
                "num_labels": self.num_labels,
                "latents": self.latents,
                "logits": self.logits,
                "policy_latents": self.policy_latents,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "LatentBank":
        payload = torch.load(path, map_location="cpu", weights_only=False)
        bank = cls(payload["hidden_size"], payload["num_labels"])
        bank.latents = payload["latents"]
        bank.logits = payload["logits"]
        bank.policy_latents = payload["policy_latents"]
        return bank
