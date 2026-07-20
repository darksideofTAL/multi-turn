"""Latent shard IO + torch Dataset/collate for aggregator training.

A shard is a torch.save'd list of per-conversation dicts (latents stored fp16
to halve disk/RAM; ~8 KB/turn). The backbone is frozen, so latents are computed
once (scripts/precompute_latents.py) and every training run is minutes of pure
aggregator work.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset

from mtlib.aggregator import IGNORE_INDEX
from mtlib.encoder import ConversationLatents
from mtlib.schema import Conversation


def shard_item(conv: Conversation, latents: ConversationLatents) -> dict[str, Any]:
    return {
        "conv_id": conv.conv_id,
        "guardrail_id": conv.guardrail_id,
        "label": conv.label,
        "first_violation_turn": conv.first_violation_turn,
        "source": conv.source,
        "policy_latent": latents.policy_latent.to(torch.float16),
        "turn_latents": latents.turn_latents.to(torch.float16),
        "turn_logits": latents.turn_logits.to(torch.float16),
        "role_ids": latents.role_ids.to(torch.int8),
        "per_turn_labels": torch.tensor(conv.per_turn_labels(), dtype=torch.int8),
    }


def shard_item_from_bank(conv: Conversation, bank) -> dict[str, Any] | None:
    """Assemble a shard item by GATHERING per-turn latents from a LatentBank
    instead of running the 12B. Returns None if any (policy, turn) pair or the
    policy latent is missing from the bank (caller skips it)."""
    from mtlib.schema import ROLE_IDS, turn_block

    policy = conv.policy_prompt
    if policy.strip() not in bank.policy_latents:
        return None
    turn_latents, turn_logits, role_ids = [], [], []
    for turn in conv.turns:
        block = turn_block(turn)
        if not bank.has(policy, block):
            return None
        lat, log = bank.get(policy, block)
        turn_latents.append(lat)
        turn_logits.append(log)
        role_ids.append(ROLE_IDS[turn.role])
    return {
        "conv_id": conv.conv_id,
        "guardrail_id": conv.guardrail_id,
        "label": conv.label,
        "first_violation_turn": conv.first_violation_turn,
        "source": conv.source,
        "policy_latent": bank.get_policy(policy),
        "turn_latents": torch.stack(turn_latents),
        "turn_logits": torch.stack(turn_logits),
        "role_ids": torch.tensor(role_ids, dtype=torch.int8),
        "per_turn_labels": torch.tensor(conv.per_turn_labels(), dtype=torch.int8),
    }


def write_shard(items: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(items, path)


def read_shard(path: str | Path) -> list[dict[str, Any]]:
    return torch.load(path, map_location="cpu", weights_only=False)


def shard_paths(roots: str) -> list[Path]:
    """Comma-separated dirs and/or .pt files -> flat sorted shard list."""
    out: list[Path] = []
    for root in roots.split(","):
        p = Path(root.strip())
        out.extend(sorted(p.glob("*.pt")) if p.is_dir() else [p])
    if not out:
        raise ValueError(f"no shards under {roots!r}")
    return out


class LatentConversationDataset(Dataset):
    def __init__(
        self,
        shard_paths: list[str | Path],
        oversample: dict[str, int] | None = None,
    ) -> None:
        """``oversample`` maps a datagen source tag to a repeat factor, e.g.
        {"natural_compositional": 4} — used on the TRAIN split to keep the
        scarce compositional pattern from being drowned out by compose rows."""
        self.items: list[dict[str, Any]] = []
        for path in shard_paths:
            for item in read_shard(path):
                repeat = (oversample or {}).get(item.get("source", ""), 1)
                self.items.extend([item] * max(1, repeat))
        if not self.items:
            raise ValueError(f"No conversations in shards: {shard_paths}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = self.items[index]
        onset = item["first_violation_turn"]
        return {
            "conv_id": item["conv_id"],
            "policy_latent": item["policy_latent"].float(),
            "turn_latents": item["turn_latents"].float(),
            "turn_logits": item["turn_logits"].float(),
            "role_ids": item["role_ids"].long(),
            "labels": item["per_turn_labels"].long(),
            "onset": -1 if onset is None else int(onset),
            "conv_label": int(item["per_turn_labels"].max().item()),
        }


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Right-pad variable-length conversations; labels padded with IGNORE_INDEX."""
    max_turns = max(item["turn_latents"].shape[0] for item in batch)
    hidden = batch[0]["turn_latents"].shape[1]
    num_labels = batch[0]["turn_logits"].shape[1]
    size = len(batch)

    turn_latents = torch.zeros(size, max_turns, hidden)
    turn_logits = torch.zeros(size, max_turns, num_labels)
    role_ids = torch.zeros(size, max_turns, dtype=torch.long)
    attention_mask = torch.zeros(size, max_turns, dtype=torch.long)
    labels = torch.full((size, max_turns), IGNORE_INDEX, dtype=torch.long)
    policy_latent = torch.zeros(size, hidden)
    onset = torch.empty(size, dtype=torch.long)
    conv_label = torch.empty(size, dtype=torch.long)

    for i, item in enumerate(batch):
        turns = item["turn_latents"].shape[0]
        turn_latents[i, :turns] = item["turn_latents"]
        turn_logits[i, :turns] = item["turn_logits"]
        role_ids[i, :turns] = item["role_ids"]
        attention_mask[i, :turns] = 1
        labels[i, :turns] = item["labels"]
        policy_latent[i] = item["policy_latent"]
        onset[i] = item["onset"]
        conv_label[i] = item["conv_label"]

    return {
        "conv_ids": [item["conv_id"] for item in batch],
        "turn_latents": turn_latents,
        "turn_logits": turn_logits,
        "role_ids": role_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "policy_latent": policy_latent,
        "onset": onset,
        "conv_label": conv_label,
    }
