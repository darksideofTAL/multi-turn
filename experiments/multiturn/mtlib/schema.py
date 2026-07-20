"""Conversation data model + transcript parsing.

A Conversation is a policy plus an ordered list of role-tagged turns, with a
conversation-level label and (for violating conversations) the earliest turn
index at which the conversation-so-far violates the policy. Per-turn labels are
monotone: once a conversation has violated, it stays violated.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator

from mtlib.common import NEGATIVE_LABEL, POSITIVE_LABEL

ROLES = ("user", "agent")
ROLE_TAGS = {"user": "User", "agent": "Agent"}
# Aggregator role-embedding ids; 0 is reserved for the policy token.
ROLE_IDS = {"policy": 0, "user": 1, "agent": 2}

# Conversation sources whose positives are GENUINELY compositional (no single
# turn violates); the compositional slice, oversampling, and recombination all
# key on these. Recombined variants are prefixed "recomb_".
COMPOSITIONAL_SOURCES = ("natural_compositional", "decompose")

# Turns are delimited by OPENING tags only: real generated transcripts contain
# mismatched closing tags (e.g. an <Agent> turn closed with </User>), so closing
# tags are stripped rather than matched.
_OPEN_TAG_RE = re.compile(r"<(User|Agent)>", re.IGNORECASE)
_CLOSE_TAG_RE = re.compile(r"</(User|Agent)>", re.IGNORECASE)


@dataclass
class Turn:
    role: str  # "user" | "agent"
    text: str


@dataclass
class Conversation:
    conv_id: str
    guardrail_id: str
    policy_prompt: str
    turns: list[Turn]
    label: str  # POSITIVE_LABEL (violates) | NEGATIVE_LABEL
    # Earliest t such that turns[:t+1] violates the policy; None iff benign.
    first_violation_turn: int | None = None
    source: str = ""  # datagen bucket tag, e.g. "compose", "decompose", "policy_swap"
    meta: dict[str, Any] = field(default_factory=dict)

    def per_turn_labels(self) -> list[int]:
        """Monotone per-turn conversation-status labels: y_t = 1 for every
        t >= first_violation_turn."""
        if self.first_violation_turn is None:
            return [0] * len(self.turns)
        return [int(t >= self.first_violation_turn) for t in range(len(self.turns))]

    def validate(self) -> None:
        if self.label not in (POSITIVE_LABEL, NEGATIVE_LABEL):
            raise ValueError(f"{self.conv_id}: bad label {self.label!r}")
        if not self.turns:
            raise ValueError(f"{self.conv_id}: empty conversation")
        for turn in self.turns:
            if turn.role not in ROLES:
                raise ValueError(f"{self.conv_id}: bad role {turn.role!r}")
            if not turn.text.strip():
                raise ValueError(f"{self.conv_id}: empty turn text")
        violating = self.label == POSITIVE_LABEL
        if violating != (self.first_violation_turn is not None):
            raise ValueError(
                f"{self.conv_id}: label {self.label} inconsistent with "
                f"first_violation_turn={self.first_violation_turn}"
            )
        if self.first_violation_turn is not None and not (
            0 <= self.first_violation_turn < len(self.turns)
        ):
            raise ValueError(
                f"{self.conv_id}: first_violation_turn {self.first_violation_turn} "
                f"out of range for {len(self.turns)} turns"
            )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "Conversation":
        return cls(
            conv_id=row["conv_id"],
            guardrail_id=row["guardrail_id"],
            policy_prompt=row["policy_prompt"],
            turns=[Turn(role=t["role"], text=t["text"]) for t in row["turns"]],
            label=row["label"],
            first_violation_turn=row.get("first_violation_turn"),
            source=row.get("source", ""),
            meta=row.get("meta", {}),
        )


def new_conv_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def split_transcript(block: str) -> list[Turn]:
    """Split a <User>/<Agent>-tagged transcript into turns.

    Tolerant by construction: splits on opening tags, strips all closing tags,
    drops empty segments. Text before the first opening tag is discarded."""
    turns: list[Turn] = []
    matches = list(_OPEN_TAG_RE.finditer(block))
    for i, m in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(block)
        text = _CLOSE_TAG_RE.sub("", block[m.end() : end]).strip()
        if text:
            turns.append(Turn(role=m.group(1).lower(), text=text))
    return turns


def turn_block(turn: Turn) -> str:
    """Render one turn as the input_block the frozen classifier sees. Tagged the
    same way training transcripts are, so the encoder stays in-distribution."""
    tag = ROLE_TAGS[turn.role]
    return f"<{tag}>{turn.text}</{tag}>"


def transcript(turns: list[Turn]) -> str:
    """Full tagged transcript (the full-concat baseline / prefix-labeling input)."""
    return "\n".join(turn_block(t) for t in turns)


def read_conversations(path: str | Path) -> Iterator[Conversation]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield Conversation.from_dict(json.loads(line))


def write_conversations(convs: list[Conversation], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for conv in convs:
            f.write(json.dumps(conv.to_dict()) + "\n")
    return len(convs)
