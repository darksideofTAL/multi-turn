"""Streaming multi-turn monitor.

Per-conversation state is just the policy latent plus one latent (+ head
logits) per turn — ~8 KB/turn. Each feed costs exactly one single-turn encode
(what a single-turn deploy already pays) plus a sub-millisecond aggregator pass
over the cached latents. Nothing persists inside the 12B between turns, so
none of the KV-cache surgery from the v1 streaming classifier exists here.

State is exportable (plain tensors), so a stateless server can hand it back to
the client between turns.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

import torch

from mtlib.aggregator import ConversationAggregator
from mtlib.encoder import TurnEncoder
from mtlib.schema import ROLE_IDS, Turn, turn_block


@dataclass
class MonitorState:
    policy: str
    policy_latent: torch.Tensor  # [H]
    turn_latents: list[torch.Tensor] = field(default_factory=list)  # each [H]
    turn_logits: list[torch.Tensor] = field(default_factory=list)  # each [num_labels]
    role_ids: list[int] = field(default_factory=list)

    def export(self) -> dict[str, Any]:
        return {
            "policy": self.policy,
            "policy_latent": self.policy_latent,
            "turn_latents": torch.stack(self.turn_latents) if self.turn_latents else None,
            "turn_logits": torch.stack(self.turn_logits) if self.turn_logits else None,
            "role_ids": list(self.role_ids),
        }

    @classmethod
    def from_export(cls, payload: dict[str, Any]) -> "MonitorState":
        state = cls(policy=payload["policy"], policy_latent=payload["policy_latent"])
        if payload["turn_latents"] is not None:
            state.turn_latents = list(payload["turn_latents"])
            state.turn_logits = list(payload["turn_logits"])
            state.role_ids = list(payload["role_ids"])
        return state


class MultiTurnMonitor:
    def __init__(
        self,
        encoder: TurnEncoder,
        aggregator: ConversationAggregator,
        tau: float = 0.5,
        device: str = "cpu",
    ) -> None:
        self.encoder = encoder
        self.device = torch.device(device)
        self.aggregator = aggregator.to(self.device).eval()
        self.tau = tau
        self.positive_id = encoder.positive_id
        self._states: dict[str, MonitorState] = {}

    # ------------------------------------------------------------------ session
    def start(self, policy: str, conv_id: str | None = None) -> str:
        conv_id = conv_id or uuid.uuid4().hex[:16]
        if conv_id in self._states:
            raise ValueError(f"conversation {conv_id} already active")
        self._states[conv_id] = MonitorState(
            policy=policy, policy_latent=self.encoder.encode_policy(policy)
        )
        return conv_id

    def feed(self, conv_id: str, role: str, text: str) -> dict[str, Any]:
        """Add one turn; returns the single-turn verdict for the new turn and the
        updated conversation-so-far verdict."""
        state = self._states[conv_id]
        latents, logits = self.encoder.encode(
            [state.policy], [turn_block(Turn(role=role, text=text))]
        )
        state.turn_latents.append(latents[0])
        state.turn_logits.append(logits[0])
        state.role_ids.append(ROLE_IDS[role])

        conv_probs = self._conversation_probs(state)
        turn_prob = float(self.encoder.violation_probs(logits[0]).item())
        conv_prob = conv_probs[-1]
        return {
            "conv_id": conv_id,
            "turn_index": len(state.turn_latents) - 1,
            "role": role,
            "turn_violation_prob": turn_prob,
            "conversation_violation_prob": conv_prob,
            "per_turn_conversation_probs": conv_probs,
            "flagged": conv_prob >= self.tau,
            "tau": self.tau,
        }

    def finish(self, conv_id: str) -> dict[str, Any]:
        """Final conversation verdict; drops the state."""
        state = self._states.pop(conv_id)
        conv_probs = self._conversation_probs(state)
        prob = conv_probs[-1] if conv_probs else 0.0
        return {
            "conv_id": conv_id,
            "n_turns": len(conv_probs),
            "conversation_violation_prob": prob,
            "per_turn_conversation_probs": conv_probs,
            "flagged": prob >= self.tau,
            "tau": self.tau,
        }

    def abort(self, conv_id: str) -> bool:
        return self._states.pop(conv_id, None) is not None

    # --------------------------------------------------------------- stateless
    def export_state(self, conv_id: str) -> dict[str, Any]:
        return self._states[conv_id].export()

    def import_state(self, payload: dict[str, Any], conv_id: str | None = None) -> str:
        conv_id = conv_id or uuid.uuid4().hex[:16]
        self._states[conv_id] = MonitorState.from_export(payload)
        return conv_id

    # ---------------------------------------------------------------- one-shot
    def score_conversation(self, policy: str, turns: list[Turn]) -> dict[str, Any]:
        """Score a full conversation in one call. Same math as the feed loop
        (causal aggregator), so the per-turn probs match streaming exactly."""
        latents = self.encoder.encode_conversation(policy, turns)
        state = MonitorState(policy=policy, policy_latent=latents.policy_latent)
        state.turn_latents = list(latents.turn_latents)
        state.turn_logits = list(latents.turn_logits)
        state.role_ids = latents.role_ids.tolist()
        conv_probs = self._conversation_probs(state)
        return {
            "n_turns": len(turns),
            "conversation_violation_prob": conv_probs[-1],
            "per_turn_conversation_probs": conv_probs,
            "turn_violation_probs": self.encoder.violation_probs(latents.turn_logits).tolist(),
            "flagged": conv_probs[-1] >= self.tau,
            "tau": self.tau,
        }

    # ---------------------------------------------------------------- internal
    @torch.no_grad()
    def _conversation_probs(self, state: MonitorState) -> list[float]:
        if not state.turn_latents:
            return []
        turn_latents = torch.stack(state.turn_latents).unsqueeze(0).to(self.device)
        turn_logits = torch.stack(state.turn_logits).unsqueeze(0).to(self.device)
        role_ids = torch.tensor([state.role_ids], dtype=torch.long, device=self.device)
        mask = torch.ones_like(role_ids)
        logits = self.aggregator(
            turn_latents=turn_latents,
            turn_logits=turn_logits,
            role_ids=role_ids,
            attention_mask=mask,
            policy_latent=state.policy_latent.unsqueeze(0).to(self.device),
        )
        probs = torch.softmax(logits[0], dim=-1)[:, self.positive_id]
        return [float(p) for p in probs]
