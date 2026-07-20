"""Conversation aggregator: a small fp32 causal transformer over per-turn latents.

Input sequence = [policy latent, e_1 .. e_T] where e_t is the frozen
classifier's pooled vector for turn t (already policy-conditioned; the policy
token is an optional extra signal). Per-turn head logits can be concatenated as
features so the single-turn verdict is trivially recoverable — the aggregator
can only add signal over the max-over-turns baseline, not lose it.

Causal attention makes it a streaming monitor: position t reads only turns
<= t, so the head output at t answers "does the conversation UP TO turn t
violate the policy". RoPE over turn index (not learned absolute positions) for
length generalization past the training horizon.

Kept fully fp32: at ~15M params the bf16 batch-invariance problem from the
deploy work never enters.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

IGNORE_INDEX = -100


@dataclass
class AggregatorConfig:
    input_dim: int  # backbone hidden size (3840 for Gemma3-12B)
    num_labels: int = 2
    d_model: int = 512
    n_layers: int = 3
    n_heads: int = 8
    ffn_mult: int = 4
    dropout: float = 0.1
    rope_base: float = 10000.0
    use_policy_token: bool = True
    use_logit_features: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "AggregatorConfig":
        return cls(**d)


def rope_cache(
    seq_len: int, head_dim: int, base: float, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """(cos, sin) tables of shape [seq_len, head_dim/2]. Position 0 is the identity
    rotation, so the policy token (position 0) is unrotated."""
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, device=device).to(dtype) / head_dim))
    positions = torch.arange(seq_len, device=device, dtype=dtype)
    angles = torch.outer(positions, inv_freq)  # [T, hd/2]
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate q/k. x: [B, heads, T, head_dim]; cos/sin: [T, head_dim/2]."""
    x1, x2 = x.chunk(2, dim=-1)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return torch.cat((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)


class _Block(nn.Module):
    """Pre-norm transformer block: RoPE multi-head attention + GELU MLP."""

    def __init__(self, config: AggregatorConfig) -> None:
        super().__init__()
        d, h = config.d_model, config.n_heads
        if d % h != 0:
            raise ValueError(f"d_model {d} not divisible by n_heads {h}")
        if (d // h) % 2 != 0:
            raise ValueError(f"head_dim {d // h} must be even for RoPE")
        self.n_heads = h
        self.head_dim = d // h
        self.norm_attn = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.attn_out = nn.Linear(d, d)
        self.norm_mlp = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, config.ffn_mult * d),
            nn.GELU(),
            nn.Linear(config.ffn_mult * d, d),
        )
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self, x: torch.Tensor, attn_mask: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
    ) -> torch.Tensor:
        batch, seq, _ = x.shape
        q, k, v = self.qkv(self.norm_attn(x)).chunk(3, dim=-1)
        q = q.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq, self.n_heads, self.head_dim).transpose(1, 2)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        attn = nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        attn = attn.transpose(1, 2).reshape(batch, seq, -1)
        x = x + self.dropout(self.attn_out(attn))
        x = x + self.dropout(self.mlp(self.norm_mlp(x)))
        return x


class ConversationAggregator(nn.Module):
    def __init__(self, config: AggregatorConfig) -> None:
        super().__init__()
        self.config = config
        in_dim = config.input_dim + (config.num_labels if config.use_logit_features else 0)
        self.proj = nn.Linear(in_dim, config.d_model)
        self.norm_in = nn.LayerNorm(config.d_model)
        # Role ids: 0=policy token, 1=user, 2=agent (mtlib.schema.ROLE_IDS).
        self.role_emb = nn.Embedding(3, config.d_model)
        self.blocks = nn.ModuleList(_Block(config) for _ in range(config.n_layers))
        self.norm_out = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.num_labels)
        # Zero-init head for calm first steps (same rationale as the base classifier).
        self.head.weight.data.zero_()
        self.head.bias.data.zero_()
        self.float()

    def _attn_mask(self, attention_mask: torch.Tensor) -> torch.Tensor:
        """Additive [B, 1, S, S] mask: causal AND key-not-pad. The diagonal is
        always allowed so fully-masked pad query rows don't NaN the softmax."""
        batch, seq = attention_mask.shape
        device = attention_mask.device
        causal = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=device))
        keep = causal.unsqueeze(0) & attention_mask.bool().unsqueeze(1)  # [B, S, S]
        keep = keep | torch.eye(seq, dtype=torch.bool, device=device).unsqueeze(0)
        mask = torch.zeros(batch, seq, seq, device=device)
        mask.masked_fill_(~keep, torch.finfo(torch.float32).min)
        return mask.unsqueeze(1)

    def forward(
        self,
        turn_latents: torch.Tensor,  # [B, T, H] fp32
        turn_logits: torch.Tensor | None,  # [B, T, num_labels] fp32 (required if use_logit_features)
        role_ids: torch.Tensor,  # [B, T] int64
        attention_mask: torch.Tensor,  # [B, T] 1=real turn, 0=pad
        policy_latent: torch.Tensor | None = None,  # [B, H] (required if use_policy_token)
    ) -> torch.Tensor:
        """Per-turn conversation-status logits [B, T, num_labels]; position t sees
        only the policy token and turns <= t."""
        cfg = self.config
        if cfg.use_logit_features:
            if turn_logits is None:
                raise ValueError("config.use_logit_features=True requires turn_logits")
            features = torch.cat([turn_latents, turn_logits], dim=-1)
        else:
            features = turn_latents
        x = self.norm_in(self.proj(features)) + self.role_emb(role_ids)

        if cfg.use_policy_token:
            if policy_latent is None:
                raise ValueError("config.use_policy_token=True requires policy_latent")
            if cfg.use_logit_features:
                pad = policy_latent.new_zeros(policy_latent.shape[0], cfg.num_labels)
                policy_features = torch.cat([policy_latent, pad], dim=-1)
            else:
                policy_features = policy_latent
            policy_tok = self.norm_in(self.proj(policy_features)) + self.role_emb(
                torch.zeros(x.shape[0], dtype=torch.long, device=x.device)
            )
            x = torch.cat([policy_tok.unsqueeze(1), x], dim=1)
            mask = torch.cat(
                [attention_mask.new_ones(attention_mask.shape[0], 1), attention_mask], dim=1
            )
        else:
            mask = attention_mask

        seq = x.shape[1]
        cos, sin = rope_cache(seq, self.blocks[0].head_dim, cfg.rope_base, x.device, x.dtype)
        attn_mask = self._attn_mask(mask)
        for block in self.blocks:
            x = block(x, attn_mask, cos, sin)
        logits = self.head(self.norm_out(x))
        if cfg.use_policy_token:
            logits = logits[:, 1:]  # drop the policy slot; outputs align with turns
        return logits

    def save(self, path: str | Path, extra: dict | None = None) -> None:
        """Checkpoint = {config, state_dict, **extra}. `extra` carries e.g. the
        stamped decision threshold tau."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"config": self.config.to_dict(), "state_dict": self.state_dict()}
        payload.update(extra or {})
        torch.save(payload, path)
        with open(path.with_suffix(".json"), "w") as f:
            json.dump({k: v for k, v in payload.items() if k != "state_dict"}, f, indent=2)

    @classmethod
    def load(cls, path: str | Path, map_location: str = "cpu") -> tuple["ConversationAggregator", dict]:
        payload = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(AggregatorConfig.from_dict(payload["config"]))
        model.load_state_dict(payload["state_dict"])
        model.eval()
        extra = {k: v for k, v in payload.items() if k not in ("config", "state_dict")}
        return model, extra


def per_turn_loss(
    logits: torch.Tensor,  # [B, T, num_labels]
    labels: torch.Tensor,  # [B, T] with IGNORE_INDEX on pads
    onset: torch.Tensor,  # [B] first-violation turn index, -1 for benign
    onset_weight: float = 2.0,
) -> torch.Tensor:
    """Per-position CE, mean over valid positions, with the first-violation
    position upweighted (trains detection lag, not just final verdicts). Plain
    CE only — hinge terms were seed-sensitive in the token-supervision work."""
    batch, seq, num_labels = logits.shape
    ce = nn.functional.cross_entropy(
        logits.reshape(-1, num_labels),
        labels.reshape(-1),
        ignore_index=IGNORE_INDEX,
        reduction="none",
    ).reshape(batch, seq)
    weights = torch.ones_like(ce)
    valid_onset = onset >= 0
    if valid_onset.any():
        rows = torch.nonzero(valid_onset, as_tuple=True)[0]
        weights[rows, onset[rows]] = onset_weight
    valid = (labels != IGNORE_INDEX).float()
    return (ce * weights * valid).sum() / (weights * valid).sum().clamp_min(1.0)
