"""Frozen single-turn classifier as a turn encoder.

Each (policy, block) pair is rendered exactly as in training/deploy and gets
one backbone forward. The turn latent is the pooled last-non-pad hidden state
(the vector the linear head reads), returned together with the fp32 head
logits, so per-turn single-turn verdicts come for free with the embedding.

Policy conditioning happens HERE: every turn is encoded with the policy in
context, so the aggregator downstream never needs the policy text. The policy
latent is the same render with an empty sample block.

Rendering / right-padding / length-bucketing mirror deploy/engine.py so latents
match served behavior.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

# Importing mtlib.common first runs its sys.path bootstrap, so the training.*
# imports below resolve against the repo's src/.
from mtlib.common import CHECKPOINT, MAX_SEQ_LENGTH, POSITIVE_LABEL
from mtlib.schema import ROLE_IDS, Turn, turn_block

from transformers import AutoTokenizer  # noqa: E402

from training.modeling import TalMonitorSequenceClassifier  # noqa: E402
from training.runner import build_prompter, render_input_block_prompt  # noqa: E402

logger = logging.getLogger(__name__)

# Sample block used to embed the policy alone.
POLICY_ONLY_BLOCK = ""

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


@dataclass
class ConversationLatents:
    """Everything the aggregator consumes for one conversation, fp32 on CPU."""

    policy_latent: torch.Tensor  # [H]
    turn_latents: torch.Tensor  # [T, H]
    turn_logits: torch.Tensor  # [T, num_labels]
    role_ids: torch.Tensor  # [T] int64, values from ROLE_IDS


class TurnEncoder:
    def __init__(
        self,
        model_dir: str | Path = CHECKPOINT,
        device: str | None = None,
        dtype: str = "bfloat16",
        max_seq_length: int = MAX_SEQ_LENGTH,
        max_forward_batch_items: int = 8,
        max_forward_batch_tokens: int = 32768,
    ) -> None:
        # Coerce a falsy model_dir (scripts pass ``args.encoder_ckpt or None``) back
        # to the default checkpoint, so an explicit None doesn't override it.
        model_dir = model_dir or CHECKPOINT
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.max_seq_length = max_seq_length
        self.max_forward_batch_items = max_forward_batch_items
        self.max_forward_batch_tokens = max_forward_batch_tokens

        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        model = TalMonitorSequenceClassifier.from_pretrained(model_dir, dtype=_DTYPES[dtype])
        model = model.to(_DTYPES[dtype])
        model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model = model.to(self.device).eval()

        self.id2label = {int(k): v for k, v in model.config.id2label.items()}
        self.labels = [self.id2label[i] for i in range(len(self.id2label))]
        self.positive_id = self.labels.index(POSITIVE_LABEL)
        self.num_labels = len(self.labels)
        self.hidden_size = model.score.weight.shape[1]
        self.prompter = build_prompter()
        logger.info(
            "TurnEncoder: %s (hidden=%d, labels=%s)", Path(model_dir).name, self.hidden_size, self.labels
        )

    def render(self, policy: str, block: str) -> str:
        return render_input_block_prompt(policy, block, self.labels, self.prompter)

    def _length_buckets(self, order: list[int], input_ids: list[list[int]]) -> list[list[int]]:
        """Pack length-sorted indices into micro-batches bounded by row and padded-token caps
        (same scheme as deploy/engine.py)."""
        buckets: list[list[int]] = []
        group: list[int] = []
        for idx in order:
            cap = len(input_ids[idx])
            if group and (
                len(group) >= self.max_forward_batch_items
                or cap * (len(group) + 1) > self.max_forward_batch_tokens
            ):
                buckets.append(group)
                group = []
            group.append(idx)
        if group:
            buckets.append(group)
        return buckets

    @torch.no_grad()
    def encode(self, policies: list[str], blocks: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode (policy, block) pairs -> (latents [N, H], logits [N, num_labels]),
        both fp32 on CPU. One backbone forward per pair; pooling and head mirror
        TalMonitorSequenceClassifier.forward."""
        if len(policies) != len(blocks):
            raise ValueError("policies and blocks must be the same length")
        if not policies:
            empty = torch.empty(0, self.hidden_size)
            return empty, torch.empty(0, self.num_labels)

        texts = [self.render(p, b) for p, b in zip(policies, blocks)]
        encodings = self.tokenizer(texts, truncation=True, max_length=self.max_seq_length)
        order = sorted(range(len(texts)), key=lambda i: len(encodings["input_ids"][i]))

        latents = torch.empty(len(texts), self.hidden_size)
        logits = torch.empty(len(texts), self.num_labels)
        weight = self.model.score.weight.float()
        bias = None if self.model.score.bias is None else self.model.score.bias.float()
        for group in self._length_buckets(order, encodings["input_ids"]):
            batch = self.tokenizer.pad(
                {key: [encodings[key][i] for i in group] for key in encodings},
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            outputs = self.model.backbone(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            )
            hidden = (
                outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
            )
            pooled = self.model._pool(hidden, batch["input_ids"], batch["attention_mask"]).float()
            group_logits = nn.functional.linear(pooled, weight, bias)
            for slot, vec, row in zip(group, pooled.cpu(), group_logits.cpu()):
                latents[slot] = vec
                logits[slot] = row
        return latents, logits

    def encode_policy(self, policy: str) -> torch.Tensor:
        latents, _ = self.encode([policy], [POLICY_ONLY_BLOCK])
        return latents[0]

    def encode_conversation(self, policy: str, turns: list[Turn]) -> ConversationLatents:
        """Encode each turn independently (policy in context) plus the policy latent.
        One flat encode call so length-bucketing spans all rows."""
        blocks = [POLICY_ONLY_BLOCK] + [turn_block(t) for t in turns]
        latents, logits = self.encode([policy] * len(blocks), blocks)
        role_ids = torch.tensor([ROLE_IDS[t.role] for t in turns], dtype=torch.long)
        return ConversationLatents(
            policy_latent=latents[0],
            turn_latents=latents[1:],
            turn_logits=logits[1:],
            role_ids=role_ids,
        )

    def violation_probs(self, logits: torch.Tensor) -> torch.Tensor:
        """P(violation) per row from head logits."""
        return torch.softmax(logits.float(), dim=-1)[..., self.positive_id]

    def score_blocks(self, policies: list[str], blocks: list[str]) -> torch.Tensor:
        """Single-turn P(violation) for (policy, block) pairs — used by the
        full-concat baseline and datagen teacher filtering."""
        _, logits = self.encode(policies, blocks)
        return self.violation_probs(logits)
