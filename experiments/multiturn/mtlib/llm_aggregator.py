"""LLM latent aggregator: a FROZEN pretrained LM reads the policy as text and
the conversation as one projected soft token per turn.

    [instruction + policy text] [soft e_1] ... [soft e_T]
        -> frozen LM -> readout at each soft-token position
        -> P(conversation up to turn t violates the policy)

Same philosophy as the original classifier (replace the head, keep the LLM's
knowledge): the composition rule lives in the policy TEXT, which the LM reads
with pretrained comprehension — nothing composition-specific is trained.
Trainable parameters: soft-token projector + role embeddings (+ head when
head="linear"). The LM is fully frozen; gradients flow through it to the
projector but update nothing inside it.

Readout variants (config.head):
  "linear"     new linear layer on the LM hidden state at each soft position —
               the mirror of TalMonitor's `score` head move.
  "verbalizer" the frozen LM's OWN lm_head logits for the " No"/" Yes" tokens
               at each soft position (prompt-tuning style). The decision
               boundary lives in the LM's language head, which is
               policy-general — the OOD-calibration bet.

Cost: ~200 policy tokens once per conversation + ONE soft token per turn.
A 30-turn conversation is a ~230-token forward through a 1.5-4B LM (few ms);
the 12B turn-encode already paid by the single-turn product dominates.

Qwen3.5 backbones on B200 need FLA_DISABLE_BACKEND_DISPATCH=1 for backward.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn

INSTRUCTION_TEMPLATE = (
    "You are monitoring a user-agent conversation for violations of the policy "
    "below. Each conversation turn appears as a special token after the policy. "
    "Judge, at every turn, whether the conversation SO FAR violates the policy.\n"
    "POLICY: {policy}\nCONVERSATION:"
)

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}

# Sentinel lm_name: use the frozen turn-encoder's own backbone as the aggregator
# LM (domain-adapted to these policies; linear head only — no lm_head on it).
TALMONITOR_BACKBONE = "talmonitor-backbone"


def resolve_lm(lm_name: str, head: str, dtype: str):
    """(lm, tokenizer) for a config lm_name, understanding the talmonitor sentinel."""
    from transformers import AutoTokenizer

    if lm_name == TALMONITOR_BACKBONE:
        if head == "verbalizer":
            raise ValueError("talmonitor backbone has no lm_head; use head='linear'")
        from mtlib.common import CHECKPOINT
        from training.modeling import TalMonitorSequenceClassifier

        model = TalMonitorSequenceClassifier.from_pretrained(CHECKPOINT, dtype=_DTYPES[dtype])
        return model.backbone.to(_DTYPES[dtype]), AutoTokenizer.from_pretrained(CHECKPOINT)

    tokenizer = AutoTokenizer.from_pretrained(lm_name)
    if head == "verbalizer":
        from transformers import AutoModelForCausalLM

        return AutoModelForCausalLM.from_pretrained(lm_name, dtype=_DTYPES[dtype]), tokenizer
    from transformers import AutoModel

    return AutoModel.from_pretrained(lm_name, dtype=_DTYPES[dtype]), tokenizer


@dataclass
class LlmAggregatorConfig:
    lm_name: str = "Qwen/Qwen3-4B"
    input_dim: int = 3840  # turn-encoder hidden size
    num_labels: int = 2  # class order matches the classifier: [False, True]
    head: str = "linear"  # "linear" | "verbalizer"
    verbalizer_tokens: tuple[str, str] = (" No", " Yes")  # (negative, positive)
    use_logit_features: bool = True
    projector_hidden: int = 2048
    max_policy_tokens: int = 512
    dtype: str = "bfloat16"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "LlmAggregatorConfig":
        d = dict(d)
        if "verbalizer_tokens" in d:
            d["verbalizer_tokens"] = tuple(d["verbalizer_tokens"])
        return cls(**d)


class LlmLatentAggregator(nn.Module):
    def __init__(
        self,
        config: LlmAggregatorConfig,
        lm: nn.Module | None = None,
        tokenizer=None,
    ) -> None:
        """``lm``/``tokenizer`` overrides exist for offline tests (tiny random
        models); normally both load from config.lm_name."""
        super().__init__()
        if config.head not in ("linear", "verbalizer"):
            raise ValueError(f"unknown head: {config.head!r}")
        self.config = config

        if lm is None:
            lm, resolved_tokenizer = resolve_lm(config.lm_name, config.head, config.dtype)
            tokenizer = tokenizer or resolved_tokenizer
        elif tokenizer is None:
            raise ValueError("pass a tokenizer when injecting a prebuilt lm")
        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.lm = lm
        self.lm.config.use_cache = False
        for p in self.lm.parameters():
            p.requires_grad_(False)
        self.lm.eval()

        # Multimodal/composite configs (Gemma3) nest hidden_size under text_config.
        lm_config = self.lm.config
        d_lm = getattr(lm_config, "hidden_size", None)
        if d_lm is None and hasattr(lm_config, "get_text_config"):
            d_lm = lm_config.get_text_config().hidden_size
        in_dim = config.input_dim + (config.num_labels if config.use_logit_features else 0)
        # fp32 trainables; soft tokens are cast to the LM dtype on entry.
        self.projector = nn.Sequential(
            nn.Linear(in_dim, config.projector_hidden),
            nn.GELU(),
            nn.Linear(config.projector_hidden, d_lm),
        )
        # Roles: 0 unused (policy is text), 1=user, 2=agent (mtlib.schema.ROLE_IDS).
        self.role_emb = nn.Embedding(3, d_lm)
        self.norm = nn.LayerNorm(d_lm)
        if config.head == "linear":
            self.head = nn.Linear(d_lm, config.num_labels)
            self.head.weight.data.zero_()
            self.head.bias.data.zero_()
        else:
            self.head = None
            self.verbalizer_ids = tuple(
                self.tokenizer(t, add_special_tokens=False)["input_ids"][0]
                for t in config.verbalizer_tokens
            )
        self._policy_cache: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------ text
    def _device(self) -> torch.device:
        return next(self.lm.parameters()).device

    def _embed_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        return self.lm.get_input_embeddings()(ids)

    def policy_token_embeddings(self, policy: str) -> torch.Tensor:
        """Embedded instruction+policy prefix [P, d_lm] (cached per policy text)."""
        key = policy.strip()
        if key not in self._policy_cache:
            text = INSTRUCTION_TEMPLATE.format(policy=key)
            ids = self.tokenizer(
                text, truncation=True, max_length=self.config.max_policy_tokens,
                return_tensors="pt",
            )["input_ids"][0]
            with torch.no_grad():
                emb = self._embed_tokens(ids.to(self._device()))
            self._policy_cache[key] = emb
        return self._policy_cache[key]

    # --------------------------------------------------------------- forward
    def forward(
        self,
        policies: list[str],  # B policy texts
        turn_latents: torch.Tensor,  # [B, T, H] fp32
        turn_logits: torch.Tensor | None,  # [B, T, num_labels]
        role_ids: torch.Tensor,  # [B, T]
        attention_mask: torch.Tensor,  # [B, T] 1=real turn
    ) -> torch.Tensor:
        """Per-turn conversation-status logits [B, T, num_labels] in the
        classifier's class order [False, True]."""
        cfg = self.config
        device = self._device()
        lm_dtype = next(self.lm.parameters()).dtype
        if cfg.use_logit_features:
            if turn_logits is None:
                raise ValueError("use_logit_features=True requires turn_logits")
            features = torch.cat([turn_latents, turn_logits], dim=-1)
        else:
            features = turn_latents
        soft = self.projector(features.to(device)) + self.role_emb(role_ids.to(device))
        soft = self.norm(soft)  # [B, T, d_lm], fp32

        prefixes = [self.policy_token_embeddings(p) for p in policies]
        batch, turns = soft.shape[0], soft.shape[1]
        max_len = max(p.shape[0] for p in prefixes) + turns
        d_lm = soft.shape[-1]

        # LEFT-pad so every row ends at max_len: pad never sits between the
        # policy and the turns, and soft positions align at the causal tail.
        inputs = torch.zeros(batch, max_len, d_lm, device=device, dtype=lm_dtype)
        mask = torch.zeros(batch, max_len, dtype=torch.long, device=device)
        soft_pos = torch.zeros(batch, turns, dtype=torch.long, device=device)
        for i, prefix in enumerate(prefixes):
            n = prefix.shape[0]
            start = max_len - n - turns
            inputs[i, start : start + n] = prefix.to(lm_dtype)
            inputs[i, start + n :] = soft[i].to(lm_dtype)
            mask[i, start:] = 1
            soft_pos[i] = torch.arange(start + n, max_len, device=device)

        outputs = self.lm(inputs_embeds=inputs, attention_mask=mask)
        if cfg.head == "linear":
            hidden = outputs.last_hidden_state
            gathered = torch.gather(hidden, 1, soft_pos.unsqueeze(-1).expand(-1, -1, d_lm))
            logits = self.head(gathered.float())
        else:
            vocab_logits = outputs.logits  # [B, S, V]
            gathered = torch.gather(
                vocab_logits, 1, soft_pos.unsqueeze(-1).expand(-1, -1, vocab_logits.shape[-1])
            )
            neg_id, pos_id = self.verbalizer_ids
            logits = torch.stack(
                [gathered[..., neg_id], gathered[..., pos_id]], dim=-1
            ).float()  # [B, T, 2] in [False, True] order
        return logits * attention_mask.to(device).unsqueeze(-1)

    # -------------------------------------------------------------- save/load
    def save(self, path: str | Path, extra: dict | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        trainable = {k: v for k, v in self.state_dict().items() if not k.startswith("lm.")}
        payload = {"config": self.config.to_dict(), "state_dict": trainable}
        payload.update(extra or {})
        torch.save(payload, path)
        with open(path.with_suffix(".json"), "w") as f:
            json.dump({k: v for k, v in payload.items() if k != "state_dict"}, f, indent=2)

    @classmethod
    def load(
        cls,
        path: str | Path,
        map_location: str = "cpu",
        lm: nn.Module | None = None,
        tokenizer=None,
    ) -> tuple["LlmLatentAggregator", dict]:
        payload = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(LlmAggregatorConfig.from_dict(payload["config"]), lm=lm, tokenizer=tokenizer)
        missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
        missing = [k for k in missing if not k.startswith("lm.")]
        unexpected = [k for k in unexpected if not k.startswith("lm.")]
        if missing or unexpected:
            raise RuntimeError(f"bad checkpoint: missing={missing} unexpected={unexpected}")
        model.eval()
        extra = {k: v for k, v in payload.items() if k not in ("config", "state_dict")}
        return model, extra

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
