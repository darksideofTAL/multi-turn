import json
import logging
import threading
from pathlib import Path

import torch
from transformers import AutoTokenizer

from deploy.config import DeployConfig
from training.modeling import TalMonitorSequenceClassifier
from training.runner import build_prompter, render_input_block_prompt

logger = logging.getLogger(__name__)


def is_quantized_checkpoint(model_dir) -> bool:
    """True if the checkpoint's config declares a real quantization scheme (e.g. fp8).
    Such checkpoints hold fp8 weight buffers that must be loaded natively (never cast)."""
    config_path = Path(model_dir) / "config.json"
    if not config_path.exists():
        return False
    try:
        return bool(json.loads(config_path.read_text()).get("quantization"))
    except (json.JSONDecodeError, OSError):
        return False

_DTYPES = {
    "auto": None,
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}


def resolve_dtype(name: str) -> torch.dtype | None:
    key = name.strip().lower()
    if key not in _DTYPES:
        raise ValueError(f"Unsupported dtype: {name!r}")
    return _DTYPES[key]


def resolve_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ClassifierEngine:
    """Transformers classifier behind the (policy, input_block) -> softmax contract.

    Mirrors training/eval.py exactly so served predictions match offline eval.
    """

    def __init__(self, config: DeployConfig):
        self.model_name = config.model_dir.name
        self.max_seq_length = config.max_seq_length
        self.max_forward_batch_tokens = config.max_forward_batch_tokens
        self.max_forward_batch_items = config.max_forward_batch_items
        self.device = resolve_device(config.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            config.model_dir, trust_remote_code=config.trust_remote_code
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"

        # Real-fp8 checkpoints store fp8 weight buffers; casting them (model.to(bf16))
        # breaks _scaled_mm, so load them natively (dtype=None), like training/eval.py.
        dtype = None if is_quantized_checkpoint(config.model_dir) else resolve_dtype(config.dtype)
        model = TalMonitorSequenceClassifier.from_pretrained(
            config.model_dir,
            dtype=dtype,
            trust_remote_code=config.trust_remote_code,
        )
        if dtype is not None:
            model = model.to(dtype)
        model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model = model.to(self.device).eval()

        if config.compile:
            # Compile the backbone only: it holds ~all the latency, while the
            # data-dependent pool, the fp32 head, and the ModelOutput return stay
            # eager (the latter would otherwise force a trailing graph break).
            # no-cudagraphs + dynamic avoids per-shape recompiles on the variable
            # (batch, seq) shapes that length-bucketing produces.
            torch.set_float32_matmul_precision("high")
            self.model.backbone = torch.compile(
                self.model.backbone,
                mode=config.compile_mode,
                dynamic=config.compile_dynamic,
            )
            logger.info(
                "Compiled backbone (mode=%s, dynamic=%s)",
                config.compile_mode,
                config.compile_dynamic,
            )

        # Class-id order is the source of truth (same as training/eval.py).
        self.id2label = {int(k): v for k, v in self.model.config.id2label.items()}
        self.labels = [self.id2label[i] for i in range(len(self.id2label))]
        if config.labels and list(config.labels) != self.labels:
            raise ValueError(
                f"Config labels {list(config.labels)} do not match checkpoint "
                f"id2label order {self.labels}"
            )

        self.prompter = build_prompter()
        self._lock = threading.Lock()
        logger.info("Serving %s with labels %s", self.model_name, self.labels)

    def render(self, policy: str, input_block: str) -> str:
        return render_input_block_prompt(
            policy, input_block, self.labels, self.prompter
        )

    def _length_buckets(self, order: list[int], input_ids: list[list[int]]) -> list[list[int]]:
        """Pack length-sorted indices into micro-batches bounded by row and padded-token caps."""
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

    def _result(self, row: torch.Tensor) -> dict:
        scores = {self.id2label[i]: float(row[i].item()) for i in range(len(row))}
        pred_id = int(row.argmax().item())
        label = self.id2label.get(pred_id, str(pred_id))
        return {"label": label, "scores": scores, "model": self.model_name}

    def classify_batch(
        self, policies: list[str], input_blocks: list[str]
    ) -> list[dict]:
        """Classify (policy, input_block) pairs, length-bucketing to bound padding."""
        if len(policies) != len(input_blocks):
            raise ValueError("policies and input_blocks must be the same length")
        if not policies:
            return []
        texts = [self.render(p, b) for p, b in zip(policies, input_blocks)]
        encodings = self.tokenizer(
            texts, truncation=True, max_length=self.max_seq_length
        )
        order = sorted(range(len(texts)), key=lambda i: len(encodings["input_ids"][i]))

        probs: list[torch.Tensor | None] = [None] * len(texts)
        with self._lock, torch.no_grad():
            for group in self._length_buckets(order, encodings["input_ids"]):
                batch = self.tokenizer.pad(
                    {key: [encodings[key][i] for i in group] for key in encodings},
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                group_probs = torch.softmax(self.model(**batch).logits, dim=-1).cpu()
                for slot, row in zip(group, group_probs):
                    probs[slot] = row
        return [self._result(row) for row in probs]

    def classify(self, policy: str, input_block: str) -> dict:
        return self.classify_batch([policy], [input_block])[0]
