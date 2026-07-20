"""Small utilities used across the RL pipeline.

JSONL I/O, schema validation, classifier loading, and a thin wrapper around
TALMONITOR's ``Prompter`` so callers (steps, eval) don't repeat the same
render/load boilerplate.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterable, Iterator

from data_handler.data_class import InputBlockClsfSample
from evaluations.inference_prompter import Prompter
from training.runner import build_prompter


logger = logging.getLogger(__name__)

REFUSAL_PHRASES = (
    "I cannot",
    "I can't",
    "I apologize",
    "I'm sorry, but",
    "I am sorry, but",
    "As an AI",
    "I am unable",
)
MIN_INPUT_BLOCK_CHARS = 30
MAX_INPUT_BLOCK_CHARS = 6000


def write_jsonl(rows: Iterable[dict[str, Any]], path: str | Path) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def concat_jsonl(out_path: str | Path, *in_paths: str | Path) -> int:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w") as fout:
        for in_path in in_paths:
            if not Path(in_path).exists():
                continue
            with open(in_path) as fin:
                for line in fin:
                    if line.strip():
                        fout.write(line if line.endswith("\n") else line + "\n")
                        n += 1
    return n


def load_classifier(classifier_path: str, device: Any | None = None) -> tuple[Any, Any, dict[int, str]]:
    """Load a TALMONITOR sequence classifier for inference -> ``(tokenizer, model, id2label)``,
    mirroring the trainer's pad-token setup, on ``device`` (CUDA if available), in eval mode."""
    import torch
    from transformers import AutoTokenizer

    from training.modeling import TalMonitorSequenceClassifier

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(classifier_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = TalMonitorSequenceClassifier.from_pretrained(classifier_path)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device).eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    return tokenizer, model, id2label


def render_classifier_text(policy_prompt: str, input_block: str, labels: list[str], prompter: Prompter | None = None) -> str:
    """Render one classifier input via TALMONITOR's Prompter, byte-identical to
    ``training.runner.render_input_block_prompt`` (train/inference must match)."""
    prompter = prompter or build_prompter()
    sample = InputBlockClsfSample(input_block=input_block, rule=policy_prompt, labels=labels)
    return f"{prompter.get_system_msg(sample)}\n\n{prompter.get_user_msg(sample)}"


def passes_light_filters(input_block: str) -> tuple[bool, str | None]:
    """Apply DuoGuard's light filters to a generated transcript.

    Returns ``(ok, reason_if_rejected)``.
    """
    if not input_block or not isinstance(input_block, str):
        return False, "empty"
    if "<User>" not in input_block or "<Agent>" not in input_block:
        return False, "missing_tags"
    if len(input_block) < MIN_INPUT_BLOCK_CHARS:
        return False, "too_short"
    if len(input_block) > MAX_INPUT_BLOCK_CHARS:
        return False, "too_long"
    for agent_chunk in _agent_chunks(input_block):
        stripped = agent_chunk.lstrip()
        if any(stripped.startswith(p) for p in REFUSAL_PHRASES):
            return False, "refusal_phrase"
    return True, None


def _agent_chunks(input_block: str) -> list[str]:
    chunks = []
    idx = 0
    open_tag, close_tag = "<Agent>", "</Agent>"
    while True:
        start = input_block.find(open_tag, idx)
        if start == -1:
            break
        end = input_block.find(close_tag, start)
        if end == -1:
            chunks.append(input_block[start + len(open_tag):])
            break
        chunks.append(input_block[start + len(open_tag):end])
        idx = end + len(close_tag)
    return chunks
