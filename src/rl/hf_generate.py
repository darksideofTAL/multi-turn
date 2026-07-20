"""Batched causal-LM generation via HF transformers (vLLM is broken on this B200;
see memory vllm-flashinfer-broken-b200). Prompts are pre-rendered chat-template
strings, so they are tokenized without adding a duplicate BOS."""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch
from transformers import AutoConfig, AutoTokenizer

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_tokenizer(model_id: str) -> Any:
    """Left-padded tokenizer (required for decoder-only batched generation). The
    generate/verify steps use this to render prompts; the model is loaded per-GPU
    in the workers spawned by :func:`generate_distributed`."""
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def load_lm(model_id: str, dtype: str = "bfloat16") -> tuple[Any, Any]:
    """Load a left-padded tokenizer + generative model on CUDA. Multimodal
    checkpoints (``*ConditionalGeneration``) load via AutoModelForImageTextToText."""
    torch_dtype = getattr(torch, dtype)
    tokenizer = load_tokenizer(model_id)

    arch = (
        getattr(AutoConfig.from_pretrained(model_id), "architectures", None) or [""]
    )[0]
    if "ConditionalGeneration" in arch or "ImageTextToText" in arch:
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch_dtype)
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch_dtype)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    # Best-effort: multimodal configs raise on a top-level pad_token_id read, and
    # generation also passes pad_token_id via gen_kwargs.
    try:
        if model.config.pad_token_id is None:
            model.config.pad_token_id = tokenizer.pad_token_id
    except AttributeError:
        pass
    return tokenizer, model


@torch.no_grad()
def generate_completions(
    tokenizer: Any,
    model: Any,
    prompts: list[str],
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    seed: int,
    batch_size: int = 16,
    max_input_tokens: int = 4096,
) -> list[str]:
    """Generate one completion per prompt. Returns decoded text (new tokens only)."""
    device = next(model.parameters()).device
    torch.manual_seed(seed)
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    # Add BOS unless the rendered prompt already has it: some templates (dolphin
    # ChatML) omit it, and a llama3.1 model without its BOS just echoes the prompt.
    bos = tokenizer.bos_token
    add_special = not (bos and prompts and prompts[0].lstrip().startswith(bos))
    logger.info("tokenizing with add_special_tokens=%s (bos=%r)", add_special, bos)

    outputs: list[str] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_tokens,
            add_special_tokens=add_special,
        ).to(device)
        gen = model.generate(**enc, **gen_kwargs)
        new_tokens = gen[:, enc["input_ids"].shape[1] :]
        outputs.extend(tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
        logger.info(
            "generated %d/%d", min(start + batch_size, len(prompts)), len(prompts)
        )
    return outputs


# ===========================================================================
# Data-parallel generation across multiple GPUs
# ===========================================================================


def _contiguous_shards(n_items: int, n_shards: int) -> list[tuple[int, int]]:
    base, extra = divmod(n_items, n_shards)
    bounds: list[tuple[int, int]] = []
    start = 0
    for i in range(n_shards):
        size = base + (1 if i < extra else 0)
        bounds.append((start, start + size))
        start += size
    return bounds


def generate_distributed(
    model_id: str,
    prompts: list[str],
    *,
    gpus: list[int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
    seed: int,
    batch_size: int = 16,
    max_input_tokens: int = 4096,
) -> list[str]:
    """Generate one completion per prompt, sharded across ``gpus``: contiguous shard
    per GPU, each a worker subprocess pinned via CUDA_VISIBLE_DEVICES loading its own
    model copy; outputs reassembled in prompt order. One GPU => one worker."""
    if not prompts:
        return []
    gpus = list(gpus) or [0]
    n = min(len(gpus), len(prompts))
    gpus = gpus[:n]
    shards = _contiguous_shards(len(prompts), n)

    gen = {
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "do_sample": do_sample,
        "batch_size": batch_size,
        "max_input_tokens": max_input_tokens,
    }

    env = os.environ.copy()
    env["PYTHONPATH"] = "src" + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")

    procs: list[tuple[subprocess.Popen, Path, tuple[int, int]]] = []
    with tempfile.TemporaryDirectory(prefix="rl_gen_") as tmp:
        tmp_dir = Path(tmp)
        for i, (gpu, (lo, hi)) in enumerate(zip(gpus, shards)):
            task_path = tmp_dir / f"task_{i}.json"
            out_path = tmp_dir / f"out_{i}.json"
            task = {
                "model": model_id,
                "prompts": prompts[lo:hi],
                "gen": gen,
                "seed": seed + i,  # decorrelate sampling across shards, still reproducible
                "out": str(out_path),
            }
            task_path.write_text(json.dumps(task))
            worker_env = env.copy()
            worker_env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            logger.info(
                "dispatch shard %d/%d (prompts %d:%d) -> GPU %d", i + 1, n, lo, hi, gpu
            )
            proc = subprocess.Popen(
                [sys.executable, "-m", "src.rl.hf_generate", "--worker", str(task_path)],
                env=worker_env,
                cwd=REPO_ROOT,
            )
            procs.append((proc, out_path, (lo, hi)))

        outputs: list[str] = [""] * len(prompts)
        failures: list[int] = []
        for i, (proc, out_path, (lo, hi)) in enumerate(procs):
            ret = proc.wait()
            if ret != 0 or not out_path.exists():
                failures.append(i)
                continue
            shard_out = json.loads(out_path.read_text())
            outputs[lo:hi] = shard_out
        if failures:
            raise RuntimeError(
                f"generation worker(s) {failures} failed (see logs above for the traceback)"
            )
    return outputs


def _run_worker(task_path: str) -> None:
    """Single-GPU worker: load the model on the one visible GPU and generate one shard."""
    task = json.loads(Path(task_path).read_text())
    gen = task["gen"]
    tokenizer, model = load_lm(task["model"])
    outputs = generate_completions(
        tokenizer,
        model,
        task["prompts"],
        max_new_tokens=gen["max_new_tokens"],
        temperature=gen["temperature"],
        top_p=gen["top_p"],
        do_sample=gen["do_sample"],
        seed=task["seed"],
        batch_size=gen["batch_size"],
        max_input_tokens=gen["max_input_tokens"],
    )
    Path(task["out"]).write_text(json.dumps(outputs, ensure_ascii=False))


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    parser = argparse.ArgumentParser(description="Single-GPU generation worker.")
    parser.add_argument("--worker", required=True, help="Path to the worker task JSON.")
    args = parser.parse_args()
    _run_worker(args.worker)


if __name__ == "__main__":
    main()
