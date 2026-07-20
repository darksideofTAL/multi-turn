"""Held-out adversarial evaluation for the RL classifier -- the metric the loop
targets (verbatim YAML examples saturate at F1~1.0). Two subcommands:

  build  generate + judge-verify adversarial candidates on the held-out policies
         only, freezing the verified rows as a gold set.
  score  score a classifier against a frozen build set (acc/prec/rec/F1, by policy).

Splitting them lets the expensive generate+verify run once for many classifiers.
Caveat: generator and judge may share a family, so gold = "judge is confident", not oracle.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from training.runner import build_prompter

from src.rl.config import RLConfig, _resolve, load_rl_config
from src.rl.steps import annotate, build_judge_prompts, build_prompts_from_units, parse_and_filter
from src.rl.utils import (
    iter_jsonl,
    load_classifier,
    render_classifier_text,
    write_jsonl,
)

logger = logging.getLogger(__name__)

DEFAULT_ADV_OUT = "output/rl/eval_adv/heldout_verified.jsonl"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _metrics(predictions: list[dict[str, Any]], positive_label: str) -> dict[str, float]:
    n = len(predictions)
    if n == 0:
        return {"n": 0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}
    tp = sum(
        1 for p in predictions if p["predicted_label"] == positive_label and p["label"] == positive_label
    )
    fp = sum(
        1 for p in predictions if p["predicted_label"] == positive_label and p["label"] != positive_label
    )
    fn = sum(
        1 for p in predictions if p["predicted_label"] != positive_label and p["label"] == positive_label
    )
    correct = sum(1 for p in predictions if p["correct"])
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "n": n,
        "accuracy": correct / n,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def _aggregate(
    predictions: list[dict[str, Any]], key: str, positive_label: str
) -> dict[str, dict[str, float]]:
    groups: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for p in predictions:
        groups[p[key]].append(p)
    return {str(k): _metrics(v, positive_label=positive_label) for k, v in sorted(groups.items())}


# ---------------------------------------------------------------------------
# build: generate + judge-verify the held-out gold set
# ---------------------------------------------------------------------------


def _held_out_seed_units(cfg: RLConfig, max_eval_seeds: int, rng: random.Random) -> list[dict[str, Any]]:
    """Seed units for the eval, drawn from the held-out rows of accepted_samples."""
    if not cfg.seed_samples_path:
        raise ValueError("eval build requires cfg.seed_samples_path (accepted_samples.jsonl).")
    held = set(cfg.held_out_ids)
    rows = [
        r
        for r in iter_jsonl(_resolve(cfg.seed_samples_path))
        if r.get("guardrail_id") in held and r.get("policy_prompt") and r.get("input_block")
    ]
    rng.shuffle(rows)
    rows = rows[:max_eval_seeds]
    return [
        {"guardrail_id": r["guardrail_id"], "seed_policy": r["policy_prompt"], "seed_example": r["input_block"]}
        for r in rows
    ]


def build(
    cfg: RLConfig,
    generator_model: str,
    judge_model: str,
    n_per_seed: int,
    out_path: Path,
    max_eval_seeds: int = 120,
) -> dict[str, int]:
    """Generate + judge-verify adversarial candidates on the held-out policies."""
    from src.rl.hf_generate import generate_distributed, load_tokenizer

    rng = random.Random(cfg.seed)
    gpus = cfg.gen_gpus()
    units = _held_out_seed_units(cfg, max_eval_seeds, rng)
    logger.info("build: %d held-out seed rows, n_per_seed=%d, GPUs %s", len(units), n_per_seed, gpus)

    prompts, metadata = build_prompts_from_units(units, n_per_seed, load_tokenizer(generator_model), rng)
    raw = generate_distributed(
        generator_model,
        prompts,
        gpus=gpus,
        max_new_tokens=cfg.generator.max_tokens,
        temperature=cfg.generator.temperature,
        top_p=cfg.generator.top_p,
        do_sample=cfg.generator.temperature > 0,
        seed=cfg.seed,
        batch_size=cfg.generator.batch_size,
    )
    kept, diagnostics = parse_and_filter(raw, metadata)
    logger.info("generated %d prompts -> %d kept (%d rejected)", len(prompts), len(kept), len(diagnostics))

    if not kept:
        raise RuntimeError("no candidates survived generation+filters; cannot build eval set")

    judge_prompts = build_judge_prompts(kept, load_tokenizer(judge_model), cfg)
    judge_raw = generate_distributed(
        judge_model,
        judge_prompts,
        gpus=gpus,
        max_new_tokens=cfg.verifier.max_tokens,
        temperature=cfg.verifier.temperature,
        top_p=cfg.verifier.top_p,
        do_sample=cfg.verifier.temperature > 0,
        seed=cfg.seed,
        batch_size=cfg.verifier.batch_size,
    )
    annotated = annotate(kept, judge_raw)
    verified = [r for r in annotated if r["verified"]]

    n = write_jsonl(verified, out_path)
    pos = sum(1 for r in verified if r["target_label"] == "True")
    logger.info(
        "eval set: %d verified / %d candidates (True=%d False=%d) -> %s",
        n, len(annotated), pos, n - pos, out_path,
    )
    return {"candidates": len(annotated), "verified": n, "pos": pos, "neg": n - pos}


# ---------------------------------------------------------------------------
# score: run a classifier against a frozen gold set
# ---------------------------------------------------------------------------


def score(cfg: RLConfig, classifier_path: str, in_path: Path, out_path: Path) -> dict[str, Any]:
    """Score a classifier on a frozen held-out adversarial set."""
    rows = list(iter_jsonl(in_path))
    if not rows:
        raise RuntimeError(f"empty adversarial eval set: {in_path}")
    labels = list(cfg.labels)
    positive_label = labels[1]
    max_seq_length = cfg.classifier.max_seq_length

    logger.info("scoring %s on %d held-out adversarial examples", classifier_path, len(rows))
    tokenizer, model, id2label = load_classifier(classifier_path)
    device = next(model.parameters()).device
    prompter = build_prompter()

    predictions: list[dict[str, Any]] = []
    for row in rows:
        gold = row["target_label"]  # == judge_label for verified rows
        text = render_classifier_text(row["policy_prompt"], row["input_block"], labels, prompter)
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_seq_length).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits[0], dim=-1).cpu()
        pred_label = id2label.get(int(probs.argmax().item()), str(int(probs.argmax().item())))
        predictions.append(
            {
                "guardrail_id": row["seed_guardrail_id"],
                "label": gold,
                "predicted_label": pred_label,
                "predicted_prob": float(probs.max().item()),
                "correct": pred_label == gold,
            }
        )

    result = {
        "classifier_path": str(classifier_path),
        "eval_set": str(in_path),
        "n": len(predictions),
        "overall": _metrics(predictions, positive_label=positive_label),
        "by_policy": _aggregate(predictions, key="guardrail_id", positive_label=positive_label),
        "predictions": predictions,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    o = result["overall"]
    logger.info(
        "%s: n=%d acc=%.3f prec=%.3f rec=%.3f f1=%.3f -> %s",
        Path(classifier_path).name, o["n"], o["accuracy"], o["precision"], o["recall"], o["f1"], out_path,
    )
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rl-config",
        default="config/train_rl/rl.yaml",
        help="RL config (labels, held_out_ids, seed).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Generate + verify the held-out adversarial eval set.")
    b.add_argument("--generator-model", default=None, help="Override cfg.generator.base_model.")
    b.add_argument("--judge-model", default=None, help="Override cfg.verifier.judge_model.")
    b.add_argument("--n-per-seed", type=int, default=16, help="Candidates per held-out seed (default 16).")
    b.add_argument("--max-eval-seeds", type=int, default=120, help="Max held-out seed rows to sample (default 120).")
    b.add_argument("--out", default=DEFAULT_ADV_OUT)

    s = sub.add_parser("score", help="Score a classifier on the frozen adversarial set.")
    s.add_argument("--classifier-path", required=True)
    s.add_argument("--in-path", default=DEFAULT_ADV_OUT)
    s.add_argument("--out", required=True)
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    cfg = load_rl_config(args.rl_config, require_pipeline_inputs=False)

    if args.cmd == "build":
        build(
            cfg,
            generator_model=args.generator_model or cfg.generator.base_model,
            judge_model=args.judge_model or cfg.verifier.judge_model,
            n_per_seed=args.n_per_seed,
            out_path=Path(args.out),
            max_eval_seeds=args.max_eval_seeds,
        )

    elif args.cmd == "score":
        score(cfg, args.classifier_path, Path(args.in_path), Path(args.out))


if __name__ == "__main__":
    main()
