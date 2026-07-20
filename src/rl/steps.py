"""The four data stages of one RL iteration, behind one dispatcher:

    python -m src.rl.steps {generate|verify|classify|preference} --iter N

- generate    sample (P', x̃, label, reasoning) candidates -> candidates.jsonl
- verify      judge-verify each against the generated P' (never the seed) -> verified.jsonl
- classify    score each candidate with the current classifier -> classified.jsonl
- preference  partition into DPO pairs + classifier mis_increment -> preference/mis_increment.jsonl
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

from generation.prompting import _extract_json
from training.runner import build_prompter

from src.rl.config import (
    RLConfig,
    _resolve,
    classifier_for_iter,
    generator_for_iter,
    iteration_paths,
    load_rl_config,
)
from src.rl.prompts import format_for_judge, format_for_pair
from src.rl.utils import (
    iter_jsonl,
    load_classifier,
    passes_light_filters,
    render_classifier_text,
    write_jsonl,
)


logger = logging.getLogger(__name__)


# ===========================================================================
# generate
# ===========================================================================


def load_seed_units(
    samples_path: Path,
    held_out_ids: set[str],
    max_per_guardrail: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Read accepted_samples into seed units (policy_prompt + input_block per row).
    Held-out guardrails are dropped; up to ``max_per_guardrail`` rows per guardrail (0=all)."""
    rows = [r for r in iter_jsonl(samples_path) if r.get("guardrail_id") not in held_out_ids]
    by_gid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if r.get("policy_prompt") and r.get("input_block"):
            by_gid[r["guardrail_id"]].append(r)

    units: list[dict[str, Any]] = []
    for gid, grp in by_gid.items():
        if max_per_guardrail and len(grp) > max_per_guardrail:
            grp = rng.sample(grp, max_per_guardrail)
        for r in grp:
            units.append(
                {"guardrail_id": gid, "seed_policy": r["policy_prompt"], "seed_example": r["input_block"]}
            )
    return units


def build_prompts_from_units(
    units: list[dict[str, Any]],
    n_per_seed: int,
    tokenizer: Any,
    rng: random.Random,
) -> tuple[list[str], list[dict[str, Any]]]:
    """Build ``n_per_seed`` prompts per seed unit, each with a random target label.
    Returns equal-length ``(prompts, metadata)``; metadata carries provenance."""
    prompts: list[str] = []
    metadata: list[dict[str, Any]] = []
    for unit in units:
        for _ in range(n_per_seed):
            target_label = rng.choice(["True", "False"])
            prompts.append(format_for_pair(unit["seed_policy"], unit["seed_example"], target_label, tokenizer))
            metadata.append(
                {
                    "seed_guardrail_id": unit["guardrail_id"],
                    "seed_policy": unit["seed_policy"],
                    "seed_example": unit["seed_example"],
                    "target_label": target_label,
                }
            )
    return prompts, metadata


def parse_and_filter(
    raw_outputs: list[str],
    metadata: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Parse generator outputs, apply light filters, return (kept, diagnostics)."""
    kept: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    for raw, meta in zip(raw_outputs, metadata):
        try:
            parsed = _extract_json(raw)
        except (ValueError, json.JSONDecodeError) as exc:
            diagnostics.append({**meta, "raw": raw, "reject_reason": "parse_error", "error": str(exc)[:200]})
            continue
        if not isinstance(parsed, dict):
            diagnostics.append({**meta, "raw": raw, "reject_reason": "not_object"})
            continue
        required = ("policy_prompt", "input_block", "label", "reasoning")
        missing = [k for k in required if not parsed.get(k)]
        if missing:
            diagnostics.append({**meta, "raw": raw, "reject_reason": f"missing_fields:{missing}"})
            continue
        if str(parsed["label"]).strip() not in ("True", "False"):
            diagnostics.append({**meta, "raw": raw, "reject_reason": f"bad_label:{parsed['label']!r}"})
            continue

        ok, reason = passes_light_filters(parsed["input_block"])
        if not ok:
            diagnostics.append({**meta, "raw": raw, "reject_reason": f"filter:{reason}"})
            continue

        kept.append(
            {
                **meta,
                "policy_prompt": str(parsed["policy_prompt"]).strip(),
                "input_block": str(parsed["input_block"]).strip(),
                "label": str(parsed["label"]).strip(),
                "reasoning": str(parsed["reasoning"]).strip(),
            }
        )
    return kept, diagnostics


def run_generate(cfg: RLConfig, t: int, generator_model: str) -> dict[str, int]:
    """Run one full generation pass for iteration ``t``. Returns a counts dict."""
    from src.rl.hf_generate import generate_distributed, load_tokenizer

    paths = iteration_paths(cfg, t)
    paths.iter_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(cfg.seed + t)
    held_out = set(cfg.held_out_ids)

    # Prompt rendering only needs the tokenizer/chat template; the model itself is
    # loaded per-GPU inside the worker processes (see generate_distributed).
    tokenizer = load_tokenizer(generator_model)

    samples_path = _resolve(cfg.seed_samples_path)
    units = load_seed_units(samples_path, held_out, cfg.generator.max_seeds_per_guardrail, rng)
    prompts, metadata = build_prompts_from_units(units, cfg.generator.n_per_seed, tokenizer, rng)
    gpus = cfg.gen_gpus()
    logger.info(
        "iter%d: built %d prompts from %d seed units across %d guardrails (%s); generator=%s on GPUs %s",
        t, len(prompts), len(units), len({u["guardrail_id"] for u in units}), samples_path, generator_model, gpus,
    )

    raw_outputs = generate_distributed(
        generator_model,
        prompts,
        gpus=gpus,
        max_new_tokens=cfg.generator.max_tokens,
        temperature=cfg.generator.temperature,
        top_p=cfg.generator.top_p,
        do_sample=cfg.generator.temperature > 0,
        seed=cfg.seed + t,
        batch_size=cfg.generator.batch_size,
    )
    kept, diagnostics = parse_and_filter(raw_outputs, metadata)

    n_kept = write_jsonl(kept, paths.candidates)
    n_diag = write_jsonl(diagnostics, paths.iter_dir / "candidates_diagnostics.jsonl")
    logger.info("iter%d: %d kept, %d rejected -> %s", t, n_kept, n_diag, paths.candidates)
    return {"prompts": len(prompts), "kept": n_kept, "rejected": n_diag}


# ===========================================================================
# verify
# ===========================================================================


def build_judge_prompts(rows: list[dict[str, Any]], tokenizer: Any, cfg: RLConfig) -> list[str]:
    return [
        format_for_judge(
            policy_prompt=row["policy_prompt"],
            input_block=row["input_block"],
            tokenizer=tokenizer,
            labels=list(cfg.labels)[::-1],  # ["True","False"] for the judge schema
            persona=cfg.verifier.persona,
            persona_instructions=cfg.verifier.persona_instructions,
        )
        for row in rows
    ]


def parse_judge_output(raw: str) -> dict[str, Any]:
    try:
        parsed = _extract_json(raw)
    except (ValueError, json.JSONDecodeError) as exc:
        return {"label": None, "reasoning": None, "confidence": None, "parse_error": str(exc)[:200]}
    if not isinstance(parsed, dict):
        return {"label": None, "reasoning": None, "confidence": None, "parse_error": "not_object"}
    return {
        "label": str(parsed.get("label", "")).strip() or None,
        "reasoning": parsed.get("reasoning"),
        "confidence": parsed.get("confidence"),
        "parse_error": None,
    }


def annotate(rows: list[dict[str, Any]], raw_outputs: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row, raw in zip(rows, raw_outputs):
        parsed = parse_judge_output(raw)
        verified = parsed["label"] == row["target_label"] if parsed["label"] else False
        out.append(
            {
                **row,
                "judge_label": parsed["label"],
                "judge_reasoning": parsed["reasoning"],
                "judge_confidence": parsed["confidence"],
                "judge_raw": raw if parsed["parse_error"] else None,
                "judge_parse_error": parsed["parse_error"],
                "verified": verified,
            }
        )
    return out


def run_verify(cfg: RLConfig, t: int, judge_model: str) -> dict[str, int]:
    """Verify candidates for iteration ``t``. Returns counts dict."""
    from src.rl.hf_generate import generate_distributed, load_tokenizer

    paths = iteration_paths(cfg, t)
    if not paths.candidates.exists():
        raise FileNotFoundError(f"candidates.jsonl missing for iter{t}: {paths.candidates}")

    rows = list(iter_jsonl(paths.candidates))
    if not rows:
        logger.warning("iter%d: no candidates to verify", t)
        write_jsonl([], paths.verified)
        return {"rows": 0, "verified": 0, "parse_errors": 0}

    tokenizer = load_tokenizer(judge_model)
    prompts = build_judge_prompts(rows, tokenizer, cfg)
    gpus = cfg.gen_gpus()
    logger.info("iter%d: verifying %d candidates with judge %s on GPUs %s", t, len(rows), judge_model, gpus)

    raw_outputs = generate_distributed(
        judge_model,
        prompts,
        gpus=gpus,
        max_new_tokens=cfg.verifier.max_tokens,
        temperature=cfg.verifier.temperature,
        top_p=cfg.verifier.top_p,
        do_sample=cfg.verifier.temperature > 0,
        seed=cfg.seed + t,
        batch_size=cfg.verifier.batch_size,
    )
    annotated = annotate(rows, raw_outputs)

    write_jsonl(annotated, paths.verified)
    n_verified = sum(1 for r in annotated if r["verified"])
    n_parse_errors = sum(1 for r in annotated if r["judge_parse_error"])
    logger.info(
        "iter%d: verified %d/%d candidates (parse_errors=%d) -> %s",
        t, n_verified, len(annotated), n_parse_errors, paths.verified,
    )
    return {"rows": len(annotated), "verified": n_verified, "parse_errors": n_parse_errors}


# ===========================================================================
# classify
# ===========================================================================


def run_classify(cfg: RLConfig, t: int, classifier_path: str) -> dict[str, int]:
    paths = iteration_paths(cfg, t)
    if not paths.verified.exists():
        raise FileNotFoundError(f"verified.jsonl missing for iter{t}: {paths.verified}")

    logger.info("iter%d: loading classifier %s", t, classifier_path)
    tokenizer, model, id2label = load_classifier(classifier_path)
    device = next(model.parameters()).device

    positive_label = cfg.labels[1]
    positive_id = next((i for i, l in id2label.items() if l == positive_label), 1)
    prompter = build_prompter()
    rows = list(iter_jsonl(paths.verified))

    annotated = []
    for row in rows:
        text = render_classifier_text(row["policy_prompt"], row["input_block"], list(cfg.labels), prompter)
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=cfg.classifier.max_seq_length
        ).to(device)
        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits[0], dim=-1).cpu()
        pred_id = int(probs.argmax().item())
        annotated.append(
            {
                **row,
                "predicted_label": id2label.get(pred_id, str(pred_id)),
                "predicted_prob": float(probs[pred_id].item()),
                "positive_prob": float(probs[positive_id].item()) if positive_id < len(probs) else 0.0,
            }
        )

    n_written = write_jsonl(annotated, paths.classified)
    logger.info("iter%d: classified %d rows -> %s", t, n_written, paths.classified)
    return {"rows": n_written}


# ===========================================================================
# preference
# ===========================================================================


def partition(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Bucket rows into level1 / level2 / level3 / drop.

    - level1 (chosen):      verified AND predicted_label != target_label (fooled C)
    - level2 (rejected):    verified AND predicted_label == target_label (C correct)
    - level3 (weak chosen): not verified AND predicted_label != target_label
    - drop:                 not verified AND predicted_label == target_label
    """
    buckets: dict[str, list[dict[str, Any]]] = {"level1": [], "level2": [], "level3": [], "drop": []}
    for row in rows:
        verified = bool(row.get("verified"))
        correct = row.get("predicted_label") == row.get("target_label")
        if verified and not correct:
            buckets["level1"].append(row)
        elif verified and correct:
            buckets["level2"].append(row)
        elif not verified and not correct:
            buckets["level3"].append(row)
        else:
            buckets["drop"].append(row)
    return buckets


def make_pairs(
    grouped: dict[str, list[dict[str, Any]]],
    rng: random.Random,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Produce (chosen, rejected) DPO pairs for one seed group.

    Strategy: shuffle L1 and L2; pair them 1-1 up to ``min(|L1|, |L2|)``. Any
    excess L1 pairs with shuffled L3. If |L1|==0 the group yields no pairs.
    """
    level1 = list(grouped["level1"])
    level2 = list(grouped["level2"])
    level3 = list(grouped["level3"])
    rng.shuffle(level1)
    rng.shuffle(level2)
    rng.shuffle(level3)

    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    n_standard = min(len(level1), len(level2))
    for chosen, rejected in zip(level1[:n_standard], level2[:n_standard]):
        pairs.append((chosen, rejected))

    leftover_l1 = level1[n_standard:]
    n_weak = min(len(leftover_l1), len(level3))
    for chosen, rejected in zip(leftover_l1[:n_weak], level3[:n_weak]):
        pairs.append((chosen, rejected))
    return pairs


def render_generator_output(row: dict[str, Any]) -> str:
    """Reconstruct the full JSON string the generator produced for this row."""
    return json.dumps(
        {
            "policy_prompt": row["policy_prompt"],
            "input_block": row["input_block"],
            "label": row["label"],
            "reasoning": row.get("reasoning", ""),
        },
        ensure_ascii=False,
    )


def build_preference_rows(
    rows: list[dict[str, Any]],
    tokenizer: Any,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Group by seed_guardrail_id, partition, pair into preference dicts. chosen/rejected
    are full generator JSON (DPO trains the joint P'/x̃); prompt is re-rendered from provenance."""
    by_seed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_seed[row["seed_guardrail_id"]].append(row)

    out: list[dict[str, Any]] = []
    for gid, group in by_seed.items():
        buckets = partition(group)
        if not buckets["level1"]:
            logger.debug("group %s skipped (no level1)", gid)
            continue
        pairs = make_pairs(buckets, rng)
        for chosen, rejected in pairs:
            prompt = format_for_pair(
                seed_policy=_seed_policy_for(chosen),
                seed_example=chosen["seed_example"],
                target_label=chosen["target_label"],
                tokenizer=tokenizer,
            )
            out.append(
                {
                    "seed_guardrail_id": gid,
                    "prompt": prompt,
                    "chosen": render_generator_output(chosen),
                    "rejected": render_generator_output(rejected),
                }
            )
    return out


def build_mis_increment(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Level-1 rows only, projected into TALMONITOR's classifier schema."""
    incr: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("verified"):
            continue
        if row.get("predicted_label") == row.get("target_label"):
            continue
        incr.append(
            {
                "guardrail_id": row["seed_guardrail_id"],
                "policy_prompt": row["policy_prompt"],
                "input_block": row["input_block"],
                "label": row["target_label"],
            }
        )
    return incr


_seed_policy_cache: dict[str, str] = {}


def _seed_policy_for(row: dict[str, Any]) -> str:
    # Per-row seed policy (accepted_samples varies policy per row); fall back to YAML.
    return row.get("seed_policy") or _seed_policy_cache.get(row["seed_guardrail_id"], "")


def _populate_seed_policy_cache(cfg: RLConfig) -> None:
    from generation.policies import load_policy_files

    yaml_paths = sorted(str(p) for p in cfg.resolved_policies_dir().glob("*.yaml"))
    for policy in load_policy_files(yaml_paths, Path(".")):
        _seed_policy_cache[policy["guardrail_id"]] = policy["verbatim_excerpt"]


def run_preference(cfg: RLConfig, t: int, generator_model: str) -> dict[str, int]:
    from transformers import AutoTokenizer

    paths = iteration_paths(cfg, t)
    if not paths.classified.exists():
        raise FileNotFoundError(f"classified.jsonl missing for iter{t}: {paths.classified}")

    rows = list(iter_jsonl(paths.classified))
    _populate_seed_policy_cache(cfg)
    tokenizer = AutoTokenizer.from_pretrained(generator_model)
    rng = random.Random(cfg.seed + t)

    pref_rows = build_preference_rows(rows, tokenizer, rng)
    mis_rows = build_mis_increment(rows)

    n_pref = write_jsonl(pref_rows, paths.preference)
    n_mis = write_jsonl(mis_rows, paths.mis_increment)

    buckets = partition(rows)
    logger.info(
        "iter%d: rows=%d L1=%d L2=%d L3=%d drop=%d -> preference=%d mis=%d",
        t, len(rows), len(buckets["level1"]), len(buckets["level2"]),
        len(buckets["level3"]), len(buckets["drop"]), n_pref, n_mis,
    )
    return {
        "rows": len(rows),
        "level1": len(buckets["level1"]),
        "level2": len(buckets["level2"]),
        "level3": len(buckets["level3"]),
        "drop": len(buckets["drop"]),
        "pairs": n_pref,
        "mis": n_mis,
    }


# ===========================================================================
# CLI dispatcher
# ===========================================================================

STEPS = ("generate", "verify", "classify", "preference")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=STEPS, help="Which iteration stage to run.")
    parser.add_argument("--rl-config", default="config/train_rl/rl.yaml")
    parser.add_argument("--iter", type=int, required=True, dest="iteration")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    cfg = load_rl_config(args.rl_config)
    t = args.iteration

    if args.step == "generate":
        run_generate(cfg, t, generator_for_iter(cfg, t))
    elif args.step == "verify":
        run_verify(cfg, t, cfg.verifier.judge_model)
    elif args.step == "classify":
        run_classify(cfg, t, classifier_for_iter(cfg, t))
    elif args.step == "preference":
        run_preference(cfg, t, generator_for_iter(cfg, t))


if __name__ == "__main__":
    main()
