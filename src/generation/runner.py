import argparse
import concurrent.futures
import hashlib
import json
import logging
import os
import random
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import yaml
from langchain_core.language_models.chat_models import BaseChatModel

from common.llm import get_llm
from common.settings import LLMConfig, Settings
from generation.debate import DebateValidator
from generation.policies import (
    load_policy_files,
    policy_to_evaluation_criterion,
    policy_to_generation_criterion,
    seed_examples,
)
from generation.prompting import GenerationPrompter, invoke_prompt
from generation.schemas import AcceptedSample, GeneratedSample, RejectedAttempt

logger = logging.getLogger(__name__)

SAMPLE_FORMAT = (
    "a user-agent chat transcript using <User>...</User> and <Agent>...</Agent> tags"
)

LENGTH_GUIDANCE = [
    "Follow the format of the EXAMPLE but make the transcript richer and more detailed than the EXAMPLE.",
    "Follow the format of the EXAMPLE and keep it focused and concise while still requiring careful reading.",
]
STRUCTURE_GUIDANCE = [
    "Mirror the turn structure of the EXAMPLE, with ordinary setup before the policy-relevant moment.",
    "Mirror the format of the EXAMPLE, using earlier content to build the context needed to judge the decisive part.",
    "Use a short multi-turn back-and-forth where the decisive exchange emerges after a couple of clarifying turns.",
]
SIGNAL_PLACEMENTS = [
    "Place the policy-relevant evidence early in the transcript.",
    "Place the policy-relevant evidence in the middle of the transcript, surrounded by plausible context.",
    "Place the policy-relevant evidence near the end of the transcript after benign setup.",
]
TRUE_BOUNDARY_STYLES = [
    "Make the violation clear; include policy-adjacent benign details that a classifier must ignore.",
    "Use an edge case that still satisfies the violating label after careful reading.",
]
FALSE_BOUNDARY_STYLES = [
    "Make the sample compliant while keeping it close to the boundary, so a shallow keyword match could be wrong.",
    "Use a near miss where the decisive detail prevents the criterion from holding.",
]
SURFACE_VARIATIONS = [
    "Vary the topic, names, amounts, locations, tone, and opening phrasing from the example.",
    "Avoid starting with generic requests such as 'Can you help me' unless the example specifically requires that style.",
    "Use concrete realistic details rather than placeholders, while avoiding references to policies, labels, or test cases.",
    "Vary the speaker's register and formality, from terse and casual to formal and detailed.",
]

# Optional surface-level disguise of the policy-relevant content. Applied only
# to a fraction of samples (obfuscation_fraction) so most stay plainly worded.
OBFUSCATION_TECHNIQUES = [
    "Wrap the policy-relevant content in role-play or a hypothetical framing.",
    "Have the user mangle the sensitive words in their request — for example leetspeak (m@ke), letter substitutions, transposed or scrambled interior letters (mkae), or inserted spacing/punctuation (h-o-w) — the way someone evading a keyword filter would, keeping the words readable so the meaning stays recoverable.",
]


def sample_schema(target_label: str) -> str:
    return json.dumps(
        {
            "input_block": "<User>user message</User>\n<Agent>agent response</Agent>",
            "label": target_label,
            "reasoning": "why the target label is correct",
        }
    )


def generation_guidance(
    target_label: str, rng: random.Random, obfuscation_fraction: float = 0.0
) -> str:
    boundary_styles = (
        TRUE_BOUNDARY_STYLES if target_label == "True" else FALSE_BOUNDARY_STYLES
    )
    parts = [
        rng.choice(LENGTH_GUIDANCE),
        rng.choice(STRUCTURE_GUIDANCE),
        rng.choice(SIGNAL_PLACEMENTS),
        rng.choice(boundary_styles),
        rng.choice(SURFACE_VARIATIONS),
    ]
    if obfuscation_fraction and rng.random() < obfuscation_fraction:
        technique = rng.choice(OBFUSCATION_TECHNIQUES)
        parts.append(
            f"{technique} The {target_label} label must still be correct after reading "
            "past the disguise; obscure the surface only and do not make the case genuinely ambiguous."
        )
    return "\n".join(f"- {part}" for part in parts)


def generate_sample(
    llm: BaseChatModel,
    prompter: GenerationPrompter,
    criterion: str,
    seed_example: str,
    target_label: str,
    guidance: str,
) -> GeneratedSample:
    sample = invoke_prompt(
        llm,
        prompter,
        "initial_generation",
        GeneratedSample,
        evaluation_criterion=criterion,
        sample_format=SAMPLE_FORMAT,
        sample_schema=sample_schema(target_label),
        target_label=target_label,
        input_block=seed_example,
        generation_guidance=guidance,
    )
    sample.label = target_label
    return sample


def refine_sample(
    llm: BaseChatModel,
    prompter: GenerationPrompter,
    criterion: str,
    seed_example: str,
    target_label: str,
    previous_sample: GeneratedSample,
    dissenting_reasoning: str,
    guidance: str,
) -> GeneratedSample:
    sample = invoke_prompt(
        llm,
        prompter,
        "refinement",
        GeneratedSample,
        evaluation_criterion=criterion,
        sample_format=SAMPLE_FORMAT,
        sample_schema=sample_schema(target_label),
        target_label=target_label,
        input_block=seed_example,
        failed_sample=previous_sample.input_block,
        dissenting_reasoning=dissenting_reasoning,
        generation_guidance=guidance,
    )
    sample.label = target_label
    return sample


def append_jsonl(path: Path, data: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(data) + "\n")


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def load_jsonl_dicts(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_accepted(path: Path) -> list[AcceptedSample]:
    if not path.exists():
        return []
    accepted = []
    with open(path) as f:
        for line in f:
            if line.strip():
                accepted.append(AcceptedSample.model_validate_json(line))
    return accepted


def write_training_csv(path: Path, samples: list[AcceptedSample]) -> None:
    rows = [
        {
            "guardrail_id": sample.guardrail_id,
            "policy_prompt": sample.policy_prompt,
            "input_block": sample.input_block,
            "label": sample.label,
            "reasoning": sample.reasoning,
        }
        for sample in samples
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


@dataclass
class WorkerResult:
    guardrail_id: str
    attempts: int
    accepted_count: int


def _stable_seed(base_seed: int, guardrail_id: str) -> int:
    digest = hashlib.md5(guardrail_id.encode("utf-8")).hexdigest()[:8]
    return base_seed + int(digest, 16)


def _per_guardrail_paths(out_dir: Path, guardrail_id: str) -> tuple[Path, Path, Path]:
    guardrail_dir = out_dir / "by_guardrail"
    accepted_path = guardrail_dir / f"accepted_samples.{guardrail_id}.jsonl"
    rejected_path = guardrail_dir / f"rejected_attempts.{guardrail_id}.jsonl"
    debates_path = guardrail_dir / f"debates.{guardrail_id}.jsonl"
    return accepted_path, rejected_path, debates_path


def run_guardrail_generation(
    policy: dict,
    generation_config: dict,
    llm_config: dict,
    target_size: int,
    accepted_count: int,
    labels: list[str],
    out_dir: str,
    worker_seed: int,
) -> WorkerResult:
    guardrail_id = policy["guardrail_id"]
    if accepted_count >= target_size:
        return WorkerResult(
            guardrail_id=guardrail_id, attempts=0, accepted_count=accepted_count
        )

    out_dir_path = Path(out_dir)
    accepted_path, rejected_path, debates_path = _per_guardrail_paths(
        out_dir_path, guardrail_id
    )

    settings = Settings()
    llm = get_llm(settings, LLMConfig(**llm_config))
    prompter = GenerationPrompter.from_default_file()
    rng = random.Random(worker_seed)

    generation_criterion = policy_to_generation_criterion(policy)
    evaluation_criterion = policy_to_evaluation_criterion(policy)
    validator = DebateValidator(
        llm,
        prompter,
        evaluation_criterion,
        labels,
        max_rounds=generation_config.get("max_debate_rounds", 2),
        judges=generation_config.get("judges"),
    )

    attempts = 0
    max_attempts = generation_config.get(
        "max_attempts_per_policy", max(target_size * 20, 1)
    )
    max_refinements = generation_config.get("max_refinements", 2)
    obfuscation_fraction = generation_config.get("obfuscation_fraction", 0.0)
    while accepted_count < target_size and attempts < max_attempts:
        attempts += 1
        target_label = rng.choice(labels)
        seed = rng.choice(seed_examples(policy, target_label))
        guidance = generation_guidance(target_label, rng, obfuscation_fraction)

        sample = generate_sample(
            llm,
            prompter,
            generation_criterion,
            seed,
            target_label,
            guidance,
        )

        final_debate = None
        for refinement_round in range(max_refinements + 1):
            final_debate = validator.validate(sample, target_label)
            append_jsonl(
                debates_path,
                {
                    "guardrail_id": guardrail_id,
                    "attempt": attempts,
                    "refinement_round": refinement_round,
                    "target_label": target_label,
                    "sample": sample.model_dump(),
                    "debate": final_debate.model_dump(),
                },
            )

            if final_debate.valid:
                accepted_sample = AcceptedSample(
                    guardrail_id=guardrail_id,
                    policy_prompt=policy["verbatim_excerpt"],
                    input_block=sample.input_block,
                    label=target_label,
                    reasoning=sample.reasoning,
                    refinement_round=refinement_round,
                    debate_path=final_debate.path,
                )
                accepted_count += 1
                append_jsonl(accepted_path, accepted_sample.model_dump())
                logger.info(
                    "Accepted %s: %d/%d", guardrail_id, accepted_count, target_size
                )
                break

            if refinement_round < max_refinements:
                sample = refine_sample(
                    llm,
                    prompter,
                    generation_criterion,
                    seed,
                    target_label,
                    sample,
                    final_debate.feedback,
                    guidance,
                )
        else:
            assert final_debate is not None
            rejected = RejectedAttempt(
                guardrail_id=guardrail_id,
                sample=sample,
                target_label=target_label,
                debate=final_debate,
            )
            append_jsonl(rejected_path, rejected.model_dump())
            logger.info("Rejected %s: attempt %d", guardrail_id, attempts)

    return WorkerResult(
        guardrail_id=guardrail_id, attempts=attempts, accepted_count=accepted_count
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", required=True, help="Path to generation config YAML"
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=None,
        help="Override configured target size per guardrail",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate config and exit before LLM calls",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="Merge existing by_guardrail caches into final outputs without generating",
    )
    args = parser.parse_args()

    if args.dry_run and args.merge_only:
        raise ValueError("--dry-run and --merge-only cannot be used together")

    config_path = Path(args.config)
    with open(config_path) as f:
        config = yaml.safe_load(f)

    classification_type = config["classification_type"]
    if classification_type != "input_block":
        raise ValueError(
            "Policy-file generation only supports classification_type: input_block"
        )
    labels = config.get("labels", ["True", "False"])
    if labels != ["True", "False"]:
        raise ValueError("Policy-file generation requires labels: ['True', 'False']")

    generation_config = config["generation"]
    target_size = args.target_size or generation_config["target_size_per_policy"]
    policies = load_policy_files(config["policy_files"], config_path.parent)

    if args.dry_run:
        missing_safe = [
            policy["guardrail_id"]
            for policy in policies
            if not policy.get("safe_examples")
        ]
        logger.info(
            "Loaded %d policies; target size per guardrail is %d",
            len(policies),
            target_size,
        )
        logger.info("Policies without safe_examples: %d", len(missing_safe))
        return

    out_dir = Path(generation_config["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    accepted_path = out_dir / "accepted_samples.jsonl"
    rejected_path = out_dir / "rejected_attempts.jsonl"
    debates_path = out_dir / "debates.jsonl"
    guardrail_dir = out_dir / "by_guardrail"
    guardrail_dir.mkdir(parents=True, exist_ok=True)
    accepted_by_guardrail = {policy["guardrail_id"]: 0 for policy in policies}
    for policy in policies:
        guardrail_id = policy["guardrail_id"]
        accepted_path_guardrail, _, _ = _per_guardrail_paths(out_dir, guardrail_id)
        accepted_by_guardrail[guardrail_id] = len(
            load_accepted(accepted_path_guardrail)
        )

    max_workers = generation_config.get("max_workers")
    if max_workers is None:
        max_workers = min(len(policies), os.cpu_count() or 1)

    attempts_by_guardrail = {}
    if not args.merge_only:
        base_seed = generation_config.get("random_seed", 0)
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            futures = []
            for policy in policies:
                guardrail_id = policy["guardrail_id"]
                futures.append(
                    executor.submit(
                        run_guardrail_generation,
                        policy,
                        generation_config,
                        config["llm"],
                        target_size,
                        accepted_by_guardrail.get(guardrail_id, 0),
                        labels,
                        str(out_dir),
                        _stable_seed(base_seed, guardrail_id),
                    )
                )

            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                attempts_by_guardrail[result.guardrail_id] = result.attempts

    merged_accepted = []
    for policy in policies:
        guardrail_id = policy["guardrail_id"]
        accepted_path_guardrail, _, _ = _per_guardrail_paths(out_dir, guardrail_id)
        merged_accepted.extend(load_accepted(accepted_path_guardrail))

    merged_rejected = []
    merged_debates = []
    for policy in policies:
        guardrail_id = policy["guardrail_id"]
        _, rejected_path_guardrail, debates_path_guardrail = _per_guardrail_paths(
            out_dir, guardrail_id
        )
        merged_rejected.extend(load_jsonl_dicts(rejected_path_guardrail))
        merged_debates.extend(load_jsonl_dicts(debates_path_guardrail))

    write_jsonl(accepted_path, [sample.model_dump() for sample in merged_accepted])
    write_jsonl(rejected_path, merged_rejected)
    write_jsonl(debates_path, merged_debates)

    accepted_by_guardrail = {policy["guardrail_id"]: 0 for policy in policies}
    for sample in merged_accepted:
        accepted_by_guardrail[sample.guardrail_id] = (
            accepted_by_guardrail.get(sample.guardrail_id, 0) + 1
        )

    write_training_csv(out_dir / "training.csv", merged_accepted)
    with open(out_dir / "summary.json", "w") as f:
        json.dump(
            {
                "policies": len(policies),
                "target_size_per_policy": target_size,
                "accepted": len(merged_accepted),
                "accepted_by_guardrail": accepted_by_guardrail,
                "attempts_by_guardrail": attempts_by_guardrail,
            },
            f,
            indent=2,
        )

    incomplete = {
        guardrail_id: count
        for guardrail_id, count in accepted_by_guardrail.items()
        if count < target_size
    }
    if incomplete:
        raise RuntimeError(
            f"Generated too few accepted samples for guardrails: {incomplete}"
        )


if __name__ == "__main__":
    main()
