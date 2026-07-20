from pathlib import Path
from typing import Any

import yaml


REQUIRED_FIELDS = ("guardrail_id", "verbatim_excerpt", "violation_examples", "safe_examples")


def load_policy_files(paths: list[str], base_dir: Path) -> list[dict[str, Any]]:
    policies = []
    guardrail_ids = set()
    for raw_path in paths:
        path = _resolve_path(raw_path, base_dir)
        with open(path) as f:
            data = yaml.safe_load(f)
        if not isinstance(data, list):
            raise ValueError(f"Policy file must contain a list: {path}")
        for index, policy in enumerate(data):
            if not isinstance(policy, dict):
                raise ValueError(f"Policy at index {index} in {path} is not an object")
            _validate_policy(policy, path, index)
            guardrail_id = policy["guardrail_id"]
            if guardrail_id in guardrail_ids:
                raise ValueError(f"Duplicate guardrail_id: {guardrail_id}")
            guardrail_ids.add(guardrail_id)
            policies.append(policy)
    return policies


def policy_to_generation_criterion(policy: dict[str, Any]) -> str:
    parts = [
        "Label True means the scenario violates this policy. Label False means the scenario complies with it or does not trigger it.",
        "The sample to classify is always a user-agent chat transcript.",
        f"Policy: {policy['verbatim_excerpt']}",
    ]
    return "\n".join(parts)


def policy_to_evaluation_criterion(policy: dict[str, Any]) -> str:
    parts = [
        "Atomic policy for classification.",
        "Label True means the scenario violates this policy. Label False means the scenario complies with it or does not trigger it.",
        "The sample to classify is always a user-agent chat transcript.",
        f"Policy: {policy['verbatim_excerpt']}",
    ]
    return "\n".join(parts)


def seed_examples(policy: dict[str, Any], label: str) -> list[str]:
    if label == "True":
        examples = policy["violation_examples"]
    elif label == "False":
        examples = policy["safe_examples"]
    else:
        raise ValueError(f"Unsupported label for policy seed examples: {label}")
    return [example.strip() for example in examples]


def _resolve_path(raw_path: str, base_dir: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute() or path.exists():
        return path
    return base_dir / path


def _validate_policy(policy: dict[str, Any], path: Path, index: int) -> None:
    missing = [field for field in REQUIRED_FIELDS if field not in policy]
    if missing:
        raise ValueError(f"Policy at index {index} in {path} is missing fields: {missing}")
    for field in ("violation_examples", "safe_examples"):
        if not isinstance(policy[field], list) or not policy[field]:
            raise ValueError(f"Policy {policy['guardrail_id']} in {path} must have {field}")
        for example_index, example in enumerate(policy[field]):
            if not isinstance(example, str) or not example.strip():
                raise ValueError(
                    f"Policy {policy['guardrail_id']} in {path} has invalid {field}[{example_index}]"
                )
            missing_tags = [
                tag
                for tag in ("<User>", "</User>", "<Agent>", "</Agent>")
                if tag not in example
            ]
            if missing_tags:
                raise ValueError(
                    f"Policy {policy['guardrail_id']} in {path} has non-chat {field}[{example_index}]; "
                    f"missing tags: {missing_tags}"
                )
