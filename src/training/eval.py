import argparse
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoTokenizer

from training.modeling import TalMonitorSequenceClassifier
from training.runner import build_prompter, render_input_block_prompt

DEFAULT_EVAL_CONFIG = Path(__file__).parents[2] / "config/train_sft/training_eval.yaml"
DEFAULT_LABELS = ["False", "True"]


def parse_name(name):
    name = name.split("-")
    # remove first and second last, then join with -
    return "-".join(name[1:-2] + name[-1:])


DEFAULT_CHECKPOINTS: dict[str, Path] = {
    # **{f"rl1-iter{i}": Path(f"output/rl/iter{i}/classifier") for i in range(5)},
    # **{f"rl2-iter{i}": Path(f"output/rl/iter{i}/classifier") for i in range(5)},
    # **{f"rl3-iter{i}": Path(f"output/rl3/iter{i}/classifier") for i in range(5)},
    # "gemma-v3": Path("output/training/gemma-3-4b-it-v3/"),
    # "gemma-v1": Path("output/training/gemma-3-4b-it/"),
    **{
        parse_name(x.name): x
        for x in Path("/raid/frontiers_ashoka/policy_monitor/models").iterdir()
        if "7500" in x.name
    },
}
DEFAULT_CHECKPOINTS = {
    name: path
    for name, path in DEFAULT_CHECKPOINTS.items()
    if path.exists() and (path / "config.json").exists()
}
MAX_SEQ_LENGTH = 4096

print(f"Found {len(DEFAULT_CHECKPOINTS)} checkpoints")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        dest="checkpoints",
        action="append",
        type=Path,
        default=None,
        help="Checkpoint directory to evaluate. Repeat for multiple checkpoints. "
        "Defaults to DEFAULT_CHECKPOINTS when omitted.",
    )
    parser.add_argument(
        "--eval-config",
        type=Path,
        default=DEFAULT_EVAL_CONFIG,
        help="YAML file containing labels and test_set cases",
    )
    parser.add_argument("--max-seq-length", type=int, default=MAX_SEQ_LENGTH)
    parser.add_argument("--trust-remote-code", action="store_true")
    return parser.parse_args()


def load_eval_config(path: Path) -> tuple[list[str], list[dict[str, Any]]]:
    if not path.exists():
        raise FileNotFoundError(f"Evaluation config not found: {path}")
    with open(path) as f:
        config = yaml.safe_load(f) or {}

    labels = list(config.get("labels", DEFAULT_LABELS))
    if len(labels) != 2:
        raise ValueError("Evaluation requires exactly two labels")
    test_set = config.get("test_set") or []
    if not isinstance(test_set, list) or not test_set:
        raise ValueError("Evaluation config must include a non-empty test_set list")
    return labels, test_set


def format_table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        widths = [max(width, len(value)) for width, value in zip(widths, row)]

    border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
    header = (
        "| "
        + " | ".join(value.ljust(width) for value, width in zip(headers, widths))
        + " |"
    )
    body = [
        "| "
        + " | ".join(value.ljust(width) for value, width in zip(row, widths))
        + " |"
        for row in rows
    ]
    return "\n".join([border, header, border, *body, border])


def flatten_test_set(
    test_set: list[dict[str, Any]], labels: list[str]
) -> list[dict[str, str]]:
    samples = []
    for policy_index, policy_case in enumerate(test_set, start=1):
        policy = policy_case["policy"]
        for prompt_index, prompt_case in enumerate(
            policy_case.get("prompts") or [], start=1
        ):
            label = prompt_case["label"]
            if label not in labels:
                raise ValueError(
                    f"Unsupported label {label!r}; expected one of {labels}"
                )
            samples.append(
                {
                    "id": f"P{policy_index}.{prompt_index}",
                    "policy": policy,
                    "prompt": prompt_case["prompt"],
                    "label": label,
                }
            )
    if not samples:
        raise ValueError("test_set must include at least one prompt")
    return samples


def model_name(checkpoint_dir: Path | str) -> str:
    checkpoint_path = Path(checkpoint_dir)
    return checkpoint_path.name or str(checkpoint_path)


def evaluate_model(
    model: Any,
    tokenizer: Any,
    samples: list[dict[str, str]],
    labels: list[str],
    max_seq_length: int,
    device: torch.device,
    name: str,
) -> dict[str, Any]:
    """Run the test set through an in-memory model. No disk checkpoint needed."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device).eval()

    prompter = build_prompter()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    positive_label = labels[1]
    positive_id = next(
        (index for index, label in id2label.items() if label == positive_label), 1
    )

    predictions = []
    tp = fp = tn = fn = correct = 0
    for sample in samples:
        text = render_input_block_prompt(
            sample["policy"], sample["prompt"], labels, prompter
        )
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=max_seq_length
        ).to(device)

        with torch.no_grad():
            probs = torch.softmax(model(**inputs).logits[0], dim=-1).cpu()

        pred_id = int(probs.argmax().item())
        pred_label = id2label.get(pred_id, str(pred_id))
        expected = sample["label"]
        is_correct = pred_label == expected
        correct += int(is_correct)

        if pred_label == positive_label and expected == positive_label:
            tp += 1
        elif pred_label == positive_label:
            fp += 1
        elif expected == positive_label:
            fn += 1
        else:
            tn += 1

        predictions.append(
            {
                "id": sample["id"],
                "label": pred_label,
                "expected": expected,
                "confidence": float(probs[pred_id].item()),
                "positive_prob": (
                    float(probs[positive_id].item())
                    if positive_id < len(probs)
                    else 0.0
                ),
                "correct": is_correct,
            }
        )

    total = len(samples)
    return {
        "model": name,
        "predictions": predictions,
        "accuracy": correct / total,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def evaluate_checkpoint(
    checkpoint_dir: Path | str,
    samples: list[dict[str, str]],
    labels: list[str],
    args: argparse.Namespace,
    device: torch.device,
    name: str | None = None,
) -> dict[str, Any]:
    checkpoint_path = Path(checkpoint_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_path, trust_remote_code=args.trust_remote_code
    )
    # Labels live in TalMonitorClassifierConfig, so they load from config reliably.
    model = TalMonitorSequenceClassifier.from_pretrained(
        checkpoint_path, trust_remote_code=args.trust_remote_code
    )
    return evaluate_model(
        model,
        tokenizer,
        samples,
        labels,
        args.max_seq_length,
        device,
        name or model_name(checkpoint_path),
    )


def evaluate_local_checkpoint(
    checkpoint_dir: Path | str,
    max_seq_length: int = MAX_SEQ_LENGTH,
    trust_remote_code: bool = False,
    device: torch.device | None = None,
    eval_config: Path = DEFAULT_EVAL_CONFIG,
) -> dict[str, Any]:
    labels, test_set = load_eval_config(eval_config)
    args = argparse.Namespace(
        max_seq_length=max_seq_length, trust_remote_code=trust_remote_code
    )
    return evaluate_checkpoint(
        checkpoint_dir,
        flatten_test_set(test_set, labels),
        labels,
        args,
        device or torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )


def print_prompt_key(samples: list[dict[str, str]]) -> None:
    rows = [[sample["id"], sample["label"], sample["prompt"]] for sample in samples]
    print("\nPrompt Key")
    print(format_table(["ID", "Label", "Prompt"], rows))


def print_prediction_table(
    results: list[dict[str, Any]], samples: list[dict[str, str]]
) -> None:
    headers = ["Prompt", "Label", *[result["model"] for result in results]]
    rows = []
    for sample_index, sample in enumerate(samples):
        cells = []
        for result in results:
            prediction = result["predictions"][sample_index]
            marker = "" if prediction["correct"] else "*"
            cells.append(
                f"{prediction['label']}{marker} ({prediction['confidence']:.3f})"
            )
        rows.append([sample["id"], sample["label"], *cells])
    print("\nPredictions")
    print(format_table(headers, rows))
    print("* = incorrect prediction")


def f1_score(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def print_metrics_table(results: list[dict[str, Any]]) -> None:
    rows = [
        [
            result["model"],
            f"{f1_score(result['precision'], result['recall']):.3f}",
            f"{result['accuracy']:.3f}",
            f"{result['recall']:.3f}",
            f"{result['precision']:.3f}",
        ]
        for result in sorted(
            results,
            key=lambda r: f1_score(r["precision"], r["recall"]),
            reverse=True,
        )
    ]
    print("\nMetrics")
    print(format_table(["Model", "F1", "Accuracy", "Recall", "Precision"], rows))


def main() -> None:
    args = parse_args()
    checkpoints = (
        {model_name(p): p for p in args.checkpoints}
        if args.checkpoints
        else DEFAULT_CHECKPOINTS
    )
    if not checkpoints:
        raise ValueError("Pass at least one --checkpoint")

    labels, test_set = load_eval_config(args.eval_config)
    samples = flatten_test_set(test_set, labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results = [
        evaluate_checkpoint(checkpoint_dir, samples, labels, args, device, name)
        for name, checkpoint_dir in checkpoints.items()
    ]

    print_prediction_table(results, samples)
    print_metrics_table(results)


if __name__ == "__main__":
    main()
