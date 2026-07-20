import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from transformers import DataCollatorWithPadding, Trainer

from common.prompt_utils import render_jinja_prompts
from data_handler.data_class import InputBlockClsfSample
from evaluations.inference_prompter import Prompter
from training.modeling import bool_config, build_optimizer, compute_metrics, learning_rates, load_tokenizer_and_model, training_arguments


logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("guardrail_id", "policy_prompt", "input_block", "label")
DEFAULT_LABELS = ["False", "True"]


@dataclass(frozen=True)
class PreparedData:
    path: Path
    all_df: pd.DataFrame
    train_df: pd.DataFrame
    eval_df: pd.DataFrame
    labels: list[str]
    label2id: dict[str, int]
    id2label: dict[int, str]


class ClassificationDataset:
    def __init__(self, texts: list[str], labels: list[int], tokenizer: Any, max_length: int):
        self.encodings = tokenizer(texts, max_length=max_length, truncation=True, padding=False)
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, Any]:
        item = {key: values[index] for key, values in self.encodings.items()}
        item["labels"] = self.labels[index]
        return item


def load_training_df(path: Path, labels: list[str]) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Training data not found: {path}")
    if path.suffix == ".jsonl":
        df = pd.read_json(path, lines=True)
    elif path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("Training data must be a .jsonl or .csv file")

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"Training data is missing required columns: {missing}")
    if df.empty:
        raise ValueError(f"Training data is empty: {path}")

    df = df.loc[:, REQUIRED_COLUMNS].copy()
    for column in REQUIRED_COLUMNS:
        df[column] = df[column].astype(str).str.strip()
    invalid_labels = sorted(set(df["label"]) - set(labels))
    if invalid_labels:
        raise ValueError(f"Unsupported labels in training data: {invalid_labels}. Expected {labels}")
    return df


def split_df(df: pd.DataFrame, validation_size: float | int | str | None, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if validation_size in (None, 0, 0.0, "0", "0.0"):
        return df.reset_index(drop=True), df.iloc[0:0].copy()

    guardrail_ids = pd.Series(sorted(df["guardrail_id"].unique()))
    if len(guardrail_ids) < 2:
        raise ValueError("Policy-level validation split requires at least two guardrail_id values")

    if isinstance(validation_size, int) or (isinstance(validation_size, str) and validation_size.isdigit()):
        eval_count = int(validation_size)
    else:
        validation_fraction = float(validation_size)
        if not 0 < validation_fraction < 1:
            raise ValueError("data.validation_size must be 0, a fraction between 0 and 1, or an integer count")
        eval_count = round(len(guardrail_ids) * validation_fraction)

    eval_count = min(max(eval_count, 1), len(guardrail_ids) - 1)
    eval_guardrail_ids = set(guardrail_ids.sample(n=eval_count, random_state=seed))
    return (
        df[~df["guardrail_id"].isin(eval_guardrail_ids)].reset_index(drop=True),
        df[df["guardrail_id"].isin(eval_guardrail_ids)].reset_index(drop=True),
    )


def prepare_data(config: dict[str, Any], data_path_override: str | None) -> PreparedData:
    data_config = config.get("data", {})
    labels = list(data_config.get("labels", DEFAULT_LABELS))
    if len(labels) != 2:
        raise ValueError("Binary classifier training requires exactly two labels")

    data_path = Path(data_path_override or data_config["path"])
    df = load_training_df(data_path, labels)
    seed = int(data_config.get("seed", 0))
    train_df, eval_df = split_df(df, data_config.get("validation_size", 0.1), seed)

    num_samples = data_config.get("num_training_samples")
    if num_samples is not None:
        n = int(num_samples)
        if n < 1:
            raise ValueError("data.num_training_samples must be a positive integer")
        if len(train_df) < n:
            raise ValueError(f"num_training_samples={n} exceeds available training rows ({len(train_df)})")
        train_df = train_df.sample(n=n, random_state=seed).reset_index(drop=True)

    label2id = {label: index for index, label in enumerate(labels)}
    id2label = {index: label for label, index in label2id.items()}
    return PreparedData(data_path, df, train_df, eval_df, labels, label2id, id2label)


def label_counts(df: pd.DataFrame) -> dict[str, int]:
    return df["label"].value_counts().to_dict()


def log_data_summary(data: PreparedData, include_all_rows: bool = False) -> None:
    if include_all_rows:
        logger.info("Rows: %d %s", len(data.all_df), label_counts(data.all_df))
    logger.info("Training rows: %d %s", len(data.train_df), label_counts(data.train_df))
    logger.info("Validation rows: %d %s", len(data.eval_df), label_counts(data.eval_df))
    logger.info("Training policies: %d", data.train_df["guardrail_id"].nunique())
    logger.info("Validation policies: %d %s", data.eval_df["guardrail_id"].nunique(), sorted(data.eval_df["guardrail_id"].unique()))


def build_prompter() -> Prompter:
    prompts_file = Path(__file__).parent.parent.parent / "prompts/prompts.yaml.jinja2"
    return Prompter(render_jinja_prompts(str(prompts_file), {"clsf_type": "input_block"}))


def render_input_block_prompt(policy_prompt: str, input_block: str, labels: list[str], prompter: Prompter) -> str:
    sample = InputBlockClsfSample(input_block=input_block, rule=policy_prompt, labels=labels)
    return f"{prompter.get_system_msg(sample)}\n\n{prompter.get_user_msg(sample)}"


def render_texts(df: pd.DataFrame, labels: list[str], prompter: Prompter) -> list[str]:
    return [render_input_block_prompt(row.policy_prompt, row.input_block, labels, prompter) for row in df.itertuples(index=False)]


def make_dataset(
    df: pd.DataFrame,
    labels: list[str],
    label2id: dict[str, int],
    tokenizer: Any,
    max_seq_length: int,
    prompter: Prompter,
) -> ClassificationDataset:
    texts = render_texts(df, labels, prompter)
    return ClassificationDataset(texts, [label2id[label] for label in df["label"]], tokenizer, max_seq_length)


def resolve_training_config(config: dict[str, Any], output_dir: str | None) -> dict[str, Any]:
    values = dict(config.get("training", {}))
    if output_dir:
        values["output_dir"] = output_dir
    return values


def resolve_model_name(config: dict[str, Any], model_name: str | None) -> str:
    name = model_name or config.get("model", {}).get("name")
    if not name:
        raise ValueError("Model name is required via model.name or --model-name")
    return name


def metadata(data: PreparedData, base_model: str, max_seq_length: int, training_args: Any, train_config: dict[str, Any]) -> dict[str, Any]:
    base_lr, classifier_lr = learning_rates(train_config)
    return {
        "base_model": base_model,
        "data_path": str(data.path),
        "labels": data.labels,
        "label2id": data.label2id,
        "id2label": data.id2label,
        "max_seq_length": max_seq_length,
        "train_rows": len(data.train_df),
        "eval_rows": len(data.eval_df),
        "train_label_counts": label_counts(data.train_df),
        "eval_label_counts": label_counts(data.eval_df),
        "train_policy_count": data.train_df["guardrail_id"].nunique(),
        "eval_policy_count": data.eval_df["guardrail_id"].nunique(),
        "eval_guardrail_ids": sorted(data.eval_df["guardrail_id"].unique()),
        "base_learning_rate": base_lr,
        "classifier_learning_rate": classifier_lr,
        "warmup_steps": training_args.warmup_steps,
    }


class TalMonitorTrainer(Trainer):
    """Builds the per-group optimizer in ``create_optimizer`` instead of via
    ``optimizers=``, which FSDP forbids (the optimizer must be created post-wrap)."""

    def __init__(self, *args: Any, training_config: dict[str, Any] | None = None, **kwargs: Any) -> None:
        self._training_config = training_config or {}
        super().__init__(*args, **kwargs)

    def create_optimizer(self, model: Any = None) -> Any:
        if self.optimizer is None:
            self.optimizer = build_optimizer(model or self.model, self._training_config)
        return self.optimizer


def train(
    config: dict[str, Any],
    model_name: str | None = None,
    data_path: str | None = None,
    output_dir: str | None = None,
    max_seq_length: int | None = None,
    save_model: bool = True,
) -> Trainer:
    data = prepare_data(config, data_path)
    train_config = resolve_training_config(config, output_dir)
    model_config = config.get("model", {})
    name = resolve_model_name(config, model_name)
    max_length = max_seq_length or int(model_config.get("max_seq_length", 4096))

    tokenizer, model = load_tokenizer_and_model(
        name,
        model_config,
        train_config,
        data.labels,
        data.label2id,
        data.id2label,
    )
    prompter = build_prompter()
    train_dataset = make_dataset(data.train_df, data.labels, data.label2id, tokenizer, max_length, prompter)
    eval_dataset = None if data.eval_df.empty else make_dataset(data.eval_df, data.labels, data.label2id, tokenizer, max_length, prompter)
    train_args = training_arguments(train_config, eval_dataset is not None)
    trainer = TalMonitorTrainer(
        model=model,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics if eval_dataset is not None else None,
        processing_class=tokenizer,
        training_config=train_config,
    )

    log_data_summary(data)
    trainer.train(ignore_keys_for_eval=["past_key_values"])

    if save_model:
        output_dir = Path(train_args.output_dir)
        logger.info("Saving final classifier to %s", output_dir)
        trainer.save_model(str(output_dir))
        tokenizer.save_pretrained(str(output_dir))
        with open(output_dir / "training_metadata.json", "w") as f:
            json.dump(metadata(data, name, max_length, train_args, train_config), f, indent=2)
        logger.info("Saved final classifier to %s", output_dir)

        # Optionally convert the just-saved checkpoint to a real fp8 (W8A8) deployable
        # one. Done on rank 0 by reloading from disk (robust under FSDP/torchrun).
        if bool_config(train_config.get("export_fp8", False)) and trainer.is_world_process_zero():
            from training.fp8_runtime import export_to_real_fp8

            dst = Path(f"{output_dir}-fp8-real")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("Auto-exporting real fp8 checkpoint to %s", dst)
            export_to_real_fp8(output_dir, dst)

    return trainer


def dry_run(
    config: dict[str, Any],
    model_name: str | None = None,
    data_path: str | None = None,
    output_dir: str | None = None,
) -> None:
    data = prepare_data(config, data_path)
    train_config = resolve_training_config(config, output_dir)
    train_args = training_arguments(train_config, not data.eval_df.empty)
    base_lr, classifier_lr = learning_rates(train_config)
    preview = render_texts(data.all_df.head(1), data.labels, build_prompter())[0]

    logger.info("Model: %s", model_name or config.get("model", {}).get("name"))
    log_data_summary(data, include_all_rows=True)
    logger.info("Resolved base_learning_rate: %s", base_lr)
    logger.info("Resolved classifier_learning_rate: %s", classifier_lr)
    logger.info("Resolved warmup_steps: %s", train_args.warmup_steps)
    logger.info("First formatted input preview:\n%s", preview[:1200])
    logger.info("True label for preview: %s", data.all_df.iloc[0]["label"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to training config YAML")
    parser.add_argument("--model-name", default=None, help="Override model.name, e.g. google/gemma-4-...")
    parser.add_argument("--data-path", default=None, help="Override data.path")
    parser.add_argument("--output-dir", default=None, help="Override training.output_dir")
    parser.add_argument("--max-seq-length", type=int, default=None, help="Override model.max_seq_length")
    parser.add_argument("--dry-run", action="store_true", help="Validate config/data and print a prompt preview")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    options = {
        "model_name": args.model_name,
        "data_path": args.data_path,
        "output_dir": args.output_dir,
    }
    if args.dry_run:
        dry_run(config, **options)
    else:
        train(config, max_seq_length=args.max_seq_length, **options)


if __name__ == "__main__":
    main()
