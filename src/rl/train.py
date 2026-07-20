"""Model training for one RL iteration, behind one dispatcher:

    python -m src.rl.train {classifier|generator} --iter N

- classifier  re-finetune the classifier from the TALMONITOR checkpoint (not iter t-1)
              on the accumulated training.jsonl, via ``training.runner.train``.
- generator   DPO-train the generator from the base model G_0 on preference.jsonl (TRL).
"""

from __future__ import annotations

import argparse
import logging

from training.runner import train as train_runner

from src.rl.config import (
    RLConfig,
    dump_classifier_yaml,
    iteration_paths,
    load_rl_config,
)

logger = logging.getLogger(__name__)


def train_classifier(cfg: RLConfig, t: int) -> None:
    """Train the classifier for iter ``t`` from the TALMONITOR checkpoint on the
    accumulated training.jsonl -> ``output/rl/iter{t}/classifier/``."""
    paths = iteration_paths(cfg, t)
    if not paths.training.exists():
        raise FileNotFoundError(
            f"Iteration {t} training data not found at {paths.training}. "
            "Did you run the orchestrator's accumulate step first?"
        )

    yaml_config = dump_classifier_yaml(
        cfg,
        data_path=paths.training,
        output_dir=paths.classifier_dir,
        target=paths.classifier_cfg,
    )
    paths.classifier_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        "iter%d: training classifier from %s on %s -> %s",
        t,
        cfg.classifier.talmonitor_checkpoint,
        paths.training,
        paths.classifier_dir,
    )
    train_runner(
        config=yaml_config,
        model_name=cfg.classifier.talmonitor_checkpoint,
        data_path=str(paths.training),
        output_dir=str(paths.classifier_dir),
        max_seq_length=cfg.classifier.max_seq_length,
        save_model=True,
    )
    logger.info("iter%d: classifier checkpoint saved to %s", t, paths.classifier_dir)


def train_generator(cfg: RLConfig, t: int) -> None:
    """Run DPO for iteration ``t`` from the base generator."""
    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import DPOConfig, DPOTrainer

    base_model = cfg.generator.base_model
    paths = iteration_paths(cfg, t)
    if not paths.preference.exists():
        raise FileNotFoundError(
            f"preference.jsonl missing for iter{t}: {paths.preference}"
        )

    logger.info(
        "iter%d: DPO from %s, prefs=%s -> %s",
        t,
        base_model,
        paths.preference,
        paths.generator_dir,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = load_dataset(
        "json", data_files=str(paths.preference), split="train"
    )
    required = {"prompt", "chosen", "rejected"}
    missing = required - set(train_dataset.column_names)
    if missing:
        raise ValueError(f"preference.jsonl missing required columns: {missing}")

    args = DPOConfig(
        output_dir=str(paths.generator_dir),
        beta=cfg.dpo.beta,
        learning_rate=cfg.dpo.learning_rate,
        num_train_epochs=cfg.dpo.num_train_epochs,
        per_device_train_batch_size=cfg.dpo.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.dpo.gradient_accumulation_steps,
        max_length=cfg.dpo.max_length,  # TRL >=1.4 has no separate max_prompt_length
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        gradient_checkpointing=True,
        bf16=True,
        optim="adamw_torch",
        save_strategy="no",
        logging_steps=10,
        report_to="none",
        seed=cfg.seed + t,
        remove_unused_columns=False,
    )

    model = AutoModelForCausalLM.from_pretrained(base_model, dtype="bfloat16")
    trainer = DPOTrainer(
        model=model,
        ref_model=None,  # TRL uses the frozen initial weights as the reference when None
        args=args,
        train_dataset=train_dataset,
        processing_class=tokenizer,  # TRL >=0.12 renamed `tokenizer` -> `processing_class`
    )
    trainer.train()
    paths.generator_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(paths.generator_dir))
    tokenizer.save_pretrained(str(paths.generator_dir))
    logger.info("iter%d: generator saved to %s", t, paths.generator_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "target", choices=("classifier", "generator"), help="Which model to train."
    )
    parser.add_argument("--rl-config", default="config/train_rl/rl.yaml")
    parser.add_argument("--iter", type=int, required=True, dest="iteration")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    args = parse_args()
    cfg = load_rl_config(args.rl_config)
    if args.target == "classifier":
        train_classifier(cfg, args.iteration)
    else:
        train_generator(cfg, args.iteration)


if __name__ == "__main__":
    main()
