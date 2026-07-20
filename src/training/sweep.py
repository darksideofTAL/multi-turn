"""W&B grid-sweep entry point for the four training hyperparameters.

One agent run = one grid cell. It overrides only the swept knobs in the base
training config, trains the model in memory (no checkpoint written to disk),
then logs the metrics to W&B.

Launched by the W&B agent (see config/train_sft/sweep.yaml / scripts/run_sweep.sh).
"""

import logging
import os
import sys
from pathlib import Path

import torch
import wandb
import yaml

# Make the `training` package importable regardless of how the agent invokes us.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from training.eval import (  # noqa: E402
    DEFAULT_EVAL_CONFIG,
    evaluate_model,
    flatten_test_set,
    load_eval_config,
)
from training.runner import train  # noqa: E402

logger = logging.getLogger(__name__)

SWEPT_PARAMS = (
    "base_learning_rate",
    "classifier_learning_rate",
    "max_steps",
    "per_device_train_batch_size",
)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    wandb.init()

    config_path = os.environ.get("TALMONITOR_TRAIN_CONFIG", "config/train_sft/train.yaml")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    training_config = config["training"]
    for key in SWEPT_PARAMS:
        if key in wandb.config:
            training_config[key] = wandb.config[key]
    training_config["report_to"] = "wandb"
    training_config["save_strategy"] = "no"
    training_config["output_dir"] = f"output/training/sweeps/{wandb.run.id}"
    logger.info(
        "Sweep run %s overrides: %s",
        wandb.run.id,
        {key: training_config[key] for key in SWEPT_PARAMS},
    )

    trainer = train(config, save_model=False)

    # Release optimizer/grad state so the 4B model fits on one GPU for the eval pass.
    trainer.optimizer = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    labels, test_set = load_eval_config(DEFAULT_EVAL_CONFIG)
    samples = flatten_test_set(test_set, labels)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    max_seq_length = int(config.get("model", {}).get("max_seq_length", 4096))
    result = evaluate_model(
        trainer.model,
        trainer.processing_class,
        samples,
        labels,
        max_seq_length,
        device,
        name=wandb.run.id,
    )

    # `test/*` namespace keeps this held-out result distinct from HF's
    # per-epoch validation metrics, which WandbCallback logs as `eval/*`.
    summary = {
        "test/accuracy": result["accuracy"],
        "test/recall": result["recall"],
        "test/precision": result["precision"],
        "test/tp": result["tp"],
        "test/fp": result["fp"],
        "test/tn": result["tn"],
        "test/fn": result["fn"],
    }
    wandb.log(summary)
    wandb.run.summary.update(summary)
    logger.info("Sweep run %s test-set metrics: %s", wandb.run.id, summary)
    wandb.finish()


if __name__ == "__main__":
    main()
