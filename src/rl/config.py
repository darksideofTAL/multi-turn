"""Configuration dataclasses for the RL pipeline.

A single ``RLConfig`` is loaded from ``config/train_rl/rl.yaml`` and is the
only object the orchestrator (``iterate.py``) needs to drive one iteration.
Per-iteration paths are derived from ``RLConfig.output_root`` via
:func:`iteration_paths`.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LABELS = ("False", "True")


@dataclass
class GeneratorConfig:
    base_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    n_per_seed: int = 8
    max_seeds_per_guardrail: int = 4  # seed rows sampled per non-held-out guardrail; 0 = all
    temperature: float = 0.9
    top_p: float = 1.0
    max_tokens: int = 2048
    batch_size: int = 8


@dataclass
class VerifierConfig:
    judge_model: str = "meta-llama/Llama-3.1-8B-Instruct"
    persona: str = "strict precision-oriented"
    persona_instructions: str = (
        "Use no flexibility or interpretive leeway. Choose the positive label "
        "only when the criterion clearly holds."
    )
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 1024
    batch_size: int = 8


@dataclass
class DPOConfig:
    beta: float = 0.01
    learning_rate: float = 5.0e-7
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    max_length: int = 2048


@dataclass
class ClassifierConfig:
    talmonitor_checkpoint: str = ""  # required
    train_config_template: str = "config/train_sft/train.yaml"
    freeze_backbone: bool = False  # SFT template freezes; RL unfreezes to learn from adversarial examples
    max_seq_length: int = 4096
    per_device_train_batch_size: int = 8
    gradient_accumulation_steps: int = 4
    base_learning_rate: float = 5.0e-6
    classifier_learning_rate: float = 1.0e-5
    num_train_epochs: int = 2
    warmup_ratio: float = 0.1
    validation_size: float | int = 0


@dataclass
class RLConfig:
    output_root: str = "output/rl"
    seed_training_path: str = "output/rl/seed/training.jsonl"
    policies_dir: str = "config/generation/policies"
    # Optional: seed the generator from a generated-examples JSONL
    # (accepted_samples.jsonl: rows of guardrail_id/policy_prompt/input_block/label)
    # instead of the policy YAMLs. Empty => seed from policies_dir YAMLs.
    seed_samples_path: str = ""
    held_out_ids: list[str] = field(
        default_factory=lambda: [
            "GR-TALMONITOR-HEALTHE-001",
            "GR-TALMONITOR-PLAN-VERIFICATION-001",
            "GR-TALMONITOR-GPS-DISCLOSURE-001",
            "GR-TALMONITOR-MESSAGE-REPETITION-001",
            "GR-TALMONITOR-AGENT-INTEGRITY-001",
        ]
    )
    labels: list[str] = field(default_factory=lambda: list(DEFAULT_LABELS))
    num_iterations: int = 3
    seed: int = 42
    # Both mandatory (validated in load_rl_config); list explicit GPU indices, e.g. [0,1].
    generation_gpus: list[int] = field(default_factory=list)  # generate/verify shard data-parallel, one worker/GPU
    training_gpus: list[int] = field(default_factory=list)  # classifier+DPO via torchrun DDP (not DataParallel)

    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    verifier: VerifierConfig = field(default_factory=VerifierConfig)
    dpo: DPOConfig = field(default_factory=DPOConfig)

    def resolved_output_root(self) -> Path:
        return _resolve(self.output_root)

    def resolved_policies_dir(self) -> Path:
        return _resolve(self.policies_dir)

    def gen_gpus(self) -> list[int]:
        return list(self.generation_gpus)

    def train_gpus(self) -> list[int]:
        return list(self.training_gpus)


@dataclass(frozen=True)
class IterationPaths:
    iter_dir: Path
    candidates: Path
    verified: Path
    classified: Path
    preference: Path
    mis_increment: Path
    training: Path
    classifier_dir: Path
    generator_dir: Path
    classifier_cfg: Path


def iteration_paths(cfg: RLConfig, t: int) -> IterationPaths:
    root = cfg.resolved_output_root() / f"iter{t}"
    return IterationPaths(
        iter_dir=root,
        candidates=root / "candidates.jsonl",
        verified=root / "verified.jsonl",
        classified=root / "classified.jsonl",
        preference=root / "preference.jsonl",
        mis_increment=root / "mis_increment.jsonl",
        training=root / "training.jsonl",
        classifier_dir=root / "classifier",
        generator_dir=root / "generator",
        classifier_cfg=root / "classifier_cfg.yaml",
    )


def generator_for_iter(cfg: RLConfig, t: int) -> str:
    """Generator to sample from at iteration ``t``: base model at iter0, else the
    previous iteration's DPO checkpoint (falling back to base if it is missing)."""
    if t == 0:
        return cfg.generator.base_model
    prev = iteration_paths(cfg, t - 1).generator_dir
    return str(prev) if prev.exists() else cfg.generator.base_model


def classifier_for_iter(cfg: RLConfig, t: int) -> str:
    """Classifier to score with at iteration ``t``: the TALMONITOR checkpoint at iter0,
    else the previous iteration's trained classifier."""
    if t == 0:
        return cfg.classifier.talmonitor_checkpoint
    prev = iteration_paths(cfg, t - 1).classifier_dir
    if not prev.exists():
        raise FileNotFoundError(
            f"Previous classifier missing at {prev} (expected for iter{t})"
        )
    return str(prev)


def load_rl_config(path: str | Path, require_pipeline_inputs: bool = True) -> RLConfig:
    """Load ``rl.yaml``. With ``require_pipeline_inputs`` (the training loop), a
    classifier checkpoint and seed-samples file are mandatory; eval passes False."""
    cfg_path = _resolve(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"RL config not found: {cfg_path}")
    with open(cfg_path) as f:
        data = yaml.safe_load(f) or {}

    cfg = RLConfig(
        classifier=ClassifierConfig(**(data.pop("classifier", {}) or {})),
        generator=GeneratorConfig(**(data.pop("generator", {}) or {})),
        verifier=VerifierConfig(**(data.pop("verifier", {}) or {})),
        dpo=DPOConfig(**(data.pop("dpo", {}) or {})),
        **data,
    )
    if len(cfg.labels) != 2:
        raise ValueError("rl.labels must be a list of exactly two strings")

    for name in ("generation_gpus", "training_gpus"):
        if not getattr(cfg, name):
            raise ValueError(f"{name} is required in rl.yaml: list explicit GPU indices, e.g. [0, 1].")

    # Checkpoint may come from $TALMONITOR_CHECKPOINT (flows to subprocesses); YAML wins.
    if not cfg.classifier.talmonitor_checkpoint:
        cfg.classifier.talmonitor_checkpoint = os.environ.get("TALMONITOR_CHECKPOINT", "")

    if require_pipeline_inputs:
        if not cfg.classifier.talmonitor_checkpoint:
            raise ValueError(
                "classifier.talmonitor_checkpoint is required: set it in rl.yaml or export "
                "TALMONITOR_CHECKPOINT=<path-or-hf-id-of-a-TALMONITOR-pretrained-classifier>."
            )
        if not cfg.seed_samples_path:
            raise ValueError(
                "seed_samples_path is required: point it at an accepted_samples.jsonl "
                "(rows of guardrail_id/policy_prompt/input_block/label) to seed generation."
            )
    return cfg


def label2id(labels: list[str]) -> dict[str, int]:
    return {label: i for i, label in enumerate(labels)}


def _resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def dump_classifier_yaml(
    cfg: RLConfig, data_path: Path, output_dir: Path, target: Path
) -> dict[str, Any]:
    """Write a classifier training YAML for ``training.runner``: the SFT template
    with model/data/training fields overridden from ``RLConfig``."""
    template_path = _resolve(cfg.classifier.train_config_template)
    with open(template_path) as f:
        template = yaml.safe_load(f) or {}

    template.setdefault("model", {})["name"] = cfg.classifier.talmonitor_checkpoint
    template["model"]["max_seq_length"] = cfg.classifier.max_seq_length

    template.setdefault("data", {})
    template["data"]["path"] = str(data_path)
    template["data"]["labels"] = list(cfg.labels)
    template["data"]["validation_size"] = cfg.classifier.validation_size
    template["data"]["seed"] = cfg.seed

    training = template.setdefault("training", {})
    training["output_dir"] = str(output_dir)
    training["freeze_backbone"] = cfg.classifier.freeze_backbone  # override template's freeze=true
    training["per_device_train_batch_size"] = cfg.classifier.per_device_train_batch_size
    training["gradient_accumulation_steps"] = cfg.classifier.gradient_accumulation_steps
    training["base_learning_rate"] = cfg.classifier.base_learning_rate
    training["classifier_learning_rate"] = cfg.classifier.classifier_learning_rate
    training["num_train_epochs"] = cfg.classifier.num_train_epochs
    training["warmup_ratio"] = cfg.classifier.warmup_ratio
    # Keep the pretrained TALMONITOR head; reinitializing it collapses accuracy.
    training["reinit_classifier_head"] = False
    # Drop step-based settings so the epoch/ratio values above win.
    training.pop("max_steps", None)
    training.pop("warmup_steps", None)

    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w") as f:
        yaml.safe_dump(template, f, sort_keys=False)
    return template
