import json
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    TrainingArguments,
)
from transformers.modeling_outputs import SequenceClassifierOutput

from training.qat import apply_qat, normalize_qat_config

logger = logging.getLogger(__name__)

# The head is the submodule named "score"; everything under "backbone." is the base model.
HEAD_PREFIX = "score"


def sharding_enabled(training_config: dict[str, Any]) -> bool:
    """Whether the model should be FSDP-sharded: either explicitly requested via
    ``training.fsdp`` or launched on more than one GPU (torchrun sets ``WORLD_SIZE``).
    """
    return (
        bool(training_config.get("fsdp")) or int(os.environ.get("WORLD_SIZE", "1")) > 1
    )


def _text_hidden_size(backbone_config: PretrainedConfig) -> int:
    """Hidden size of the text stream. Multimodal/composite configs (Gemma3,
    Qwen3.5) expose it under ``text_config`` rather than at the top level, so go
    through ``get_text_config()`` when available."""
    text_config = (
        backbone_config.get_text_config()
        if hasattr(backbone_config, "get_text_config")
        else backbone_config
    )
    hidden = getattr(text_config, "hidden_size", None) or getattr(
        backbone_config, "hidden_size", None
    )
    if hidden is None:
        raise ValueError(
            "Could not determine hidden_size from backbone config "
            f"({type(backbone_config).__name__})"
        )
    return int(hidden)


def _resolve_backbone_config(backbone_config: Any) -> PretrainedConfig | None:
    """Turn a serialized backbone config (dict, on reload) back into a config
    object. In-library model types go through ``AutoConfig.for_model``; anything
    else (e.g. trust_remote_code backbones) keeps its ``auto_map`` via a generic
    ``PretrainedConfig`` so ``AutoModel.from_config(..., trust_remote_code=True)``
    can still resolve it."""
    if backbone_config is None or isinstance(backbone_config, PretrainedConfig):
        return backbone_config
    model_type = backbone_config.get("model_type")
    if model_type in CONFIG_MAPPING:
        return AutoConfig.for_model(**backbone_config)
    return PretrainedConfig.from_dict(backbone_config)


class TalMonitorClassifierConfig(PretrainedConfig):
    """Config for a backbone-agnostic sequence classifier.

    Owns the label maps and pooling choice as first-class fields so they
    round-trip through ``save_pretrained``/``from_pretrained`` for every model
    family (some families drop ``id2label``/``label2id`` from a nested config)."""

    model_type = "talmonitor_classifier"

    def __init__(
        self,
        backbone_config: Any = None,
        pooling: str = "last_token",
        classifier_dropout: float = 0.0,
        backbone_trust_remote_code: bool = False,
        qat: Any = None,
        quantization: Any = None,
        **kwargs: Any,
    ) -> None:
        self.backbone_config = _resolve_backbone_config(backbone_config)
        self.pooling = pooling
        self.classifier_dropout = classifier_dropout
        self.backbone_trust_remote_code = backbone_trust_remote_code
        # Canonical fp8/bf8 QAT spec (or None). Persisted so the saved checkpoint
        # re-applies fake-quant on reload -> eval/inference sees the quantized model.
        self.qat = normalize_qat_config(qat)
        # Real (deployed) quantization spec for an exported checkpoint, e.g.
        # {"scheme": "fp8_e4m3_w8a8_dynamic"}. When set, the backbone is built with
        # RealFp8Linear (fp8 storage + _scaled_mm) instead of fake-quant.
        self.quantization = quantization
        # id2label/label2id/num_labels are handled by the base class from kwargs.
        super().__init__(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        output = super().to_dict()
        if isinstance(self.backbone_config, PretrainedConfig):
            output["backbone_config"] = self.backbone_config.to_dict()
        return output


class TalMonitorSequenceClassifier(PreTrainedModel):
    """A bare ``AutoModel`` backbone + a linear classification head.

    Generalizes across architectures (including those without a registered
    ``*ForSequenceClassification``) by pooling the last non-pad token's hidden
    state and projecting it to ``num_labels``. Honors the ``model(**inputs).logits``
    contract every TALMONITOR call site depends on, plus a CrossEntropy ``loss`` for
    ``Trainer``."""

    config_class = TalMonitorClassifierConfig
    base_model_prefix = "backbone"
    supports_gradient_checkpointing = True

    def __init__(
        self, config: TalMonitorClassifierConfig, backbone: Any = None
    ) -> None:
        super().__init__(config)
        self.num_labels = config.num_labels
        if backbone is None:
            # Reload path: weights are loaded over this by from_pretrained.
            backbone = AutoModel.from_config(
                config.backbone_config,
                trust_remote_code=getattr(config, "backbone_trust_remote_code", False),
            )
        self.backbone = backbone
        if getattr(self.backbone, "config", None) is not None:
            self.backbone.config.use_cache = False
        # Swap backbone Linears BEFORE post_init/weight-loading so state_dict keys
        # line up. A real-fp8 checkpoint (config.quantization) builds RealFp8Linear
        # placeholders; otherwise config.qat builds fake-quant layers for training.
        if getattr(config, "quantization", None):
            from training.fp8_runtime import build_fp8_structure

            build_fp8_structure(self.backbone)
        elif getattr(config, "qat", None):
            apply_qat(self.backbone, config.qat)
        hidden_size = _text_hidden_size(config.backbone_config)
        self.dropout = nn.Dropout(config.classifier_dropout)
        self.score = nn.Linear(hidden_size, config.num_labels, bias=False)
        self._init_head()
        # post_init gathers tied-weights keys from the backbone (no re-init in
        # transformers >=5); from_pretrained needs all_tied_weights_keys present.
        self.post_init()

    def _init_head(self) -> None:
        # Zero-init so initial logits are 0 (loss = ln(num_labels)). A random head on
        # large-magnitude hidden states spikes the first-step grad norm and, under
        # max_grad_norm=1.0 clipping, leaves the backbone effectively frozen.
        self.score.weight.data.zero_()
        if self.score.bias is not None:
            self.score.bias.data.zero_()

    # Delegate embedding + gradient-checkpointing hooks to the backbone so
    # Trainer and tokenizer-resize utilities behave normally.
    def get_input_embeddings(self) -> nn.Module:
        return self.backbone.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.backbone.set_input_embeddings(value)

    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs: Any = None
    ) -> None:
        self.backbone.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
        )

    def gradient_checkpointing_disable(self) -> None:
        self.backbone.gradient_checkpointing_disable()

    @classmethod
    def from_backbone(
        cls,
        model_name: str,
        *,
        id2label: dict[int, str],
        label2id: dict[str, int],
        dtype: Any = None,
        trust_remote_code: bool = False,
        pooling: str = "last_token",
        classifier_dropout: float = 0.0,
        qat: Any = None,
        **model_kwargs: Any,
    ) -> "TalMonitorSequenceClassifier":
        """Build a fresh classifier on top of a pretrained base model."""
        backbone = AutoModel.from_pretrained(
            model_name,
            dtype=dtype,
            trust_remote_code=trust_remote_code,
            **model_kwargs,
        )
        config = TalMonitorClassifierConfig(
            backbone_config=backbone.config,
            id2label={int(k): v for k, v in id2label.items()},
            label2id=dict(label2id),
            pooling=pooling,
            classifier_dropout=classifier_dropout,
            backbone_trust_remote_code=trust_remote_code,
            qat=qat,
        )
        model = cls(config, backbone=backbone)
        if dtype is not None:
            model.to(dtype)
            model.score.to(torch.float32)  # keep the classification head in fp32
        return model

    def _pool(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.config.pooling != "last_token":
            raise ValueError(f"Unsupported pooling: {self.config.pooling!r}")
        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        device = hidden_states.device
        # Index of the RIGHTMOST non-pad token. Gemma pads LEFT, Qwen RIGHT, so a
        # mask.sum()-1 index lands on a pad slot for left-padded batches; argmax over
        # (position * keep) picks the last kept token either way.
        if attention_mask is not None:
            keep = attention_mask.to(device)
        elif input_ids is not None and self.config.pad_token_id is not None:
            keep = input_ids.to(device) != self.config.pad_token_id
        else:
            keep = torch.ones((batch_size, seq_len), device=device)
        positions = torch.arange(seq_len, device=device)
        last_idx = (positions * keep.to(positions.dtype)).argmax(dim=1)
        return hidden_states[torch.arange(batch_size, device=device), last_idx]

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> SequenceClassifierOutput:
        # Absorb (and drop) Trainer/collator extras like num_items_in_batch and
        # past_key_values that the backbone would reject.
        backbone_kwargs: dict[str, Any] = {}
        if token_type_ids is not None:
            backbone_kwargs["token_type_ids"] = token_type_ids
        if position_ids is not None:
            backbone_kwargs["position_ids"] = position_ids
        if inputs_embeds is not None:
            backbone_kwargs["inputs_embeds"] = inputs_embeds

        outputs = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, **backbone_kwargs
        )
        hidden_states = (
            outputs.last_hidden_state
            if hasattr(outputs, "last_hidden_state")
            else outputs[0]
        )
        pooled = self.dropout(self._pool(hidden_states, input_ids, attention_mask))
        # Head + loss in fp32 regardless of backbone dtype (bf16 softmax on large
        # logits is unstable; the upcast is cheap).
        weight = self.score.weight.float()
        bias = None if self.score.bias is None else self.score.bias.float()
        logits = nn.functional.linear(pooled.float(), weight, bias)

        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits, labels.to(logits.device).view(-1).long()
            )
        return SequenceClassifierOutput(loss=loss, logits=logits)


def is_talmonitor_checkpoint(model_name: str) -> bool:
    """True when ``model_name`` is a local directory holding a saved
    TalMonitorSequenceClassifier (so it should be reloaded, not wrapped fresh)."""
    config_path = Path(model_name) / "config.json"
    if not config_path.exists():
        return False
    try:
        with open(config_path) as f:
            return (
                json.load(f).get("model_type") == TalMonitorClassifierConfig.model_type
            )
    except (json.JSONDecodeError, OSError):
        return False


def bool_config(value: Any) -> bool:
    return (
        value
        if isinstance(value, bool)
        else str(value).strip().lower() in ("true", "1", "yes", "y", "on")
    )


def strategy_config(value: Any, default: str) -> str:
    return "no" if value is False else str(default if value is None else value)


def freeze_backbone(model: Any) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False

    for name, parameter in model.named_parameters():
        if is_classifier_parameter(name):
            parameter.requires_grad = True

    trainable = [
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    ]
    if not trainable:
        raise RuntimeError(
            "freeze_backbone=true left no trainable classifier-head parameters"
        )
    logger.info("Training classifier head parameters: %s", trainable)


def is_classifier_parameter(name: str) -> bool:
    return name.startswith(HEAD_PREFIX)


def learning_rates(training_config: dict[str, Any]) -> tuple[float, float]:
    base_lr = float(
        training_config.get(
            "base_learning_rate", training_config.get("learning_rate", 2e-5)
        )
    )
    classifier_lr = float(
        training_config.get(
            "classifier_learning_rate", training_config.get("learning_rate", base_lr)
        )
    )
    return base_lr, classifier_lr


def build_optimizer(
    model: Any, training_config: dict[str, Any]
) -> torch.optim.Optimizer:
    base_lr, classifier_lr = learning_rates(training_config)
    weight_decay = float(training_config.get("weight_decay", 0.01))
    base_decay, base_no_decay, head_decay, head_no_decay = [], [], [], []

    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if is_classifier_parameter(name):
            (head_decay if parameter.ndim > 1 else head_no_decay).append(parameter)
        else:
            (base_decay if parameter.ndim > 1 else base_no_decay).append(parameter)

    optimizer_groups = []
    for params, lr, decay in (
        (base_decay, base_lr, weight_decay),
        (base_no_decay, base_lr, 0.0),
        (head_decay, classifier_lr, weight_decay),
        (head_no_decay, classifier_lr, 0.0),
    ):
        if params:
            optimizer_groups.append({"params": params, "lr": lr, "weight_decay": decay})
    logger.info(
        "Optimizer learning rates: base=%s classifier=%s", base_lr, classifier_lr
    )
    return torch.optim.AdamW(optimizer_groups)


def reinit_unstable_classifier_head(model: Any) -> None:
    head = getattr(model, HEAD_PREFIX, None)
    if head is None:
        return

    # Zero-init, matching _init_head.
    with torch.no_grad():
        for parameter in head.parameters():
            parameter.zero_()
    logger.info("Reinitialized classifier head parameters (zero-init)")


def validate_finite_parameters(model: Any) -> None:
    nonfinite = [
        name
        for name, parameter in model.named_parameters()
        if not torch.isfinite(parameter).all().item()
    ]
    if nonfinite:
        preview = ", ".join(nonfinite[:10])
        suffix = "" if len(nonfinite) <= 10 else f", ... ({len(nonfinite)} total)"
        raise RuntimeError(
            f"Model contains non-finite parameters after initialization: {preview}{suffix}"
        )


def load_tokenizer_and_model(
    model_name: str,
    model_config: dict[str, Any],
    training_config: dict[str, Any],
    labels: list[str],
    label2id: dict[str, int],
    id2label: dict[int, str],
) -> tuple[Any, Any]:
    # fla's TileLang backward backend crashes (misaligned address) on B200; force Triton.
    os.environ.setdefault("FLA_DISABLE_BACKEND_DISPATCH", "1")

    trust_remote_code = bool_config(model_config.get("trust_remote_code", False))
    model_kwargs = dict(model_config.get("model_kwargs") or {})
    # fp8/bf8 Quantization-Aware Training spec (None when disabled). Lives under
    # training.qat in the YAML; baked into the saved config so eval reuses it.
    qat_cfg = normalize_qat_config(training_config.get("qat"))

    logger.info("Loading tokenizer: %s", model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=trust_remote_code,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if "dtype" not in model_kwargs:
        # Sharded runs train in pure bf16, so the model must load in bf16.
        if bool_config(training_config.get("bf16", False)) or sharding_enabled(
            training_config
        ):
            model_kwargs["dtype"] = torch.bfloat16
        elif bool_config(training_config.get("fp16", False)):
            model_kwargs["dtype"] = torch.float16

    # Reload a saved classifier (RL loop continues from its checkpoint) or wrap a
    # fresh head onto a base LM.
    if is_talmonitor_checkpoint(model_name):
        logger.info("Reloading TALMONITOR classifier checkpoint: %s", model_name)
        model = TalMonitorSequenceClassifier.from_pretrained(
            model_name, trust_remote_code=trust_remote_code, **model_kwargs
        )
    else:
        logger.info("Building classifier on backbone: %s", model_name)
        model = TalMonitorSequenceClassifier.from_backbone(
            model_name,
            id2label=id2label,
            label2id=label2id,
            dtype=model_kwargs.pop("dtype", None),
            trust_remote_code=trust_remote_code,
            classifier_dropout=float(model_config.get("classifier_dropout", 0.0)),
            qat=qat_cfg,
            **model_kwargs,
        )

    # FSDP rejects mixed dtypes within a shard unit, so match the fp32 head to the
    # bf16 backbone; forward() upcasts the head's matmul/loss to fp32 regardless.
    if sharding_enabled(training_config):
        backbone_dtype = next(model.backbone.parameters()).dtype
        if model.score.weight.dtype != backbone_dtype:
            model.score.to(backbone_dtype)
            logger.info(
                "FSDP enabled: cast classifier head to backbone dtype %s",
                backbone_dtype,
            )

    model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model.backbone, "config", None) is not None:
        model.backbone.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False
    model.config.keys_to_ignore_at_inference = ["past_key_values"]
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = False
    logger.info("Disabled model cache for classifier training")

    # Reinit suits a fresh base LM but is destructive when continuing from a trained
    # classifier (the RL loop); default True keeps SFT behavior.
    if bool_config(training_config.get("reinit_classifier_head", True)):
        reinit_unstable_classifier_head(model)
    else:
        logger.info(
            "Preserving pretrained classifier head (reinit_classifier_head=false)"
        )
    validate_finite_parameters(model)
    if bool_config(training_config.get("freeze_backbone", False)):
        freeze_backbone(model)

    return tokenizer, model


def compute_metrics(eval_pred: Any) -> dict[str, float]:
    logits, gold_labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    predictions = np.argmax(logits, axis=-1)
    accuracy = float((predictions == gold_labels).mean())
    positive_id = 1
    true_positive = int(
        ((predictions == positive_id) & (gold_labels == positive_id)).sum()
    )
    predicted_positive = int((predictions == positive_id).sum())
    actual_positive = int((gold_labels == positive_id).sum())
    precision = true_positive / predicted_positive if predicted_positive else 0.0
    recall = true_positive / actual_positive if actual_positive else 0.0
    return {"accuracy": accuracy, "precision": precision, "recall": recall}


def training_arguments(
    training_config: dict[str, Any], has_eval: bool
) -> TrainingArguments:
    if not training_config.get("output_dir"):
        raise ValueError("training.output_dir is required")

    kwargs = {
        "output_dir": training_config["output_dir"],
        "per_device_train_batch_size": int(
            training_config.get("per_device_train_batch_size", 1)
        ),
        "per_device_eval_batch_size": int(
            training_config.get("per_device_eval_batch_size", 1)
        ),
        "gradient_accumulation_steps": int(
            training_config.get("gradient_accumulation_steps", 16)
        ),
        "learning_rate": learning_rates(training_config)[0],
        "weight_decay": float(training_config.get("weight_decay", 0.01)),
        "logging_steps": int(training_config.get("logging_steps", 10)),
        "save_strategy": strategy_config(training_config.get("save_strategy"), "no"),
        "eval_strategy": (
            strategy_config(training_config.get("eval_strategy"), "epoch")
            if has_eval
            else "no"
        ),
        "bf16": bool_config(training_config.get("bf16", False)),
        "fp16": bool_config(training_config.get("fp16", False)),
        "report_to": training_config.get("report_to", "none"),
        "logging_nan_inf_filter": bool_config(
            training_config.get("logging_nan_inf_filter", False)
        ),
        "remove_unused_columns": False,
    }

    if "max_steps" in training_config:
        kwargs["max_steps"] = int(training_config["max_steps"])
    elif "num_train_epochs" in training_config:
        kwargs["max_steps"] = -1
        kwargs["num_train_epochs"] = float(training_config["num_train_epochs"])
    else:
        kwargs["max_steps"] = 1000

    # Warmup: explicit step count wins; otherwise honor a ratio if provided.
    if "warmup_steps" in training_config:
        kwargs["warmup_steps"] = int(training_config["warmup_steps"])
    elif "warmup_ratio" in training_config:
        kwargs["warmup_ratio"] = float(training_config["warmup_ratio"])
    else:
        kwargs["warmup_steps"] = 0

    if has_eval:
        kwargs["load_best_model_at_end"] = bool_config(
            training_config.get("load_best_model_at_end", False)
        )
        kwargs["metric_for_best_model"] = training_config.get(
            "metric_for_best_model", "accuracy"
        )

    casters = {
        "save_total_limit": int,
        "dataloader_num_workers": int,
        "gradient_checkpointing": bool_config,
        "optim": str,
        "lr_scheduler_type": str,
    }
    kwargs.update(
        {
            key: cast(training_config[key])
            for key, cast in casters.items()
            if key in training_config
        }
    )

    kwargs.setdefault("gradient_checkpointing", True)
    kwargs.setdefault("gradient_checkpointing_kwargs", {"use_reentrant": False})

    # Shard multi-GPU runs with FSDP instead of replicating per GPU; explicit
    # training.fsdp / fsdp_config overrides these defaults.
    if sharding_enabled(training_config):
        kwargs["fsdp"] = training_config.get("fsdp", "full_shard auto_wrap")
        kwargs["fsdp_config"] = training_config.get(
            "fsdp_config",
            {
                "min_num_params": 100_000_000,  # the tiny fp32 head stays unsharded
                "use_orig_params": True,  # so build_optimizer's name-based grouping survives the wrap
                "backward_prefetch": "backward_pre",
                "limit_all_gathers": True,
            },
        )
        # Pure bf16: leaving mixed precision on makes FSDP keep an fp32 master copy
        # (+fp32 Adam moments), ~2x the memory, which OOMs large models even sharded.
        kwargs["bf16"] = False
        kwargs["fp16"] = False

    return TrainingArguments(**kwargs)
