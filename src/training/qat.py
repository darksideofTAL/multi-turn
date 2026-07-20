"""Fake-quantization for fp8 / bf8 Quantization-Aware Training (QAT).

The backbone keeps full-precision (bf16) master weights; the forward pass
simulates the rounding error of an fp8 cast via a straight-through estimator
(STE), so the model learns weights that survive fp8/bf8 quantization. The exact
same fake-quant is re-applied at eval/inference time (driven by ``config.qat``),
so ``training/eval.py`` measures the *quantized* model rather than the bf16 one,
with no changes to eval.py.

Two formats, named after their OCP fp8 layouts:
  * ``fp8`` / ``fp8_e4m3``  -> ``torch.float8_e4m3fn``  (4 exp, 3 mantissa; max 448)
  * ``bf8`` / ``fp8_e5m2``  -> ``torch.float8_e5m2``    (5 exp, 2 mantissa; max 57344)

Quantization is per-output-channel for weights and per-token (dynamic) for
activations -- the standard fp8 inference recipe (matches vLLM/torchao rowwise
scaling), which keeps a single coarse per-tensor outlier from wrecking accuracy.
"""

import logging
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

logger = logging.getLogger(__name__)

# Alias -> canonical name. Canonical names are what we persist in config.json
# (torch.dtype objects are not JSON-serializable).
_CANONICAL = {
    "fp8": "fp8_e4m3",
    "fp8_e4m3": "fp8_e4m3",
    "e4m3": "fp8_e4m3",
    "float8_e4m3fn": "fp8_e4m3",
    "bf8": "fp8_e5m2",
    "fp8_e5m2": "fp8_e5m2",
    "e5m2": "fp8_e5m2",
    "float8_e5m2": "fp8_e5m2",
}
_TORCH_DTYPE = {
    "fp8_e4m3": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
}
_DISABLED = (None, False, "", "none", "null", "off", "disabled")


def canonical_fp8_name(name: Any) -> str:
    key = str(name).strip().lower()
    if key not in _CANONICAL:
        raise ValueError(
            f"Unsupported fp8 format {name!r}; expected one of "
            f"{sorted(set(_CANONICAL))}"
        )
    return _CANONICAL[key]


def resolve_fp8_dtype(name: Any) -> torch.dtype:
    return _TORCH_DTYPE[canonical_fp8_name(name)]


def normalize_qat_config(cfg: Any) -> dict[str, str] | None:
    """Canonicalize a raw qat config (from YAML or a reloaded config) into a
    JSON-serializable dict of strings, or ``None`` when QAT is disabled.

    Idempotent: re-normalizing an already-normalized dict returns the same dict.
    """
    if cfg in _DISABLED:
        return None
    if not isinstance(cfg, dict):
        raise ValueError(f"qat config must be a mapping or null, got {type(cfg)!r}")
    if cfg.get("enabled") is False:
        return None

    weight_dtype = canonical_fp8_name(cfg.get("weight_dtype", "fp8_e4m3"))
    act_raw = cfg.get("activation_dtype", weight_dtype)
    activation_dtype = None if act_raw in _DISABLED else canonical_fp8_name(act_raw)
    # Quantization is always per-channel weights + per-token activations (the
    # standard fp8 scheme), so granularity is not a config knob.
    return {"weight_dtype": weight_dtype, "activation_dtype": activation_dtype}


@dataclass(frozen=True)
class QATSettings:
    weight_dtype: torch.dtype
    activation_dtype: torch.dtype | None

    @classmethod
    def from_config(cls, cfg: Any) -> "QATSettings":
        if isinstance(cfg, cls):
            return cfg
        norm = normalize_qat_config(cfg)
        if norm is None:
            raise ValueError("Cannot build QATSettings from a disabled qat config")
        return cls(
            weight_dtype=resolve_fp8_dtype(norm["weight_dtype"]),
            activation_dtype=(
                None
                if norm["activation_dtype"] is None
                else resolve_fp8_dtype(norm["activation_dtype"])
            ),
        )


def _scaled_fake_quant(x: torch.Tensor, fp8_dtype: torch.dtype, amax: torch.Tensor) -> torch.Tensor:
    """Scale ``x`` so ``amax`` maps to the fp8 max, round-trip through fp8, unscale.

    Uses a straight-through estimator (identity gradient) so the rounding is
    differentiable. ``amax`` broadcasts against ``x`` (per-channel/per-token).
    """
    fp8_max = torch.finfo(fp8_dtype).max
    scale = fp8_max / amax.clamp(min=1e-12)
    quant = (x * scale).to(fp8_dtype).to(x.dtype) / scale
    # STE: forward uses the quantized value, backward passes the gradient through.
    return x + (quant - x).detach()


def fake_quantize_weight(weight: torch.Tensor, fp8_dtype: torch.dtype) -> torch.Tensor:
    # Per output channel (one scale per row).
    amax = weight.detach().abs().amax(dim=1, keepdim=True)
    return _scaled_fake_quant(weight, fp8_dtype, amax)


def fake_quantize_activation(x: torch.Tensor, fp8_dtype: torch.dtype) -> torch.Tensor:
    # Per token (one scale per last-dim row), dynamic.
    amax = x.detach().abs().amax(dim=-1, keepdim=True)
    return _scaled_fake_quant(x, fp8_dtype, amax)


class FakeQuantLinear(nn.Module):
    """Drop-in replacement for ``nn.Linear`` that fake-quantizes the weight (and,
    optionally, the input activations) to fp8/bf8 with a straight-through
    estimator.

    Reuses the original Linear's ``weight``/``bias`` Parameters under the same
    attribute names, so ``state_dict`` keys are byte-for-byte identical to the
    un-wrapped model and checkpoints load/save without remapping.
    """

    def __init__(
        self,
        weight: nn.Parameter,
        bias: nn.Parameter | None,
        settings: QATSettings,
    ) -> None:
        super().__init__()
        self.register_parameter("weight", weight)
        self.register_parameter("bias", bias)
        self.weight_dtype = settings.weight_dtype
        self.activation_dtype = settings.activation_dtype

    @classmethod
    def from_linear(cls, linear: nn.Linear, settings: QATSettings) -> "FakeQuantLinear":
        return cls(linear.weight, linear.bias, settings)

    @property
    def in_features(self) -> int:
        return self.weight.shape[1]

    @property
    def out_features(self) -> int:
        return self.weight.shape[0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_dtype is not None:
            x = fake_quantize_activation(x, self.activation_dtype)
        weight = fake_quantize_weight(self.weight, self.weight_dtype)
        return F.linear(x, weight, self.bias)


def apply_qat(module: nn.Module, qat_config: Any) -> int:
    """Recursively swap every ``nn.Linear`` under ``module`` for a
    ``FakeQuantLinear``. Idempotent (already-swapped layers are skipped).
    Returns the number of layers replaced.

    Apply to the *backbone* only -- the fp32 classification head stays exact.
    """
    settings = QATSettings.from_config(qat_config)
    replaced = _replace_linears(module, settings)
    logger.info(
        "QAT: fake-quantized %d Linear layers (weight=%s, activation=%s)",
        replaced,
        settings.weight_dtype,
        settings.activation_dtype or "none",
    )
    return replaced


def _replace_linears(parent: nn.Module, settings: QATSettings) -> int:
    replaced = 0
    for name, child in list(parent.named_children()):
        if isinstance(child, FakeQuantLinear):
            continue
        if isinstance(child, nn.Linear):
            setattr(parent, name, FakeQuantLinear.from_linear(child, settings))
            replaced += 1
        else:
            replaced += _replace_linears(child, settings)
    return replaced
