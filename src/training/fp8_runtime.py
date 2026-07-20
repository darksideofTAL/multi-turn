"""Real fp8 (E4M3) W8A8 inference — the *convert* half of QAT.

Turns the QAT-trained bf16 weights into a genuinely quantized model: weights are
stored as ``torch.float8_e4m3fn`` with a static per-output-channel scale, activations
are dynamically quantized per token, and the matmul runs on fp8 tensor cores via
``torch._scaled_mm``. This is what actually buys the ~2x weight-memory reduction and
the fp8 GEMM speedup (unlike the training-time fake-quant in ``qat.py``).

Scales follow the same convention as the QAT fake-quant (amax / fp8_max), so a
QAT-trained checkpoint converts with near-zero accuracy loss.
"""

import logging

import torch
from torch import nn

from training.qat import FakeQuantLinear

logger = logging.getLogger(__name__)

E4M3 = torch.float8_e4m3fn
E4M3_MAX = float(torch.finfo(E4M3).max)  # 448.0


def _quantize_per_row(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-row (last-dim) symmetric fp8 quant. Returns (fp8 tensor, fp32 dequant scale)."""
    amax = x.abs().amax(dim=-1, keepdim=True).float().clamp(min=1e-12)
    scale = amax / E4M3_MAX
    xq = (x / scale.to(x.dtype)).clamp(-E4M3_MAX, E4M3_MAX).to(E4M3)
    return xq, scale


class RealFp8Linear(nn.Module):
    """fp8 weight + per-channel scale; dynamic per-token activation fp8; `_scaled_mm`.

    Weights/scales are buffers (inference only, no autograd). State-dict keys are
    ``weight`` (fp8), ``weight_scale`` (fp32 [1,out]), and optional ``bias`` (bf16).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.register_buffer("weight", torch.empty(out_features, in_features, dtype=E4M3))
        self.register_buffer("weight_scale", torch.empty(1, out_features, dtype=torch.float32))
        self.register_buffer(
            "bias", torch.empty(out_features, dtype=torch.bfloat16) if bias else None
        )

    @classmethod
    def from_float(cls, weight: torch.Tensor, bias: torch.Tensor | None) -> "RealFp8Linear":
        out_features, in_features = weight.shape
        w = weight.detach().float()
        amax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)  # [out,1]
        scale = amax / E4M3_MAX
        w8 = (w / scale).clamp(-E4M3_MAX, E4M3_MAX).to(E4M3)
        module = cls(in_features, out_features, bias is not None)
        module.to(weight.device)  # buffers default to CPU; keep them with the source weight
        module.weight.copy_(w8)
        module.weight_scale.copy_(scale.reshape(1, out_features).float())
        if bias is not None:
            module.bias.copy_(bias.detach().to(torch.bfloat16))
        return module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2 = x.reshape(-1, x.shape[-1])
        if x2.dtype != torch.bfloat16:
            x2 = x2.to(torch.bfloat16)
        x8, a_scale = _quantize_per_row(x2)  # x8: [M,K] fp8, a_scale: [M,1] fp32
        out = torch._scaled_mm(
            x8,
            self.weight.t(),                 # [K,out] column-major view
            scale_a=a_scale,
            scale_b=self.weight_scale,       # [1,out] fp32
            bias=self.bias,
            out_dtype=torch.bfloat16,
            use_fast_accum=True,
        )
        return out.reshape(*x.shape[:-1], self.out_features)


def _convertible(linear: nn.Module) -> bool:
    # _scaled_mm needs the contraction dim (and output dim) aligned to 16.
    return (
        isinstance(linear, (nn.Linear, FakeQuantLinear))
        and linear.in_features % 16 == 0
        and linear.out_features % 16 == 0
    )


def convert_to_fp8(module: nn.Module) -> tuple[int, int]:
    """Replace every (convertible) Linear/FakeQuantLinear under ``module`` with a
    ``RealFp8Linear``. Returns (converted, skipped). Apply to the backbone only."""
    converted, skipped = _convert(module)
    logger.info("Real fp8: converted %d Linear layers (%d skipped, kept bf16)", converted, skipped)
    return converted, skipped


def _convert(parent: nn.Module) -> tuple[int, int]:
    converted = skipped = 0
    for name, child in list(parent.named_children()):
        if isinstance(child, (nn.Linear, FakeQuantLinear)):
            if _convertible(child):
                setattr(parent, name, RealFp8Linear.from_float(child.weight, child.bias))
                converted += 1
            else:
                skipped += 1
        else:
            c, s = _convert(child)
            converted += c
            skipped += s
    return converted, skipped


def build_fp8_structure(module: nn.Module) -> int:
    """Swap convertible Linears for EMPTY ``RealFp8Linear`` placeholders so a saved
    real-fp8 checkpoint's state_dict (fp8 weight + weight_scale) loads via
    ``from_pretrained``. Called from the model ``__init__`` reload path; the same
    16-divisibility guard as ``convert_to_fp8`` keeps the layer set identical."""
    return _build(module)


def _build(parent: nn.Module) -> int:
    n = 0
    for name, child in list(parent.named_children()):
        if isinstance(child, (nn.Linear, FakeQuantLinear)):
            if _convertible(child):
                setattr(
                    parent,
                    name,
                    RealFp8Linear(child.in_features, child.out_features, child.bias is not None),
                )
                n += 1
        else:
            n += _build(child)
    return n


# Quantization scheme recorded in the exported checkpoint's config + metadata.
SCHEME = {
    "scheme": "fp8_e4m3_w8a8_dynamic",
    "weight": "per_channel_static",
    "activation": "per_token_dynamic",
    "kernel": "torch._scaled_mm",
}


def export_to_real_fp8(src_dir, dst_dir) -> tuple[int, int]:
    """Load a QAT/bf16 checkpoint from ``src_dir``, convert its backbone to real
    E4M3 W8A8 (fp8 storage + ``_scaled_mm``), and save a deployable checkpoint to
    ``dst_dir`` (loadable unchanged by ``from_pretrained``/eval.py/the deploy engine).
    Returns (converted, skipped) Linear counts."""
    import json
    from pathlib import Path

    from transformers import AutoTokenizer

    from training.modeling import TalMonitorSequenceClassifier

    src, dst = Path(src_dir), Path(dst_dir)
    tok = AutoTokenizer.from_pretrained(src)
    model = TalMonitorSequenceClassifier.from_pretrained(src, dtype=torch.bfloat16).eval()
    n_conv, n_skip = convert_to_fp8(model.backbone)
    model.config.quantization = SCHEME
    model.config.qat = None  # the exported model is real-fp8, not fake-quant

    dst.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(dst)
    tok.save_pretrained(dst)

    meta_path = src / "training_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta["quantization"] = {
        **SCHEME,
        "converted_linears": n_conv,
        "skipped_linears": n_skip,
        "exported_from": src.name,
    }
    (dst / "training_metadata.json").write_text(json.dumps(meta, indent=2))
    logger.info("Exported real-fp8 to %s (converted=%d, skipped=%d)", dst, n_conv, n_skip)
    return n_conv, n_skip
