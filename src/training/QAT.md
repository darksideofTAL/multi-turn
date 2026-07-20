# Quantization-Aware Training (fp8)

Two phases: **train** with simulated fp8 rounding so the backbone learns quantization-robust weights (`qat.py`), then **export** to a real fp8 model with genuine speed/memory gains (`fp8_runtime.py`).

## 1. Train (fake-quant)

Add a `qat` block under `training:` in your train YAML, then run the normal trainer:

```yaml
training:
  # ... usual settings ...
  export_fp8: true             # auto-convert to a real fp8 checkpoint after training
  qat:
    weight_dtype: fp8          # fp8 (E4M3, recommended) | bf8 (E5M2)
    activation_dtype: fp8      # fp8 | none (weight-only)
```

```bash
PYTHONPATH=src python -m training.runner --config <train.yaml>
```

The spec is saved in `config.qat`, so the checkpoint re-applies fake-quant on reload — `eval.py` already measures the quantized accuracy. Notes: weights are quantized per-channel, activations per-token (dynamic); the fp32 head stays exact; QAT adds ~1.85× per-step time. **This checkpoint is still bf16 on disk** (fake-quant = accuracy only, no speed/memory win) — that comes from the export.

## 2. Export to a real fp8 checkpoint (automatic)

With `export_fp8: true` (above), training **automatically** writes a real-fp8 checkpoint to
`<output_dir>-fp8-real` — no separate step. It converts the trained weights to E4M3 W8A8
(`RealFp8Linear` + `torch._scaled_mm`), sets `config.quantization`, and is ~2× smaller (e.g.
gemma-3-12b: 23 → 13 GB on disk, 22.8 → 12.4 GiB resident).

To re-export an existing checkpoint without retraining:

```bash
PYTHONPATH=src python scripts/export_fp8.py --src <qat_dir> --dst <fp8_real_dir>
```

## 3. Deploy

The exported checkpoint loads through `from_pretrained` / `eval.py` / `deploy/engine.py` **unchanged** (`__init__` rebuilds the fp8 layers from `config.quantization`):

```bash
PYTHONPATH=src python -m training.eval --checkpoint <fp8_real_dir>      # accuracy
```

**Serve it compiled** (the deploy engine torch.compiles the backbone): eager fp8 is *slower* than bf16 because of per-token quant overhead; compiled it is ~1.1–1.2× faster end-to-end and uses ~46% less memory. See `report.MD` for the measured speed/memory numbers.

## Gotchas

- **E4M3 only for deployment.** Train and deploy in the *same* format. bf8/E5M2 has no inference GEMM, and an E5M2-trained model forced through E4M3 loses accuracy.
- Don't ship the fake-quant (phase-1) checkpoint expecting speed — it's bf16 + simulation.
- See `report.MD` and `journal*.md` for measured results.
