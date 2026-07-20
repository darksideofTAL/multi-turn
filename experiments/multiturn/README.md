# Multi-turn monitor

Detects policy violations that are DISTRIBUTED across a conversation — no single
turn trips the single-turn classifier, the conversation as a whole does. The
frozen 12B classifier encodes each turn to one policy-conditioned latent; an
aggregator attends over the latents and emits a conversation-so-far verdict at
every turn. Frozen backbones throughout, tiny trainable readouts, policy at
inference only. Two aggregators: a 15M causal transformer (`mtlib/aggregator.py`)
and a frozen-LLM soft-token reader (`mtlib/llm_aggregator.py`).

Design in `ARCHITECTURE.md`. **All tested results and the findings that came out
of them are in `RESULTS.md`** (evidence: `outputs/eval_*.json`).

TL;DR: best config is the 15M aggregator trained on the full pooled compositional
cores with light clean recombination — COMP AUROC 0.904 ± 0.007 on a 167-positive
held-out test. More DISTINCT compositional cores is the lever; parameter scaling
and conversation-count recombination do not help. Thresholds must be set on
per-conversation FPR. See `RESULTS.md` for the full table and the methodology.

## Pipeline

```bash
PY=/raid/frontiers_ashoka/BARRED/.venv/bin/python   # repo venv (torch/transformers)
export HF_HOME=/raid/frontiers_ashoka/.cache/huggingface HF_HUB_OFFLINE=1

# 1. generate compositional cores: teacher-filtered decompositions from the 27B
#    (thinking, >=16k tokens -> 37% yield) + natural scan. Encoder on one GPU,
#    generation workers on the rest.
CUDA_VISIBLE_DEVICES=2 $PY scripts/gen_dataset.py --out outputs/data --generator Qwen/Qwen3.6-27B \
    --gpus 0,4,5,7 --scan-natural --n-decomp 4 --max-tokens 12288
#    no-LLM smoke run (compose bucket only): --no-generate

# 2. latent bank: encode each unique (policy, turn) pair ONCE (frozen 12B)
CUDA_VISIBLE_DEVICES=2 $PY scripts/build_latent_bank.py --convs outputs/data/train.jsonl \
    --out /raid/.../bank.pt

# 3. multiply into training data by LIGHT compositional-only recombination (no 12B)
$PY scripts/compose_from_bank.py --convs outputs/data/train.jsonl --bank /raid/.../bank.pt \
    --comp-only --out /raid/.../latents/train --target-convs 12000

# 4. train the 15M aggregator (minutes, one GPU) and evaluate
CUDA_VISIBLE_DEVICES=2 $PY scripts/train_aggregator.py \
    --train-latents /raid/.../latents/train --val-latents /raid/.../latents/val \
    --out outputs/agg --oversample-comp 6
CUDA_VISIBLE_DEVICES=2 $PY scripts/eval_monitor.py --data outputs/data --split test \
    --agg outputs/agg/aggregator.pt --test-latents /raid/.../latents/test \
    --val-latents /raid/.../latents/val --out outputs/eval.json

# honest eval: build a guardrail-disjoint test with 100+ compositional positives
$PY scripts/build_core_eval.py --bank /raid/.../bank.pt --out-data outputs/data_eval \
    --out-lat /raid/.../eval
```

## No-GPU smoke test

The aggregator, dataset, metrics, and streaming/one-shot equivalence run without
the backbone:

```bash
$PY -m pytest -q tests/test_multiturn.py          # 32 tests, no GPU
$PY scripts/make_synthetic_latents.py --out /tmp/mt_synth --n 400 --hidden 64
$PY scripts/train_aggregator.py --train-latents /tmp/mt_synth/train \
    --val-latents /tmp/mt_synth/val --out /tmp/agg_synth --device cpu --epochs 12
```

The synthetic signal is purely cumulative (violation when a hidden coordinate's
running sum crosses a threshold), so a working aggregator beats max-over-turns:
val AUROC ~0.73→0.92, and on test the aggregator's AUROC 0.86 vs max-over-turns
0.74 — the aggregator extracts signal the single-turn max cannot.

## Layout

    mtlib/     encoder · aggregator · llm_aggregator · monitor · latent_bank ·
               dataset · metrics · datagen · datagen_prompts · schema
    scripts/   gen_dataset · build_latent_bank · compose_from_bank · precompute_latents ·
               train_aggregator · train_llm_aggregator · eval_monitor · build_core_eval ·
               live_demo · make_synthetic_latents
    seeds/     threat_families.jsonl (agentic misalignment, tool abuse, cumulative, crescendo)
    tests/     test_multiturn.py (32 tests, no GPU)
    outputs/   result JSONs (eval_*.json) + reference checkpoints; RESULTS.md is the index
