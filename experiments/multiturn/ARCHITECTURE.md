# Architecture

Hierarchical monitor: the frozen single-turn classifier encodes each turn to
one policy-conditioned latent; a small causal transformer attends over the
sequence of latents and emits a conversation-so-far verdict at every turn. Same
philosophy as the single-turn product — frozen backbone, tiny head, policy
supplied only at inference.

    policy ── render(policy, ∅) ──► 12B + head ──► p            (policy latent, once)
    turn_t ── render(policy, turn_t) ──► 12B + head ──► e_t, z_t  (per turn)

    aggregator:  [p, e_1..e_T] ─ proj(H+L→d) + role emb + RoPE(turn idx)
                 ─ N causal layers ─ head at every position ─ v_t
                 v_t = P(conversation up to turn t violates policy)

Stage 1 is the EXISTING classifier, untouched: the turn latent is the pooled
last-non-pad hidden state (the vector the linear head reads), and z_t is that
head's logits. Policy conditioning happens here — every e_t already means "this
turn, under this policy" — so the aggregator never reads policy text and a new
policy at inference needs zero retraining. z_t is concatenated as a feature, so
max-over-turns is trivially recoverable: the aggregator can only add signal.

Causal mask makes it a streaming monitor: state per conversation is just the
latents (~8 KB/turn), each feed costs one single-turn encode + a sub-ms
aggregator pass, and nothing persists inside the 12B between turns (no KV-cache
surgery). O(1) per turn, vs O(T) to re-encode the transcript each turn.

Aggregator is fully fp32 (~15M params) — the bf16 batch-invariance problem from
the deploy work never enters.

## Training

Backbone frozen; only proj + role embedding + transformer + head train.
Latents are precomputed once (backbone frozen), so each run is minutes on one
GPU over a tensor dataset and cannot regress the shipped single-turn classifier.

Labels are per-turn and MONOTONE: y_t = 1 for every t ≥ first-violation turn
(a violation is never forgotten). Loss = per-turn CE, first-violation turn
upweighted ×2 (trains detection lag, not just final verdicts). Plain CE only —
the token-supervision hinge was seed-sensitive.

tau is stamped into the checkpoint from val benign conversations at a target
PER-CONVERSATION FPR (not a per-turn rate): a 2%/turn monitor is ~18%/conv by
turn 10, so the threshold is set on per-conversation max scores.

## Data (three buckets, frozen classifier as teacher)

    compose        recombine seed rows: violating seed sandwiched in benign
                   turns → pass-through + persistence. No generation.
    decompose      generator splits a violation into N turns each benign alone;
                   KEEP iff every per-turn score < turn_max AND concat ≥
                   concat_min. That filter == exactly the compositional gap.
    hard_negative  benign decoys (decomposition-shaped, union compliant) +
                   policy-swap (relabel a positive benign under another policy).

Splits are by guardrail id, so eval policies are unseen (OOD, as in the
token-supervision experiment).

## Evaluation

Two baselines measured before trusting the aggregator:
  max-over-turns   running max of the single-turn score. The bar to beat.
  full-concat      whole transcript through the 12B, per prefix. Quality
                   ceiling + O(T²) cost strawman; measures the gap's size.

All at a tau set for a target per-conversation benign FPR on val. Report:
conversation F1/AUROC on held-out policies; detection turn-lag; and benign FPR
vs conversation length (the multi-turn failure mode).

## LLM latent aggregator (v3, `mtlib/llm_aggregator.py`)

Same latents, but the aggregator is a FROZEN pretrained LM that reads the policy
AS TEXT plus one projected soft token per turn (LLaVA-style bridge; trainable =
projector + role emb + readout, ~13M):

    [instruction + policy text] [soft e_1] ... [soft e_T] -> frozen LM
    readout at each soft position -> conversation-so-far verdict

Readouts: "linear" (new head on hidden states) or "verbalizer" (the LM's own
" No"/" Yes" logits — prompt-tuning style; best in-domain calibration: 50% comp
detection with zero FPR drift). `lm_name: talmonitor-backbone` uses the 12B
classifier's own backbone as the aggregator LM — decisive at LOW data (only
method with OOD comp detection on v2-scale data, 23.5%), matched by the 15M
transformer once v4-scale data exists. ~230-token forward per turn: 4B ≈ 3-5 ms,
12B ≈ 10-15 ms; the 12B turn-encode still dominates.

## Data scaling: latent bank + distinct cores (see RESULTS.md)

Turn latents depend only on the (policy, turn_text) pair, and composed
conversations reuse turns. So `mtlib/latent_bank.py` encodes each UNIQUE
(policy, turn) pair once with the 12B, and `scripts/compose_from_bank.py`
recombines cached latents with NO further 12B forwards, reusing the monotone
label logic of `build_compose_bucket`. Deduped (exact turn-sequence hash +
LSH/MinHash near-dup, compositional cores protected). Eval splits stay REAL.

The scaling finding (RESULTS.md): what helps is more DISTINCT compositional
CORES, not more conversations or more parameters. Recombination must be LIGHT
(~10 contexts/core) and COMPOSITIONAL-ONLY (`--comp-only`) — the single-turn
"compose" flood is anti-compositional, and heavy recombination memorizes cores
and hurts OOD. Cores come from Qwen3.6-27B (thinking, ≥16k tokens — the thinking
budget is load-bearing: 37% decompose yield vs 2-3% for dolphin/non-thinking)
over `seeds/threat_families.jsonl` (agentic misalignment, tool abuse, cumulative
disclosure, crescendo) + broad TALMONITOR. Measure on a guardrail-disjoint test
with 100+ compositional positives (`scripts/build_core_eval.py`); the old
27-positive test could not distinguish configs. `train_aggregator.py` has bf16
autocast + grad-accum for large configs (the param ladder was a null result).

## Known risks

- Bottleneck: one vector/turn may drop the entity that links turns. Fix if
  eval shows linkage failure: k learned query latents per turn (perceiver
  pool) — the genuine MLA analogue — backbone still frozen.
- Stage-1 blindness: a turn is encoded without conversation context, so a
  reference-dependent turn ("yes, do that") is embedded context-free; the
  aggregator must resolve it from latents. Middle-ground ablation: prepend turn
  t−1 in the sample block.

## Files

    mtlib/encoder.py        frozen classifier → per-turn latent + logits
    mtlib/aggregator.py     causal transformer + per-turn loss
    mtlib/llm_aggregator.py frozen-LM soft-token aggregator (v3)
    mtlib/monitor.py        streaming + one-shot inference
    mtlib/latent_bank.py    encode unique (policy,turn) pairs once, reuse
    mtlib/dataset.py        latent shard IO + collate + bank gather
    mtlib/metrics.py        AUROC/F1/detection/FPR-by-length (numpy)
    mtlib/datagen*.py       datagen buckets + generator prompts + recombination + dedup
    mtlib/schema.py         Conversation/Turn, transcript parsing
    scripts/                gen_dataset → build_latent_bank → compose_from_bank →
                            train_aggregator → eval_monitor; build_core_eval (honest test)

Results and findings: `RESULTS.md`.
