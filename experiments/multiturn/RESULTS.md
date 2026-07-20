# Results

Every configuration tested, in order, with the finding it produced. Metric =
compositional detection: does the monitor flag a conversation that violates the
policy only when its turns are read TOGETHER (no single turn trips the frozen
single-turn classifier). "COMP" columns are the compositional slice; detection
is at a 2% per-conversation FPR threshold set on val. Evidence: `outputs/eval_*.json`.

## Headline

The best deployable configuration is the **15M causal aggregator trained on the
full pooled compositional cores (v4+v6+v7) with light clean recombination**.
Measured on the honest 167-positive held-out test (3 seeds):

    COMP AUROC 0.904 ± 0.007   oracle detection 0.517 ± 0.051   (10x cores)
    vs 0.850 ± 0.032 / 0.206 ± 0.052 for the small (v4+v6) core set

Two things had to be true to see this: (1) enough DISTINCT compositional cores,
and (2) a test big enough (100+ positives) to measure it. Neither parameter
scaling nor conversation-count scaling helped.

## What moved the number, and what didn't

| lever | verdict | evidence |
|---|---|---|
| per-conversation FPR calibration | REQUIRED | every per-turn-calibrated baseline → 0% detection |
| aggregator vs baselines | aggregator only | max-over-turns & full-concat collapse at matched FPR |
| more distinct compositional cores | **the lever** | 210→554 cores: COMP AUROC 0.85→0.90 (non-overlapping CIs) |
| model size 15M→500M | no effect | flat across the ladder |
| conversation count via recombination | HURTS | 8.8k→172k: memorization, COMP AUROC 0.90→0.61 |
| frozen-LLM aggregator (vs 15M transformer) | ties in-domain, loses OOD at v4 scale | 4B/9B/12B backbones |
| Qwen3.6-27B thinking generator | 37% decompose yield vs 2-3% | thinking budget ≥16k tokens is load-bearing |

## Full run log (OOD test unless noted)

    run                     COMP_auroc  COMP_det  n_pos  note
    v1  small OOD split        0.572      0.000     30   aggregator > baselines, but 0% det @FPR
    v1b guardrail-strat split  0.580      0.000      5   tiny split
    v1c IN-DOMAIN split        0.966      0.333     12   in-domain works
    v2  6x decompose gen       0.843      0.000     17   more cores, still 0% det
    v4  full-pool gen (244 cores) 0.855   0.259     27   first nonzero OOD det
    ctrl 15M/8.8k clean recipe 0.898      0.148     27   recipe tuned (6ep, oversample 8)
    v5_P15  172k recomb        0.611      0.185     27   data recomb HURTS
    v5_P30/P100/P500           0.59/0.73/0.58  ~0.18  27  param scaling: no trend
    v5c comp-only 60k          0.685      0.111     27   removing single-turn flood helps a bit
    v6  +39 diverse cores      0.783      0.593     27   light recomb + diversity: best det on n=27
    v7  +474 diverse cores(10x) 0.705     0.481     27   looked like a regression...
    --- rebuilt test: 167 held-out compositional positives, 3 seeds ---
    cscale small (210 cores)   0.850±0.032  0.206±0.052 (oracle)  167
    cscale ALL   (554 cores)   0.904±0.007  0.517±0.051 (oracle)  167   <-- winner

    LLM-aggregator (frozen LM + soft tokens), v2/v4 data:
    llm 4B linear / verbalizer  0.74-0.87   0.06-0.22   17-27
    llm 9B / 12B-backbone       0.77-0.88   0.00-0.24   17-27  12B-backbone best at low data
    llm in-domain (v1c)         0.953      0.500      12   ties the 15M transformer

## The methodology catch (why the arc looks noisy)

The v4 test had only 27 compositional positives. Wilson 95% CIs for v4/v6/v7
(26%/59%/48% detection) all overlap — that test cannot distinguish those rates.
So the v5/v6/v7 detection comparisons were within noise; "v6 wins, v7 regresses"
was an artifact. The 10x generation is what made a proper test possible: 864
cores across 94 guardrails let us hold out 15 guardrails for a 167-positive test.
On that test, 10x cores wins with non-overlapping error bars. Rule adopted:
**never compare detection rates on <100 positives; report seed±std.**

## Known limitations / next levers

- **Calibration gap.** The 10x model ranks well (AUROC 0.90) but its val→test
  threshold transfer is noisy (det@val 0.30±0.19; a 2.8k-benign calibration set
  stabilizes it to 0.36±0.08). Deployed detection (0.36) trails the oracle (0.52)
  by a val→test guardrail shift — a calibration problem, not a capability one.
  Next: a matched-distribution benign calibration set / temperature-scaled tau.
- **Core yield on TALMONITOR is ~7%** (vs 37% for the inherently-multi-turn
  threat families) — those policies are hard to distribute. More distinct cores
  is still the lever; the cheap source is more threat-family policies.
- Recombination is only safe LIGHT and COMPOSITIONAL-ONLY; the single-turn
  "compose" flood is anti-compositional and must be excluded (`--comp-only`).
