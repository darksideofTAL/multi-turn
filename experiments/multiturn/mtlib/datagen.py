"""Multi-turn conversation datagen: three buckets, frozen classifier as teacher.

Buckets
-------
1. compose        Sandwich existing single-turn labeled rows (accepted_samples)
                  among sampled benign turns. Teaches verdict pass-through and
                  persistence (a violation is not forgotten by later benign
                  turns). No generation.
2. decompose      Generator splits a violating input_block into N turns, each
                  benign alone, union violating. Teacher-filtered: KEEP only if
                  every per-turn single-turn score < turn_max AND the full
                  concat scores >= concat_min. That filter guarantees the
                  training signal is exactly the compositional gap.
3. hard_negative  (a) benign decoys: decomposition-shaped, union compliant;
                  (b) policy-swap: a compositional positive relabeled benign
                  under a different policy it does not violate.

The frozen classifier (mtlib.encoder.TurnEncoder) is BOTH the turn encoder and
the automatic label oracle, so buckets are self-consistent with what the
monitor sees at train time.

Generation reuses src/rl/hf_generate.generate_distributed and the RL JSON
contract; only the prompts (mtlib.datagen_prompts) differ.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

# mtlib.common runs the sys.path bootstrap, so it must precede the src/ import
# (generation.prompting) below.
from mtlib.common import NEGATIVE_LABEL, POSITIVE_LABEL
from mtlib.datagen_prompts import format_benign_decoy, format_decompose
from mtlib.encoder import TurnEncoder
from mtlib.schema import (
    COMPOSITIONAL_SOURCES,
    Conversation,
    Turn,
    new_conv_id,
    split_transcript,
    transcript,
    turn_block,
)

from generation.prompting import _extract_json  # tolerant JSON extraction  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass
class DatagenConfig:
    # Teacher filter thresholds on P(violation) from the frozen classifier.
    turn_max: float = 0.5  # every individual turn must score below this
    concat_min: float = 0.5  # full concatenation must score at/above this
    # Decoy (benign) acceptance: full concat must score below this.
    decoy_max: float = 0.5
    # Compose bucket.
    compose_pad_turns: tuple[int, int] = (1, 4)  # benign turns to sandwich around
    # Generation horizon.
    turns_per_decomposition: int = 6
    n_decompositions_per_seed: int = 2
    n_decoys_per_seed: int = 1
    # Policy-swap: how many alternative policies to try relabeling a positive under.
    n_policy_swaps_per_positive: int = 1
    seed: int = 0


@dataclass
class GenSpec:
    """One generation request plus the metadata to reconstruct a Conversation."""

    prompt: str
    kind: str  # "decompose" | "decoy"
    policy: str
    guardrail_id: str


# --------------------------------------------------------------------------- io
def load_seed_rows(samples_path, held_out_ids: set[str]) -> list[dict[str, Any]]:
    """Rows with a policy + transcript + label, minus held-out guardrails."""
    from rl.utils import iter_jsonl

    rows = []
    for r in iter_jsonl(samples_path):
        if r.get("guardrail_id") in held_out_ids:
            continue
        if r.get("policy_prompt") and r.get("input_block") and r.get("label"):
            rows.append(r)
    return rows


def sample_seeds_across_guardrails(
    seed_rows: list[dict[str, Any]], limit: int, rng: random.Random
) -> list[dict[str, Any]]:
    """Take up to ``limit`` rows spread ACROSS guardrails (round-robin), not the
    first ``limit`` in file order — accepted_samples is grouped by guardrail, so
    a naive head() would draw a single policy and produce a degenerate split."""
    if not limit or limit >= len(seed_rows):
        return seed_rows
    by_gid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in seed_rows:
        by_gid[r["guardrail_id"]].append(r)
    groups = list(by_gid.values())
    for g in groups:
        rng.shuffle(g)
    rng.shuffle(groups)
    picked: list[dict[str, Any]] = []
    i = 0
    while len(picked) < limit:
        progressed = False
        for g in groups:
            if i < len(g):
                picked.append(g[i])
                progressed = True
                if len(picked) >= limit:
                    break
        if not progressed:
            break
        i += 1
    return picked


def benign_turn_pool(seed_rows: list[dict[str, Any]]) -> list[Turn]:
    """User/agent turns drawn from BENIGN seed transcripts, used as neutral
    padding in the compose bucket."""
    pool: list[Turn] = []
    for r in seed_rows:
        if r.get("label") == NEGATIVE_LABEL:
            pool.extend(split_transcript(r["input_block"]))
    return [t for t in pool if len(t.text) >= 20]


# ---------------------------------------------- bucket 0: natural compositional
def scan_natural_compositional(
    seed_rows: list[dict[str, Any]],
    encoder: TurnEncoder,
    cfg: DatagenConfig,
) -> tuple[list[Conversation], dict[str, Any]]:
    """Find violating seeds that are ALREADY compositional for the frozen
    classifier: multi-turn transcripts where no single turn scores >= turn_max
    but the whole transcript scores >= concat_min. These are genuine
    compositional positives with no generation — the violation is distributed
    across the original turns (common for inherently multi-turn policies like
    message-repetition or escalation).

    Batched: every seed's per-turn blocks and full concat are scored in one
    length-bucketed pass. Onset is resolved from batched prefix scores.
    """
    candidates: list[tuple[dict[str, Any], list[Turn]]] = []
    for r in seed_rows:
        if r.get("label") != POSITIVE_LABEL:
            continue
        turns = split_transcript(r["input_block"])
        if len(turns) >= 2:
            candidates.append((r, turns))
    if not candidates:
        return [], {"scanned": 0, "kept": 0}

    policies: list[str] = []
    blocks: list[str] = []
    spans: list[tuple[int, int, int]] = []  # (turn_lo, turn_hi, concat_idx)
    for r, turns in candidates:
        lo = len(blocks)
        policies += [r["policy_prompt"]] * len(turns)
        blocks += [turn_block(t) for t in turns]
        concat_idx = len(blocks)
        policies.append(r["policy_prompt"])
        blocks.append(transcript(turns))
        spans.append((lo, concat_idx, concat_idx))
    scores = encoder.score_blocks(policies, blocks).tolist()

    kept: list[Conversation] = []
    accepted: list[tuple[Conversation, list[Turn], int | None]] = []
    by_gid: dict[str, int] = defaultdict(int)
    for (r, turns), (lo, hi, concat_idx) in zip(candidates, spans):
        turn_scores = scores[lo:hi]
        concat = scores[concat_idx]
        if max(turn_scores) < cfg.turn_max and concat >= cfg.concat_min:
            conv = Conversation(
                conv_id=new_conv_id("natural"),
                guardrail_id=r["guardrail_id"],
                policy_prompt=r["policy_prompt"],
                turns=turns,
                label=POSITIVE_LABEL,
                first_violation_turn=len(turns) - 1,  # provisional
                source="natural_compositional",
                meta={"max_turn_score": max(turn_scores), "concat_score": concat},
            )
            accepted.append((conv, turns, None))
            kept.append(conv)
            by_gid[r["guardrail_id"]] += 1
    _resolve_onsets(accepted, encoder, cfg)
    stats = {"scanned": len(candidates), "kept": len(kept), "by_guardrail": dict(by_gid)}
    logger.info("natural compositional: kept %d / %d violating seeds", len(kept), len(candidates))
    return kept, stats


# ------------------------------------------------------------------- bucket 1
def build_compose_bucket(
    seed_rows: list[dict[str, Any]],
    cfg: DatagenConfig,
    rng: random.Random,
) -> list[Conversation]:
    """Sandwich each seed transcript's turns among sampled benign turns.

    A violating seed -> conversation whose first_violation_turn is the index of
    the turn that carried the original violation (approximated as the first turn
    of the inserted seed block: the seed block as a whole is what violates).
    Benign seeds -> benign padded conversations."""
    pool = benign_turn_pool(seed_rows)
    if not pool:
        logger.warning("compose: empty benign turn pool; skipping")
        return []

    convs: list[Conversation] = []
    lo, hi = cfg.compose_pad_turns
    for r in seed_rows:
        seed_turns = split_transcript(r["input_block"])
        if not seed_turns:
            continue
        pre = [rng.choice(pool) for _ in range(rng.randint(lo, hi))]
        post = [rng.choice(pool) for _ in range(rng.randint(lo, hi))]
        turns = pre + seed_turns + post
        if r["label"] == POSITIVE_LABEL:
            first_violation = len(pre) + len(seed_turns) - 1  # seed completes at its last turn
            label = POSITIVE_LABEL
        else:
            first_violation = None
            label = NEGATIVE_LABEL
        convs.append(
            Conversation(
                conv_id=new_conv_id("compose"),
                guardrail_id=r["guardrail_id"],
                policy_prompt=r["policy_prompt"],
                turns=turns,
                label=label,
                first_violation_turn=first_violation,
                source="compose",
                meta={"seed_label": r["label"], "n_pad_pre": len(pre), "n_pad_post": len(post)},
            )
        )
    logger.info("compose: %d conversations from %d seeds", len(convs), len(seed_rows))
    return convs


# ------------------------------------------------------- buckets 2 & 3 prompts
def build_generation_specs(
    seed_rows: list[dict[str, Any]],
    cfg: DatagenConfig,
    tokenizer: Any,
    rng: random.Random,
) -> list[GenSpec]:
    """Decompose prompts (from violating seeds) + benign-decoy prompts."""
    specs: list[GenSpec] = []
    violating = [r for r in seed_rows if r["label"] == POSITIVE_LABEL]
    for r in violating:
        for _ in range(cfg.n_decompositions_per_seed):
            specs.append(
                GenSpec(
                    prompt=format_decompose(
                        r["policy_prompt"], r["input_block"], cfg.turns_per_decomposition, tokenizer
                    ),
                    kind="decompose",
                    policy=r["policy_prompt"],
                    guardrail_id=r["guardrail_id"],
                )
            )
        for _ in range(cfg.n_decoys_per_seed):
            specs.append(
                GenSpec(
                    prompt=format_benign_decoy(
                        r["policy_prompt"], r["input_block"], cfg.turns_per_decomposition, tokenizer
                    ),
                    kind="decoy",
                    policy=r["policy_prompt"],
                    guardrail_id=r["guardrail_id"],
                )
            )
    rng.shuffle(specs)
    logger.info(
        "generation specs: %d (%d decompose, %d decoy) from %d violating seeds",
        len(specs),
        sum(s.kind == "decompose" for s in specs),
        sum(s.kind == "decoy" for s in specs),
        len(violating),
    )
    return specs


def _parse_turns(raw: str) -> tuple[list[Turn], int | None] | None:
    """Parse a generator output into (turns, onset_turn|None). Accepts either a
    `turns` list of {role,text} or a tagged `input_block` string."""
    try:
        parsed = _extract_json(raw)
    except Exception:  # noqa: BLE001 - any parse failure -> drop this candidate
        return None
    if not isinstance(parsed, dict):
        return None
    onset = parsed.get("onset_turn")
    onset = int(onset) if isinstance(onset, (int, float)) else None

    if isinstance(parsed.get("turns"), list):
        turns: list[Turn] = []
        for item in parsed["turns"]:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).lower()
            text = str(item.get("text", "")).strip()
            if role in ("user", "agent") and text:
                turns.append(Turn(role=role, text=text))
        return (turns, onset) if turns else None

    if isinstance(parsed.get("input_block"), str):
        turns = split_transcript(parsed["input_block"])
        return (turns, onset) if turns else None
    return None


# -------------------------------------------------------------- teacher filter
def _turn_scores(encoder: TurnEncoder, policy: str, turns: list[Turn]) -> list[float]:
    return encoder.score_blocks([policy] * len(turns), [turn_block(t) for t in turns]).tolist()


def _concat_score(encoder: TurnEncoder, policy: str, turns: list[Turn]) -> float:
    return float(encoder.score_blocks([policy], [transcript(turns)])[0].item())


def accept_decompositions(
    specs: list[GenSpec],
    raw_outputs: list[str],
    encoder: TurnEncoder,
    cfg: DatagenConfig,
) -> tuple[list[Conversation], list[dict[str, Any]]]:
    """Apply the teacher filter to generated conversations.

    decompose -> keep iff max per-turn score < turn_max AND concat >= concat_min.
    decoy     -> keep iff concat < decoy_max (benign as a whole).

    Scoring is batched: all candidates' per-turn blocks and full-concat blocks go
    through the encoder's length-bucketing in ONE call, instead of a couple of
    single-item forwards per candidate."""
    # 1. Parse; keep only well-formed candidates for scoring.
    parsed: list[tuple[GenSpec, list[Turn], int | None]] = []
    diagnostics: list[dict[str, Any]] = []
    for spec, raw in zip(specs, raw_outputs):
        p = _parse_turns(raw)
        if p is None:
            diagnostics.append({"kind": spec.kind, "reject": "parse", "guardrail_id": spec.guardrail_id})
            continue
        turns, onset_hint = p
        if len(turns) < 2:
            diagnostics.append({"kind": spec.kind, "reject": "too_few_turns", "guardrail_id": spec.guardrail_id})
            continue
        parsed.append((spec, turns, onset_hint))

    # 2. Batch-score every per-turn block + the full concat for each candidate.
    policies: list[str] = []
    blocks: list[str] = []
    spans: list[tuple[int, int, int]] = []  # (turn_lo, turn_hi, concat_idx) into `scores`
    for spec, turns, _ in parsed:
        lo = len(blocks)
        policies += [spec.policy] * len(turns)
        blocks += [turn_block(t) for t in turns]
        concat_idx = len(blocks)
        policies.append(spec.policy)
        blocks.append(transcript(turns))
        spans.append((lo, concat_idx, concat_idx))
    scores = encoder.score_blocks(policies, blocks).tolist() if blocks else []

    # 3. Apply accept/reject using the batched scores.
    kept: list[Conversation] = []
    accepted_decomps: list[tuple[Conversation, list[Turn], int | None]] = []
    for (spec, turns, onset_hint), (lo, hi, concat_idx) in zip(parsed, spans):
        turn_scores = scores[lo:hi]
        concat = scores[concat_idx]
        diag = {
            "kind": spec.kind,
            "guardrail_id": spec.guardrail_id,
            "max_turn_score": max(turn_scores),
            "concat_score": concat,
        }
        if spec.kind == "decompose":
            if max(turn_scores) >= cfg.turn_max:
                diagnostics.append({**diag, "reject": "turn_scores_high"})
                continue
            if concat < cfg.concat_min:
                diagnostics.append({**diag, "reject": "concat_low"})
                continue
            conv = Conversation(
                conv_id=new_conv_id("decompose"),
                guardrail_id=spec.guardrail_id,
                policy_prompt=spec.policy,
                turns=turns,
                label=POSITIVE_LABEL,
                first_violation_turn=len(turns) - 1,  # provisional; resolved below
                source="decompose",
                meta=diag,
            )
            accepted_decomps.append((conv, turns, onset_hint))
            kept.append(conv)
        else:  # decoy
            if concat >= cfg.decoy_max:
                diagnostics.append({**diag, "reject": "decoy_concat_high"})
                continue
            kept.append(
                Conversation(
                    conv_id=new_conv_id("decoy"),
                    guardrail_id=spec.guardrail_id,
                    policy_prompt=spec.policy,
                    turns=turns,
                    label=NEGATIVE_LABEL,
                    first_violation_turn=None,
                    source="hard_negative_decoy",
                    meta=diag,
                )
            )

    # 4. Resolve onset for accepted decompositions (batched prefix scores).
    _resolve_onsets(accepted_decomps, encoder, cfg)
    logger.info(
        "teacher filter: kept %d / %d (%d diag)", len(kept), len(specs), len(diagnostics)
    )
    return kept, diagnostics


def _resolve_onsets(
    accepted: list[tuple[Conversation, list[Turn], int | None]],
    encoder: TurnEncoder,
    cfg: DatagenConfig,
) -> None:
    """Set first_violation_turn = first prefix whose concat score crosses
    concat_min (fallback: generator hint, then last turn). All prefixes across
    all accepted decompositions are scored in one batched call, then written back
    onto each Conversation in place."""
    if not accepted:
        return
    policies: list[str] = []
    blocks: list[str] = []
    spans: list[tuple[int, int]] = []
    for conv, turns, _ in accepted:
        lo = len(blocks)
        for t in range(len(turns)):
            policies.append(conv.policy_prompt)
            blocks.append(transcript(turns[: t + 1]))
        spans.append((lo, len(blocks)))
    scores = encoder.score_blocks(policies, blocks).tolist()
    for (conv, turns, onset_hint), (lo, hi) in zip(accepted, spans):
        prefix_scores = scores[lo:hi]
        onset = next((t for t, s in enumerate(prefix_scores) if s >= cfg.concat_min), None)
        if onset is None:
            onset = onset_hint if (onset_hint is not None and 0 <= onset_hint < len(turns)) else len(turns) - 1
        conv.first_violation_turn = onset
    return len(turns) - 1


# --------------------------------------------------------- bucket 3b: policy swap
def build_policy_swaps(
    positives: list[Conversation],
    seed_rows: list[dict[str, Any]],
    encoder: TurnEncoder,
    cfg: DatagenConfig,
    rng: random.Random,
) -> tuple[list[Conversation], list[dict[str, Any]]]:
    """Relabel a compositional positive as benign under a DIFFERENT policy that
    it does not violate. Forces the aggregator to read the policy-conditioned
    geometry, not conversation surface features.

    Accept a swap iff, under the new policy, every per-turn score AND the concat
    score are below decoy_max (genuinely benign)."""
    policies = list({r["policy_prompt"]: r["guardrail_id"] for r in seed_rows}.items())
    if len(policies) < 2:
        return [], []
    swaps: list[Conversation] = []
    diagnostics: list[dict[str, Any]] = []
    for conv in positives:
        candidates = [(p, gid) for p, gid in policies if gid != conv.guardrail_id]
        rng.shuffle(candidates)
        for policy, gid in candidates[: cfg.n_policy_swaps_per_positive]:
            turn_scores = _turn_scores(encoder, policy, conv.turns)
            concat = _concat_score(encoder, policy, conv.turns)
            if max(turn_scores) < cfg.decoy_max and concat < cfg.decoy_max:
                swaps.append(
                    Conversation(
                        conv_id=new_conv_id("swap"),
                        guardrail_id=gid,
                        policy_prompt=policy,
                        turns=conv.turns,
                        label=NEGATIVE_LABEL,
                        first_violation_turn=None,
                        source="hard_negative_policy_swap",
                        meta={"from_conv": conv.conv_id, "max_turn_score": max(turn_scores), "concat_score": concat},
                    )
                )
                break
            diagnostics.append(
                {"from_conv": conv.conv_id, "swap_gid": gid, "reject": "not_benign_under_swap",
                 "max_turn_score": max(turn_scores), "concat_score": concat}
            )
    logger.info("policy swaps: kept %d from %d positives", len(swaps), len(positives))
    return swaps, diagnostics


def validate_all(convs: list[Conversation]) -> list[Conversation]:
    """Drop malformed conversations rather than fail the whole run."""
    ok: list[Conversation] = []
    for conv in convs:
        try:
            conv.validate()
            ok.append(conv)
        except ValueError as exc:
            logger.warning("dropping %s: %s", conv.conv_id, exc)
    return ok


# ------------------------------------------------ recombination (latent bank)
@dataclass
class RecombineConfig:
    """How many conversations to synthesize by recombining cores + benign turns."""

    benign_per_policy: int = 40  # pure-benign conversations per policy
    recontext_per_core: int = 60  # re-contextualized variants per compositional/violating core
    pad_lo: int = 1
    pad_hi: int = 5
    swap_targets_per_core: int = 3  # cross-policy benign relabels per compositional core
    max_turns: int = 24


def _pad_turns(pool: list[Turn], n: int, rng: random.Random) -> list[Turn]:
    return [rng.choice(pool) for _ in range(n)] if pool and n > 0 else []


def plan_recombinations(
    cores: list[Conversation],
    benign_pool_by_policy: dict[str, list[Turn]],
    policy_of_guardrail: dict[str, str],
    cfg: RecombineConfig,
    rng: random.Random,
) -> list[Conversation]:
    """Plan (WITHOUT latents) a large set of conversations by recombining
    compositional/violating cores with benign padding, reusing the monotone
    label/onset convention of :func:`build_compose_bucket`.

    - pure benign conversations from each policy's benign pool
    - each core re-contextualized into many benign surroundings (onset shifts
      with the prefix length) — multiplies the scarce compositional signal
    - cross-policy swaps: a compositional core relabeled benign under another
      policy (the geometry cue; latents must exist under that policy in the bank)

    A "core" is a Conversation with source in {natural_compositional, decompose,
    compose(violating seed)} — its turns collectively violate its policy.
    """
    planned: list[Conversation] = []
    policies = sorted(benign_pool_by_policy)

    # Pure-benign conversations per policy.
    for policy in policies:
        pool = benign_pool_by_policy[policy]
        if not pool:
            continue
        gid = policy_of_guardrail.get(policy, "unknown")
        for _ in range(cfg.benign_per_policy):
            n = rng.randint(cfg.pad_lo + 2, min(cfg.max_turns, cfg.pad_hi + 5))
            turns = _pad_turns(pool, n, rng)
            planned.append(
                Conversation(
                    conv_id=new_conv_id("rebenign"), guardrail_id=gid, policy_prompt=policy,
                    turns=turns, label=NEGATIVE_LABEL, first_violation_turn=None,
                    source="recomb_benign",
                )
            )

    # Re-contextualize each core.
    is_comp = set(COMPOSITIONAL_SOURCES)
    for core in cores:
        policy = core.policy_prompt
        pool = benign_pool_by_policy.get(policy, [])
        core_turns = core.turns
        core_onset = core.first_violation_turn
        for _ in range(cfg.recontext_per_core):
            pre = _pad_turns(pool, rng.randint(cfg.pad_lo, cfg.pad_hi), rng)
            post = _pad_turns(pool, rng.randint(0, cfg.pad_hi), rng)
            turns = pre + core_turns + post
            if len(turns) > cfg.max_turns:
                continue
            # Onset shifts by the inserted prefix length; a core with an internal
            # onset keeps it relative to the core's start.
            onset = len(pre) + (core_onset if core_onset is not None else len(core_turns) - 1)
            planned.append(
                Conversation(
                    conv_id=new_conv_id("recomp"), guardrail_id=core.guardrail_id,
                    policy_prompt=policy, turns=turns, label=POSITIVE_LABEL,
                    first_violation_turn=onset, source="recomb_" + core.source,
                    meta={"from_core": core.conv_id},
                )
            )
        # Cross-policy swaps (compositional cores only): benign under another policy.
        if core.source in is_comp:
            others = [p for p in policies if p != policy]
            rng.shuffle(others)
            for target in others[: cfg.swap_targets_per_core]:
                gid = policy_of_guardrail.get(target, "unknown")
                pool_t = benign_pool_by_policy.get(target, [])
                pre = _pad_turns(pool_t, rng.randint(0, cfg.pad_hi), rng)
                turns = pre + core_turns
                if len(turns) > cfg.max_turns:
                    continue
                planned.append(
                    Conversation(
                        conv_id=new_conv_id("reswap"), guardrail_id=gid, policy_prompt=target,
                        turns=turns, label=NEGATIVE_LABEL, first_violation_turn=None,
                        source="recomb_swap", meta={"from_core": core.conv_id, "swap_from": policy},
                    )
                )
    rng.shuffle(planned)
    logger.info("planned %d recombined conversations from %d cores", len(planned), len(cores))
    return planned


# --------------------------------------------------------------------- dedup
def _norm_text(text: str) -> str:
    return " ".join(text.lower().split())


def _conv_signature(conv: Conversation) -> tuple:
    """Exact key: policy + ordered normalized (role, text) turns."""
    return (_norm_text(conv.policy_prompt),) + tuple(
        (t.role, _norm_text(t.text)) for t in conv.turns
    )


def _shingles(conv: Conversation, k: int = 5) -> set[int]:
    """Hashed k-word shingles over the concatenated turn text (order-sensitive
    via a leading role marker)."""
    words = []
    for t in conv.turns:
        words.append(f"<{t.role}>")
        words.extend(_norm_text(t.text).split())
    if len(words) < k:
        return {hash(tuple(words))}
    return {hash(tuple(words[i : i + k])) for i in range(len(words) - k + 1)}


def dedup_conversations(
    convs: list[Conversation],
    jaccard_threshold: float = 0.9,
    lsh_bands: int = 20,
    protect_sources: tuple[str, ...] = COMPOSITIONAL_SOURCES,
) -> tuple[list[Conversation], dict[str, int]]:
    """Drop exact turn-sequence duplicates, then near-duplicates.

    Near-dup: LSH-banded shingle buckets restrict Jaccard comparisons to
    plausibly-similar pairs (so this stays ~linear on ~175k convs); a candidate
    is dropped if it exceeds ``jaccard_threshold`` against an already-kept conv.
    Compositional-core sources are never dropped as the LATER member of a pair
    (they are the scarce signal) — they can still evict exact dups of themselves.
    """
    stats = {"input": len(convs), "exact_dup": 0, "near_dup": 0}
    kept: list[Conversation] = []
    seen_exact: set[tuple] = set()
    kept_shingles: list[set[int]] = []
    bands: dict[tuple, list[int]] = defaultdict(list)  # band-hash -> kept indices

    for conv in convs:
        sig = _conv_signature(conv)
        if sig in seen_exact:
            stats["exact_dup"] += 1
            continue
        shingles = _shingles(conv)
        # Candidate neighbours: share at least one LSH band.
        ordered = sorted(shingles)
        band_size = max(1, len(ordered) // lsh_bands)
        band_keys = [
            (b, hash(tuple(ordered[b * band_size : (b + 1) * band_size])))
            for b in range(lsh_bands)
        ]
        candidates = {idx for bk in band_keys for idx in bands.get(bk, ())}
        is_near = False
        if conv.source not in protect_sources:
            for idx in candidates:
                other = kept_shingles[idx]
                inter = len(shingles & other)
                if inter and inter / len(shingles | other) >= jaccard_threshold:
                    is_near = True
                    break
        if is_near:
            stats["near_dup"] += 1
            continue
        seen_exact.add(sig)
        new_idx = len(kept)
        kept.append(conv)
        kept_shingles.append(shingles)
        for bk in band_keys:
            bands[bk].append(new_idx)

    stats["kept"] = len(kept)
    logger.info(
        "dedup: %d -> %d (%d exact, %d near)",
        stats["input"], stats["kept"], stats["exact_dup"], stats["near_dup"],
    )
    return kept, stats
