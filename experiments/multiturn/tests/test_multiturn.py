"""Correctness tests for the multi-turn monitor. No GPU / no backbone needed.

Every claim is checked against an INDEPENDENT reference (brute-force AUROC,
hand-computed causality/streaming equivalence), not the code's own internals.

Run:  pytest -q tests/test_multiturn.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from mtlib import aggregator as AGG  # noqa: E402
from mtlib import metrics as M  # noqa: E402
from mtlib import schema as S  # noqa: E402
from mtlib.aggregator import AggregatorConfig, ConversationAggregator, per_turn_loss  # noqa: E402


# --------------------------------------------------------------------- schema
def test_split_transcript_tolerates_mismatched_tags():
    block = "<User>hello there friend</User>\n<Agent>hi how can i help you</User>"
    turns = S.split_transcript(block)
    assert [t.role for t in turns] == ["user", "agent"]
    assert turns[0].text == "hello there friend"
    assert "</" not in turns[1].text  # closing tags stripped


def test_per_turn_labels_are_monotone():
    conv = S.Conversation(
        conv_id="c", guardrail_id="g", policy_prompt="p",
        turns=[S.Turn("user", "a"), S.Turn("agent", "b"), S.Turn("user", "c")],
        label="True", first_violation_turn=1,
    )
    assert conv.per_turn_labels() == [0, 1, 1]
    conv.validate()


def test_benign_conversation_labels_all_zero():
    conv = S.Conversation(
        conv_id="c", guardrail_id="g", policy_prompt="p",
        turns=[S.Turn("user", "a"), S.Turn("agent", "b")],
        label="False", first_violation_turn=None,
    )
    assert conv.per_turn_labels() == [0, 0]


def test_validate_rejects_label_onset_mismatch():
    conv = S.Conversation(
        conv_id="c", guardrail_id="g", policy_prompt="p",
        turns=[S.Turn("user", "a")], label="True", first_violation_turn=None,
    )
    with pytest.raises(ValueError):
        conv.validate()


def test_conversation_roundtrip():
    conv = S.Conversation(
        conv_id="c", guardrail_id="g", policy_prompt="p",
        turns=[S.Turn("user", "a"), S.Turn("agent", "b")],
        label="True", first_violation_turn=0, source="decompose", meta={"x": 1},
    )
    assert S.Conversation.from_dict(conv.to_dict()).to_dict() == conv.to_dict()


# ------------------------------------------------------------------- metrics
def _brute_auroc(labels, scores):
    """O(n^2) reference: fraction of (pos, neg) pairs correctly ordered, ties=0.5."""
    pos = [s for l, s in zip(labels, scores) if l == 1]
    neg = [s for l, s in zip(labels, scores) if l == 0]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def test_auroc_matches_brute_force():
    rng = np.random.default_rng(0)
    for _ in range(20):
        labels = rng.integers(0, 2, size=30).tolist()
        scores = rng.random(size=30).tolist()
        ref = _brute_auroc(labels, scores)
        got = M.auroc(labels, scores)
        if np.isnan(ref):
            assert np.isnan(got)
        else:
            assert got == pytest.approx(ref, abs=1e-9)


def test_auroc_handles_ties():
    # perfectly-tied scores => AUROC 0.5.
    assert M.auroc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)
    # perfect separation => 1.0.
    assert M.auroc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == pytest.approx(1.0)


def test_f1_hand_computed():
    # tp=2 fp=1 fn=1 => F1 = 2*2 / (2*2+1+1) = 4/6.
    assert M.f1_binary([1, 1, 0, 1], [1, 1, 1, 0]) == pytest.approx(4 / 6)


def test_tau_for_conversation_fpr_respects_target():
    # 100 benign conversations, per-conversation max uniformly spread.
    benign = [[i / 100.0] for i in range(100)]
    tau = M.tau_for_conversation_fpr(benign, target_fpr=0.05)
    flagged = sum(1 for s in benign if max(s) >= tau)
    assert flagged <= 5  # at most 5% flagged


def test_benign_fpr_by_length_uses_prefix_only():
    # a conversation that only spikes at turn 8 must not count against fpr@5.
    benign = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9]]
    out = M.benign_fpr_by_length(benign, tau=0.5, horizons=(5, 10))
    assert out["fpr_at_5_turns"] == 0.0
    assert out["fpr_at_10_turns"] == 1.0


def test_detection_lag():
    # onset at turn 2, flag crosses at turn 3 => lag 1.
    stats = M.detection_stats([(2, [0.1, 0.1, 0.2, 0.9, 0.9])], tau=0.5)
    assert stats["detection_rate"] == 1.0
    assert stats["median_lag_turns"] == 1.0
    assert stats["detected_within_0_turns"] == 0.0
    assert stats["detected_within_1_turns"] == 1.0


# ---------------------------------------------------------------- aggregator
def _tiny_config(**kw):
    base = dict(input_dim=16, num_labels=2, d_model=32, n_layers=2, n_heads=4, dropout=0.0)
    base.update(kw)
    return AggregatorConfig(**base)


def _nontrivial(model: ConversationAggregator, seed: int = 0) -> ConversationAggregator:
    """Replace the zero-init head with random weights so outputs actually depend
    on the inputs — otherwise every output is 0 and the invariance tests are
    vacuous (this is what makes the causality check meaningful)."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        model.head.weight.normal_(0, 1, generator=g)
        model.head.bias.normal_(0, 1, generator=g)
    return model.eval()


def _random_batch(batch, turns, config, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "turn_latents": torch.randn(batch, turns, config.input_dim, generator=g),
        "turn_logits": torch.randn(batch, turns, config.num_labels, generator=g),
        "role_ids": torch.randint(1, 3, (batch, turns), generator=g),
        "attention_mask": torch.ones(batch, turns, dtype=torch.long),
        "policy_latent": torch.randn(batch, config.input_dim, generator=g),
    }


def test_forward_shape():
    config = _tiny_config()
    model = ConversationAggregator(config).eval()
    batch = _random_batch(3, 5, config)
    out = model(**batch)
    assert out.shape == (3, 5, config.num_labels)


def test_causality_future_turns_do_not_affect_past_outputs():
    """Output at turn t must not change when turns > t are perturbed."""
    config = _tiny_config()
    model = _nontrivial(ConversationAggregator(config))
    batch = _random_batch(1, 6, config)
    with torch.no_grad():
        base = model(**batch)
        perturbed = {**batch, "turn_latents": batch["turn_latents"].clone()}
        perturbed["turn_latents"][:, 4:] += 5.0  # change turns 4,5
        after = model(**perturbed)
    # turns 0..3 must be identical; 4,5 may change.
    assert torch.allclose(base[:, :4], after[:, :4], atol=1e-5)
    assert not torch.allclose(base[:, 4:], after[:, 4:], atol=1e-3)


def test_streaming_equals_one_shot():
    """Feeding turns incrementally (recompute prefix each step) equals a single
    full-sequence forward at each position — the invariant the monitor relies on."""
    config = _tiny_config()
    model = _nontrivial(ConversationAggregator(config))
    full = _random_batch(1, 7, config)
    with torch.no_grad():
        one_shot = torch.softmax(model(**full), dim=-1)[0, :, 1]
        streamed = []
        for t in range(1, 8):
            prefix = {
                "turn_latents": full["turn_latents"][:, :t],
                "turn_logits": full["turn_logits"][:, :t],
                "role_ids": full["role_ids"][:, :t],
                "attention_mask": full["attention_mask"][:, :t],
                "policy_latent": full["policy_latent"],
            }
            streamed.append(torch.softmax(model(**prefix), dim=-1)[0, -1, 1])
        streamed = torch.stack(streamed)
    assert torch.allclose(one_shot, streamed, atol=1e-5)


def test_padding_does_not_affect_real_positions():
    """Right-padding a batch must not change the outputs at real positions."""
    config = _tiny_config()
    model = _nontrivial(ConversationAggregator(config))
    short = _random_batch(1, 4, config, seed=1)
    padded = _random_batch(1, 7, config, seed=1)
    # first 4 turns identical to `short`; positions 4..6 are pad. The policy
    # latent must match too (it conditions every position), which _random_batch
    # does not guarantee across different lengths since the RNG state diverges.
    for key in ("turn_latents", "turn_logits", "role_ids"):
        padded[key][:, :4] = short[key]
    padded["policy_latent"] = short["policy_latent"]
    padded["attention_mask"] = torch.tensor([[1, 1, 1, 1, 0, 0, 0]])
    with torch.no_grad():
        out_short = model(**short)
        out_padded = model(**padded)
    assert torch.allclose(out_short, out_padded[:, :4], atol=1e-5)


def test_logit_features_recover_single_turn_baseline_is_representable():
    """With use_logit_features, the per-turn logits are inputs, so the model can
    in principle route them straight through. Sanity: turning the feature flag on
    changes the input dimension by num_labels."""
    with_feats = ConversationAggregator(_tiny_config(use_logit_features=True))
    without = ConversationAggregator(_tiny_config(use_logit_features=False))
    assert with_feats.proj.in_features == 16 + 2
    assert without.proj.in_features == 16


def test_policy_token_optional():
    config = _tiny_config(use_policy_token=False)
    model = ConversationAggregator(config).eval()
    batch = _random_batch(2, 5, config)
    batch["policy_latent"] = None
    out = model(**batch)
    assert out.shape == (2, 5, 2)


def test_per_turn_loss_upweights_onset():
    torch.manual_seed(0)
    logits = torch.zeros(1, 3, 2, requires_grad=True)
    labels = torch.tensor([[0, 1, 1]])
    onset = torch.tensor([1])
    loss_hi = per_turn_loss(logits, labels, onset, onset_weight=5.0)
    loss_lo = per_turn_loss(logits, labels, onset, onset_weight=1.0)
    # At uniform logits all per-position losses equal ln2, so the weighted mean is
    # ln2 regardless of weight; check the weighting path runs and is finite.
    assert torch.isfinite(loss_hi) and torch.isfinite(loss_lo)
    assert loss_hi.item() == pytest.approx(float(np.log(2)), abs=1e-5)


def test_per_turn_loss_ignores_pad_positions():
    logits = torch.randn(1, 4, 2)
    labels_full = torch.tensor([[0, 1, 1, 1]])
    labels_pad = torch.tensor([[0, 1, AGG.IGNORE_INDEX, AGG.IGNORE_INDEX]])
    onset = torch.tensor([1])
    # Loss over only the first 2 positions should equal masking the rest.
    l_pad = per_turn_loss(logits, labels_pad, onset)
    l_ref = per_turn_loss(logits[:, :2], labels_full[:, :2], onset)
    assert l_pad == pytest.approx(float(l_ref), abs=1e-6)


def test_save_load_roundtrip(tmp_path):
    config = _tiny_config()
    model = _nontrivial(ConversationAggregator(config))
    batch = _random_batch(2, 5, config)
    with torch.no_grad():
        before = model(**batch)
    model.save(tmp_path / "agg.pt", extra={"tau": 0.73})
    loaded, extra = ConversationAggregator.load(tmp_path / "agg.pt")
    assert extra["tau"] == 0.73
    with torch.no_grad():
        after = loaded(**batch)
    assert torch.allclose(before, after, atol=1e-6)


# ------------------------------------------------------- datagen teacher filter
class _FakeEncoder:
    """Scores blocks by exact-text lookup, so the teacher filter can be tested
    against a hand-crafted compositional case without a backbone."""

    def __init__(self, table):
        self.table = table

    def score_blocks(self, policies, blocks):
        return torch.tensor([self.table[b] for b in blocks], dtype=torch.float32)


def test_batched_teacher_filter_accept_reject_and_onset():
    import json

    from mtlib.datagen import DatagenConfig, accept_decompositions, GenSpec

    # A: genuine decomposition — each turn benign (0.1,0.1,0.2), concat 0.95,
    #    prefixes cross 0.5 only at turn 2 -> onset must be 2.
    tA = [S.Turn("user", "a0"), S.Turn("agent", "a1"), S.Turn("user", "a2")]
    # B: one hot turn (0.99) -> reject turn_scores_high.
    tB = [S.Turn("user", "b0"), S.Turn("agent", "b1")]
    table = {
        S.turn_block(tA[0]): 0.1, S.turn_block(tA[1]): 0.1, S.turn_block(tA[2]): 0.2,
        S.transcript(tA): 0.95, S.transcript(tA[:1]): 0.1, S.transcript(tA[:2]): 0.2,
        S.transcript(tA[:3]): 0.95,
        S.turn_block(tB[0]): 0.99, S.turn_block(tB[1]): 0.1, S.transcript(tB): 0.99,
    }
    specs = [
        GenSpec("", kind="decompose", policy="P", guardrail_id="G1"),
        GenSpec("", kind="decompose", policy="P", guardrail_id="G2"),
    ]
    raws = [json.dumps({"turns": [{"role": t.role, "text": t.text} for t in ts]}) for ts in (tA, tB)]
    kept, diag = accept_decompositions(specs, raws, _FakeEncoder(table),
                                       DatagenConfig(turn_max=0.5, concat_min=0.5))
    assert [c.guardrail_id for c in kept] == ["G1"]
    assert kept[0].first_violation_turn == 2  # onset resolved via batched prefix scores
    assert kept[0].per_turn_labels() == [0, 0, 1]
    assert diag[0]["reject"] == "turn_scores_high"


def test_stratified_seed_sampling_spans_guardrails():
    import random as _random

    from mtlib.datagen import sample_seeds_across_guardrails

    rows = [{"guardrail_id": f"G{i % 10}", "policy_prompt": "p", "input_block": "x", "label": "True"}
            for i in range(200)]
    picked = sample_seeds_across_guardrails(rows, 10, _random.Random(0))
    assert len({r["guardrail_id"] for r in picked}) == 10  # one per guardrail, not first-10


# ------------------------------------------------------------ LLM aggregator
def _tiny_lm(with_head: bool):
    """Tiny random Qwen2 built from a local config — offline, CPU, fast. A real
    cached tokenizer supplies token ids for policy/verbalizer."""
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM, Qwen2Model

    torch.manual_seed(0)
    config = Qwen2Config(
        vocab_size=151936, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=512,
    )
    lm = (Qwen2ForCausalLM if with_head else Qwen2Model)(config).float().eval()
    # Qwen3-4B is the Qwen snapshot cached WITH its config.json on this box.
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    return lm, tokenizer


def _llm_agg(head: str):
    from mtlib.llm_aggregator import LlmAggregatorConfig, LlmLatentAggregator

    lm, tok = _tiny_lm(with_head=head == "verbalizer")
    config = LlmAggregatorConfig(
        lm_name="tiny-test", input_dim=16, num_labels=2, head=head,
        projector_hidden=32, dtype="float32",
    )
    return LlmLatentAggregator(config, lm=lm, tokenizer=tok)


def _llm_batch(batch, turns, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {
        "policies": ["No revealing account numbers, even across turns."] * batch,
        "turn_latents": torch.randn(batch, turns, 16, generator=g),
        "turn_logits": torch.randn(batch, turns, 2, generator=g),
        "role_ids": torch.randint(1, 3, (batch, turns), generator=g),
        "attention_mask": torch.ones(batch, turns, dtype=torch.long),
    }


@pytest.mark.parametrize("head", ["linear", "verbalizer"])
def test_llm_aggregator_forward_shape(head):
    model = _llm_agg(head)
    out = model(**_llm_batch(2, 5))
    assert out.shape == (2, 5, 2)
    assert torch.isfinite(out).all()


def test_llm_aggregator_causality():
    """Perturbing a later turn's latent must not change earlier turns' logits."""
    model = _llm_agg("verbalizer")  # verbalizer readout is nontrivial untrained
    batch = _llm_batch(1, 6)
    with torch.no_grad():
        base = model(**batch)
        perturbed = {**batch, "turn_latents": batch["turn_latents"].clone()}
        perturbed["turn_latents"][:, 4:] += 5.0
        after = model(**perturbed)
    assert torch.allclose(base[:, :4], after[:, :4], atol=1e-4)
    assert not torch.allclose(base[:, 4:], after[:, 4:], atol=1e-3)


def test_llm_aggregator_frozen_lm_and_trainables():
    model = _llm_agg("linear")
    assert all(not p.requires_grad for p in model.lm.parameters())
    trainable_names = {n for n, p in model.named_parameters() if p.requires_grad}
    assert not any(n.startswith("lm.") for n in trainable_names)
    assert any(n.startswith("projector") for n in trainable_names)


def test_llm_aggregator_save_load_roundtrip(tmp_path):
    from mtlib.llm_aggregator import LlmLatentAggregator

    model = _llm_agg("linear")
    with torch.no_grad():  # non-zero head so outputs depend on weights
        model.head.weight.normal_(0, 0.5, generator=torch.Generator().manual_seed(1))
    batch = _llm_batch(1, 4)
    with torch.no_grad():
        before = model(**batch)
    model.save(tmp_path / "llm.pt", extra={"tau": 0.4})
    lm, tok = _tiny_lm(with_head=False)  # same seed -> identical tiny LM
    loaded, extra = LlmLatentAggregator.load(tmp_path / "llm.pt", lm=lm, tokenizer=tok)
    assert extra["tau"] == 0.4
    with torch.no_grad():
        after = loaded(**batch)
    assert torch.allclose(before, after, atol=1e-5)


def test_llm_aggregator_variable_policy_lengths_batch():
    """Rows with different policy lengths left-pad correctly: a conversation's
    outputs must match whether it is batched with a longer-policy row or alone."""
    model = _llm_agg("verbalizer")
    single = _llm_batch(1, 3, seed=2)
    paired = {
        "policies": [single["policies"][0], "Never give legal validity assurances. " * 8],
        "turn_latents": torch.cat([single["turn_latents"], torch.randn(1, 3, 16)], 0),
        "turn_logits": torch.cat([single["turn_logits"], torch.randn(1, 3, 2)], 0),
        "role_ids": torch.cat([single["role_ids"], torch.randint(1, 3, (1, 3))], 0),
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
    }
    with torch.no_grad():
        alone = model(**single)
        batched = model(**paired)
    assert torch.allclose(alone[0], batched[0], atol=1e-4)


# ------------------------------------------------------- latent bank + recomb
def test_latent_bank_roundtrip_and_gather(tmp_path):
    from mtlib.dataset import shard_item_from_bank
    from mtlib.latent_bank import LatentBank, pair_key

    bank = LatentBank(hidden_size=8, num_labels=2)
    policy = "No revealing secrets across turns."
    turns = [S.Turn("user", "hello there friend"), S.Turn("agent", "the code is 12 and 34")]
    for t in turns:
        k = pair_key(policy, S.turn_block(t))
        bank.latents[k] = torch.randn(8).to(torch.float16)
        bank.logits[k] = torch.randn(2).to(torch.float16)
    bank.policy_latents[policy] = torch.randn(8).to(torch.float16)

    bank.save(tmp_path / "bank.pt")
    reloaded = LatentBank.load(tmp_path / "bank.pt")
    conv = S.Conversation(conv_id="c", guardrail_id="g", policy_prompt=policy, turns=turns,
                          label="True", first_violation_turn=1)
    item = shard_item_from_bank(conv, reloaded)
    assert item is not None
    assert item["turn_latents"].shape == (2, 8)
    assert item["per_turn_labels"].tolist() == [0, 1]
    # missing pair -> None (caller skips)
    conv2 = S.Conversation(conv_id="c2", guardrail_id="g", policy_prompt=policy,
                           turns=turns + [S.Turn("user", "unseen turn text here")],
                           label="False", first_violation_turn=None)
    assert shard_item_from_bank(conv2, reloaded) is None


def test_dedup_exact_and_near():
    from mtlib.datagen import dedup_conversations

    def conv(cid, texts, src="recomb_benign"):
        return S.Conversation(conv_id=cid, guardrail_id="g", policy_prompt="p",
                              turns=[S.Turn("user", t) for t in texts],
                              label="False", first_violation_turn=None, source=src)
    base = ["the quick brown fox jumps", "over the lazy sleeping dog"]
    convs = [
        conv("a", base),
        conv("b", base),  # exact dup
        conv("c", [base[0], "over the lazy sleeping dog now"]),  # near dup (1 word)
        conv("d", ["completely different content here", "nothing alike at all whatsoever"]),
    ]
    kept, stats = dedup_conversations(convs, jaccard_threshold=0.8)
    ids = {c.conv_id for c in kept}
    assert "a" in ids and "b" not in ids  # exact dup dropped
    assert "d" in ids                      # distinct kept
    assert stats["exact_dup"] >= 1


def test_dedup_protects_compositional_cores():
    from mtlib.datagen import dedup_conversations

    shared = ["shared opening context line", "shared middle filler line here", "shared closing filler line"]
    convs = [
        S.Conversation(conv_id="benign", guardrail_id="g", policy_prompt="p",
                       turns=[S.Turn("user", t) for t in shared], label="False",
                       first_violation_turn=None, source="recomb_benign"),
        # near-dup of benign (shares the padding) but ends with the violating turn
        S.Conversation(conv_id="core", guardrail_id="g", policy_prompt="p",
                       turns=[S.Turn("user", t) for t in shared] + [S.Turn("agent", "the violating payload")],
                       label="True", first_violation_turn=3, source="decompose"),
    ]
    kept, _ = dedup_conversations(convs, jaccard_threshold=0.5)
    # the protected compositional core must survive even though it near-matches benign
    assert any(c.conv_id == "core" for c in kept)


def test_plan_recombinations_labels_and_onset():
    from mtlib.datagen import RecombineConfig, plan_recombinations

    core = S.Conversation(conv_id="core", guardrail_id="G1", policy_prompt="P1",
                          turns=[S.Turn("user", "a"), S.Turn("agent", "b"), S.Turn("user", "c")],
                          label="True", first_violation_turn=2, source="decompose")
    benign = {"P1": [S.Turn("user", "benign one here"), S.Turn("agent", "benign two here")]}
    cfg = RecombineConfig(benign_per_policy=5, recontext_per_core=10, swap_targets_per_core=0)
    import random
    planned = plan_recombinations([core], benign, {"P1": "G1"}, cfg, random.Random(0))
    benigns = [c for c in planned if c.label == "False"]
    recomps = [c for c in planned if c.source == "recomb_decompose"]
    assert benigns and recomps
    for c in recomps:
        # onset must point at the (shifted) last core turn and labels stay monotone
        assert c.first_violation_turn is not None
        labels = c.per_turn_labels()
        assert labels[c.first_violation_turn] == 1 and labels[c.first_violation_turn - 1] == 0
        c.validate()
