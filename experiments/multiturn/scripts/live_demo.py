#!/usr/bin/env python
"""Phase D5: drive the streaming monitor turn-by-turn on one conversation.

Feeds a conversation (from a dataset row or a --policy/--transcript pair) one
turn at a time and prints, per turn, the single-turn verdict and the updated
conversation-so-far verdict. Demonstrates the O(1)-per-turn streaming path and
that streaming == one-shot.

Usage
  python scripts/live_demo.py --data outputs/data --split test --agg outputs/agg_r1/aggregator.pt --index 0
  python scripts/live_demo.py --agg outputs/agg_r1/aggregator.pt \
      --policy "..." --transcript "<User>..</User>\n<Agent>..</Agent>"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from mtlib.aggregator import ConversationAggregator  # noqa: E402
from mtlib.encoder import TurnEncoder  # noqa: E402
from mtlib.monitor import MultiTurnMonitor  # noqa: E402
from mtlib.schema import Conversation, read_conversations, split_transcript  # noqa: E402


def load_one(args) -> Conversation:
    if args.data:
        convs = list(read_conversations(Path(args.data) / f"{args.split}.jsonl"))
        return convs[args.index]
    turns = split_transcript(args.transcript.replace("\\n", "\n"))
    return Conversation(
        conv_id="adhoc", guardrail_id="adhoc", policy_prompt=args.policy,
        turns=turns, label="False",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--agg", required=True)
    ap.add_argument("--encoder-ckpt", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--policy", default=None)
    ap.add_argument("--transcript", default=None)
    args = ap.parse_args()

    encoder = TurnEncoder(model_dir=args.encoder_ckpt or None)
    model, extra = ConversationAggregator.load(args.agg)
    monitor = MultiTurnMonitor(encoder, model, tau=extra.get("tau", 0.5))

    conv = load_one(args)
    print(f"policy: {conv.policy_prompt[:100]}...")
    print(f"label={conv.label} onset={conv.first_violation_turn} tau={monitor.tau:.4f}\n")

    conv_id = monitor.start(conv.policy_prompt)
    for i, turn in enumerate(conv.turns):
        result = monitor.feed(conv_id, turn.role, turn.text)
        flag = "  <== FLAG" if result["flagged"] else ""
        print(
            f"turn {i:2d} [{turn.role:5s}] turn_p={result['turn_violation_prob']:.3f} "
            f"conv_p={result['conversation_violation_prob']:.3f}{flag}  {turn.text[:60]}"
        )
    final = monitor.finish(conv_id)
    print(f"\nfinal: conv_p={final['conversation_violation_prob']:.3f} flagged={final['flagged']}")

    # Cross-check streaming == one-shot.
    oneshot = monitor.score_conversation(conv.policy_prompt, conv.turns)
    max_abs = max(
        abs(a - b)
        for a, b in zip(final["per_turn_conversation_probs"], oneshot["per_turn_conversation_probs"])
    )
    print(f"streaming vs one-shot max |Δ| = {max_abs:.2e}")


if __name__ == "__main__":
    main()
