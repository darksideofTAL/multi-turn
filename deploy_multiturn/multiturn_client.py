#!/usr/bin/env python
"""Synchronous client for the talmonitor_multiturn Triton service.

Feeds a <User>/<Agent>-tagged transcript turn by turn and prints the
single-turn and conversation-so-far verdicts, then the final verdict.

Usage:
  echo "<User>hi</User>\n<Agent>hello</Agent>" | \
      python deploy_multiturn/multiturn_client.py --policy "..." [--url http://localhost:8700]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request

TURN_RE = re.compile(r"<(User|Agent)>(.*?)</\1>", re.DOTALL)


def infer(url: str, op: str, conv_id: str = "", role: str = "", text: str = "") -> dict:
    body = {
        "inputs": [
            {"name": "op", "shape": [1], "datatype": "BYTES", "data": [op]},
            {"name": "conv_id", "shape": [1], "datatype": "BYTES", "data": [conv_id]},
            {"name": "role", "shape": [1], "datatype": "BYTES", "data": [role]},
            {"name": "text", "shape": [1], "datatype": "BYTES", "data": [text]},
        ]
    }
    req = urllib.request.Request(
        f"{url}/v2/models/talmonitor_multiturn/infer",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req) as resp:
        payload = json.load(resp)
    return json.loads(payload["outputs"][0]["data"][0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8700")
    ap.add_argument("--policy", required=True)
    ap.add_argument("--transcript", default=None, help="file path; default stdin")
    args = ap.parse_args()

    raw = open(args.transcript).read() if args.transcript else sys.stdin.read()
    turns = [(tag.lower(), text.strip()) for tag, text in TURN_RE.findall(raw)]
    if not turns:
        sys.exit("no <User>/<Agent> turns found in transcript")

    conv_id = infer(args.url, "start", text=args.policy)["conv_id"]
    for role, text in turns:
        r = infer(args.url, "feed", conv_id=conv_id, role=role, text=text)
        if "error" in r:
            sys.exit(f"feed error: {r['error']}")
        flag = "  <== FLAG" if r["flagged"] else ""
        print(
            f"turn {r['turn_index']:2d} [{role:5s}] turn_p={r['turn_violation_prob']:.3f} "
            f"conv_p={r['conversation_violation_prob']:.3f}{flag}  {text[:60]!r}"
        )
    final = infer(args.url, "finish", conv_id=conv_id)
    print(
        f"\nfinal: conv_p={final['conversation_violation_prob']:.3f} "
        f"flagged={final['flagged']} tau={final['tau']:.4f} n_turns={final['n_turns']}"
    )


if __name__ == "__main__":
    main()
