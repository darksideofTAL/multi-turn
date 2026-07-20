"""Triton Python-backend for the multi-turn conversation monitor.

Frozen 12B single-turn classifier (turn encoder) + 15M causal aggregator
(cscale ALL winner — experiments/multiturn/RESULTS.md).

Protocol (result = JSON blob):
  op="start"  text=<policy>              -> {"conv_id"}
  op="feed"   role=<user|agent> text=<turn>
              -> {"turn_index", "turn_violation_prob",
                  "conversation_violation_prob",
                  "per_turn_conversation_probs", "flagged", "tau"}
  op="finish"                            -> {"n_turns", "conversation_violation_prob",
                                             "per_turn_conversation_probs",
                                             "flagged", "tau"}
  op="abort"                             -> {"aborted": bool}

The client must keep a conversation's requests ordered (a synchronous client
does this trivially). Idle conversations are evicted after conv_ttl_seconds.
"""

import json
import os
import sys
import time
import traceback

_SRC = os.environ.get("TALMONITOR_SRC", "/opt/talmonitor/src")
_MTDIR = os.environ.get("TALMONITOR_MULTITURN", "/opt/talmonitor/experiments/multiturn")
for _p in (_SRC, _MTDIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import triton_python_backend_utils as pb_utils

from mtlib.aggregator import ConversationAggregator
from mtlib.encoder import TurnEncoder
from mtlib.monitor import MultiTurnMonitor


def _params(model_config: dict) -> dict:
    return {k: v["string_value"] for k, v in model_config.get("parameters", {}).items()}


def _scalar(req, name: str) -> str:
    value = pb_utils.get_input_tensor_by_name(req, name).as_numpy().reshape(-1)[0]
    return value.decode("utf-8") if isinstance(value, (bytes, bytearray)) else str(value)


class TritonPythonModel:
    def initialize(self, args):
        self.logger = pb_utils.Logger
        p = _params(json.loads(args["model_config"]))
        device = f"cuda:{args.get('model_instance_device_id', '0')}"
        encoder = TurnEncoder(
            model_dir=p["model_dir"],
            device=device,
            max_seq_length=int(p.get("max_seq_length", "4096")),
        )
        aggregator, extra = ConversationAggregator.load(p["agg_path"])
        tau = float(p["tau"]) if p.get("tau") else float(extra.get("tau", 0.5))
        self.monitor = MultiTurnMonitor(encoder, aggregator, tau=tau, device=device)
        self.last_seen: dict[str, float] = {}
        self.ttl = float(p.get("conv_ttl_seconds", "600"))
        self.logger.log_info(
            f"talmonitor_multiturn ready (agg={p['agg_path']}, tau={tau})"
        )

    def _sweep_idle(self) -> None:
        now = time.time()
        for cid, ts in list(self.last_seen.items()):
            if now - ts > self.ttl:
                self.monitor.abort(cid)
                self.last_seen.pop(cid, None)

    def _handle(self, op: str, cid: str, role: str, text: str) -> dict:
        if op == "start":
            new_cid = self.monitor.start(text, cid or None)
            self.last_seen[new_cid] = time.time()
            return {"conv_id": new_cid}
        if op == "feed":
            self.last_seen[cid] = time.time()
            return self.monitor.feed(cid, role, text)
        if op == "finish":
            self.last_seen.pop(cid, None)
            return self.monitor.finish(cid)
        if op == "abort":
            self.last_seen.pop(cid, None)
            return {"aborted": self.monitor.abort(cid)}
        return {"error": f"unknown op {op!r}"}

    def execute(self, requests):
        self._sweep_idle()
        responses = []
        for req in requests:
            try:
                payload = self._handle(
                    _scalar(req, "op"),
                    _scalar(req, "conv_id"),
                    _scalar(req, "role"),
                    _scalar(req, "text"),
                )
            except Exception as err:  # noqa: BLE001
                self.logger.log_error(
                    f"talmonitor_multiturn: {err}\n{traceback.format_exc()}")
                payload = {"error": str(err)}
            responses.append(pb_utils.InferenceResponse(output_tensors=[
                pb_utils.Tensor("result", np.array([json.dumps(payload)], dtype=object))
            ]))
        return responses

    def finalize(self):
        self.monitor = None
