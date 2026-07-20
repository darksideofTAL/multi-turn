"""Shared constants and utilities for the multi-turn monitor experiment.

Self-contained: imports from src/ are read-only. Big artifacts go to /raid.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path("/home/frontiers_ashoka/BARRED")
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

RAID_OUT = Path("/raid/frontiers_ashoka/BARRED-multiturn")

# Frozen single-turn classifier used as the turn encoder (same checkpoint the
# token-supervision experiment started from). The 24 GB checkpoint lives on
# /raid (root / is full); the /home repo `models/` dir is empty on this box.
CHECKPOINT = Path("/raid/frontiers_ashoka/BARRED/models/google-gemma-3-12b-it-v4-nofreeze-n7500")
SAMPLES_PATH = REPO / "output/generation/policies_v4/accepted_samples.jsonl"

# Mirrors training/eval.py MAX_SEQ_LENGTH and the deploy defaults.
MAX_SEQ_LENGTH = 4096
POSITIVE_LABEL = "True"  # id 1 = policy violation
NEGATIVE_LABEL = "False"


def setup_logging(name: str) -> logging.Logger:
    RAID_OUT.joinpath("logs").mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    log_path = RAID_OUT / "logs" / f"{name}-{stamp}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_path)],
        force=True,
    )
    logger = logging.getLogger(name)
    logger.info("Logging to %s", log_path)
    return logger


def dump_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    return path
