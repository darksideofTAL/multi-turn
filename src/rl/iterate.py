"""Orchestrator for one RL iteration.

Each step runs as its own subprocess so the model one stage loads frees GPU
memory before the next loads its own. Also handles cross-iteration data
accumulation and the iter0 seed training set.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from generation.policies import load_policy_files

from src.rl.config import RLConfig, iteration_paths, load_rl_config
from src.rl.utils import concat_jsonl, write_jsonl


logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]


def materialize_seed_training(cfg: RLConfig) -> Path:
    """Flatten non-held-out policy YAMLs into the seed set S^(0): violation_examples
    -> label True, safe_examples -> label False. Held-out policies are excluded."""
    target = Path(cfg.seed_training_path)
    if not target.is_absolute():
        target = REPO_ROOT / target
    target.parent.mkdir(parents=True, exist_ok=True)

    if target.exists():
        logger.info("seed training data already exists: %s (skipping materialize)", target)
        return target

    yaml_paths = sorted(str(p) for p in cfg.resolved_policies_dir().glob("*.yaml"))
    policies = load_policy_files(yaml_paths, Path("."))
    held_out_ids = set(cfg.held_out_ids)

    rows = []
    for policy in policies:
        gid = policy["guardrail_id"]
        if gid in held_out_ids:
            continue
        for label, key in (("True", "violation_examples"), ("False", "safe_examples")):
            for example in policy[key]:
                rows.append(
                    {
                        "guardrail_id": gid,
                        "policy_prompt": policy["verbatim_excerpt"],
                        "input_block": example.strip(),
                        "label": label,
                    }
                )

    n = write_jsonl(rows, target)
    logger.info("Wrote %d seed training rows to %s", n, target)
    return target


def _base_env(cuda_devices: str | None = None) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = "src" + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if cuda_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_devices
    return env


def _run_module(module: str, *extra_args: str, cuda_devices: str | None = None) -> None:
    """Spawn ``python -m <module>`` (PYTHONPATH=src). ``cuda_devices`` pins this
    step's GPUs; None inherits the ambient env (generate/verify pin their own workers)."""
    env = _base_env(cuda_devices)
    cmd = [sys.executable, "-m", module, *extra_args]
    logger.info("$ %s%s", f"CUDA_VISIBLE_DEVICES={cuda_devices} " if cuda_devices else "", " ".join(cmd))
    subprocess.run(cmd, env=env, check=True, cwd=REPO_ROOT)


def _run_torchrun(module: str, gpus: list[int], *extra_args: str) -> None:
    """Launch ``python -m <module>`` under torchrun (one process per GPU) so HF
    Trainer does DDP. A single GPU is a trivial 1-process launch."""
    n = len(gpus) or 1
    env = _base_env(",".join(str(g) for g in gpus))
    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        f"--nproc_per_node={n}", "--master_port=29500",
        "-m", module, *extra_args,
    ]
    logger.info("$ CUDA_VISIBLE_DEVICES=%s %s", env["CUDA_VISIBLE_DEVICES"], " ".join(cmd))
    subprocess.run(cmd, env=env, check=True, cwd=REPO_ROOT)


def accumulate_training_data(cfg: RLConfig, t: int) -> Path:
    """Concatenate S^(0) ∪ ⋃_{i ≤ t} mis_increment[i] into iter{t}/training.jsonl."""
    paths = iteration_paths(cfg, t)
    paths.iter_dir.mkdir(parents=True, exist_ok=True)

    seed_path = Path(cfg.seed_training_path)
    if not seed_path.is_absolute():
        seed_path = REPO_ROOT / seed_path
    mis_paths = [iteration_paths(cfg, i).mis_increment for i in range(t + 1)]
    n = concat_jsonl(paths.training, seed_path, *mis_paths)
    logger.info("iter%d: training.jsonl = %d rows (seed + iter0..%d mis)", t, n, t)
    return paths.training


def iteration_complete(cfg: RLConfig, t: int) -> bool:
    """True iff iter ``t`` finished: both classifier and generator (last artifact)
    are saved. A half-finished iteration fails this and is redone wholesale."""
    paths = iteration_paths(cfg, t)
    return (paths.classifier_dir / "config.json").exists() and (
        paths.generator_dir / "config.json"
    ).exists()


def first_unfinished_iteration(cfg: RLConfig, n: int) -> int:
    """Largest contiguous completed prefix under cfg.output_root => resume point."""
    t = 0
    while t < n and iteration_complete(cfg, t):
        t += 1
    return t


def run_iteration(cfg: RLConfig, t: int) -> None:
    paths = iteration_paths(cfg, t)
    paths.iter_dir.mkdir(parents=True, exist_ok=True)

    # generate/verify run unpinned; they shard across cfg.gen_gpus() via per-GPU workers.
    logger.info("=== iter%d: 1/6 generate candidates (GPUs %s) ===", t, cfg.gen_gpus())
    _run_module("src.rl.steps", "generate", "--iter", str(t))

    logger.info("=== iter%d: 2/6 verify (GPUs %s) ===", t, cfg.gen_gpus())
    _run_module("src.rl.steps", "verify", "--iter", str(t))

    # classify needs one GPU; pin to the first generation GPU.
    logger.info("=== iter%d: 3/6 classify ===", t)
    _run_module("src.rl.steps", "classify", "--iter", str(t), cuda_devices=str(cfg.gen_gpus()[0]))

    logger.info("=== iter%d: 4/6 build preference + mis_increment ===", t)
    _run_module("src.rl.steps", "preference", "--iter", str(t))

    logger.info("=== iter%d: 5/6 accumulate training data ===", t)
    accumulate_training_data(cfg, t)

    logger.info("=== iter%d: 6a/6 train classifier from TALMONITOR ckpt (DDP GPUs %s) ===", t, cfg.train_gpus())
    _run_torchrun("src.rl.train", cfg.train_gpus(), "classifier", "--iter", str(t))

    logger.info("=== iter%d: 6b/6 train generator with DPO (DDP GPUs %s) ===", t, cfg.train_gpus())
    _run_torchrun("src.rl.train", cfg.train_gpus(), "generator", "--iter", str(t))

    logger.info("=== iter%d complete; artifacts under %s ===", t, paths.iter_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rl-config", default="config/train_rl/rl.yaml")
    parser.add_argument("--iter", type=int, default=None, dest="iteration",
                        help="Run only this single iteration (debugging). If omitted, runs the "
                             "resumable loop to cfg.num_iterations, continuing from the first unfinished iter.")
    parser.add_argument("--restart", action="store_true",
                        help="Ignore existing iterations and run from iter0 (overwrites). Default resumes.")
    parser.add_argument("--skip-bootstrap", action="store_true",
                        help="Skip the S^(0) materialization step (assumes seed_training_path already exists).")
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()
    cfg = load_rl_config(args.rl_config)

    # GPUs are assigned per-step (see run_iteration), so no global pin here.
    logger.info("GPU plan: generation/verification=%s, training(DDP)=%s", cfg.gen_gpus(), cfg.train_gpus())

    if not args.skip_bootstrap:
        materialize_seed_training(cfg)

    # Single-iteration debug mode.
    if args.iteration is not None:
        run_iteration(cfg, args.iteration)
        return

    # Resumable loop: continue from the first unfinished iteration up to num_iterations.
    n = cfg.num_iterations
    start = 0 if args.restart else first_unfinished_iteration(cfg, n)
    if start >= n:
        logger.info("All %d iterations already complete under %s; nothing to do.", n, cfg.resolved_output_root())
        return
    if start > 0:
        logger.info("Resuming: iter0..%d already complete under %s -> running iter%d..%d.",
                    start - 1, cfg.resolved_output_root(), start, n - 1)
    else:
        logger.info("Running iter0..%d under %s.", n - 1, cfg.resolved_output_root())
    for t in range(start, n):
        run_iteration(cfg, t)


if __name__ == "__main__":
    main()
