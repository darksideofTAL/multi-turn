#!/usr/bin/env python
"""Phase D3: train the conversation aggregator on precomputed latents.

Backbone stays frozen; only proj + role embedding + transformer + head train.
Minutes on one GPU over a tensor dataset. The single-turn deploy path is never
touched, so this cannot regress the shipped classifier.

Loss = per-turn CE (monotone conversation-status labels), first-violation turn
upweighted. tau is stamped into the checkpoint from val benign conversations at
a target per-conversation FPR.

Usage
  python scripts/train_aggregator.py \
      --train-latents /raid/.../latents/train --val-latents /raid/.../latents/val \
      --out outputs/agg_r1 --epochs 8 --d-model 512 --layers 3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from mtlib.aggregator import AggregatorConfig, ConversationAggregator, per_turn_loss  # noqa: E402
from mtlib.common import dump_json, setup_logging  # noqa: E402
from mtlib.dataset import LatentConversationDataset, collate, shard_paths  # noqa: E402
from mtlib.metrics import conversation_metrics, tau_for_conversation_fpr  # noqa: E402
from mtlib.schema import COMPOSITIONAL_SOURCES  # noqa: E402


@torch.no_grad()
def evaluate(model, loader, device, positive_id: int) -> tuple[dict, list, list]:
    """Conversation-level metrics from the final-turn score of each conversation."""
    model.eval()
    conv_scores, conv_labels = [], []
    benign_per_turn = []
    for batch in loader:
        logits = model(
            turn_latents=batch["turn_latents"].to(device),
            turn_logits=batch["turn_logits"].to(device),
            role_ids=batch["role_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
            policy_latent=batch["policy_latent"].to(device),
        )
        probs = torch.softmax(logits, dim=-1)[..., positive_id].cpu()  # [B, T]
        lengths = batch["attention_mask"].sum(dim=1)
        for i, n in enumerate(lengths.tolist()):
            seq = probs[i, :n]
            conv_scores.append(float(seq.max()))  # conversation flags if any turn flags
            conv_labels.append(int(batch["conv_label"][i]))
            if int(batch["conv_label"][i]) == 0:
                benign_per_turn.append(seq.tolist())
    metrics = conversation_metrics(conv_labels, conv_scores)
    return metrics, benign_per_turn, list(zip(conv_labels, conv_scores))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-latents", required=True)
    ap.add_argument("--val-latents", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--onset-weight", type=float, default=2.0)
    ap.add_argument("--no-policy-token", action="store_true")
    ap.add_argument("--no-logit-features", action="store_true")
    ap.add_argument("--oversample-comp", type=int, default=1,
                    help="repeat factor for compositional-source train conversations")
    ap.add_argument("--bf16", action="store_true", help="bf16 autocast for the blocks (big configs)")
    ap.add_argument("--grad-accum", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--target-fpr", type=float, default=0.02, help="per-conversation benign FPR for tau")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    logger = setup_logging("train_aggregator")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # Oversample the genuinely compositional sources — INCLUDING the recombined
    # variants (recomb_*), which are the scarce signal after latent-bank
    # recombination.
    comp_sources = list(COMPOSITIONAL_SOURCES) + [f"recomb_{s}" for s in COMPOSITIONAL_SOURCES]
    oversample = (
        {s: args.oversample_comp for s in comp_sources} if args.oversample_comp > 1 else None
    )
    train_ds = LatentConversationDataset(shard_paths(args.train_latents), oversample=oversample)
    val_ds = LatentConversationDataset(shard_paths(args.val_latents))
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate, drop_last=False
    )
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    logger.info("train=%d val=%d conversations", len(train_ds), len(val_ds))

    sample = train_ds[0]
    config = AggregatorConfig(
        input_dim=sample["turn_latents"].shape[1],
        num_labels=sample["turn_logits"].shape[1],
        d_model=args.d_model,
        n_layers=args.layers,
        n_heads=args.heads,
        dropout=args.dropout,
        use_policy_token=not args.no_policy_token,
        use_logit_features=not args.no_logit_features,
    )
    positive_id = 1 if config.num_labels == 2 else config.num_labels - 1
    model = ConversationAggregator(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("aggregator: %.1fM params, config=%s", n_params / 1e6, config.to_dict())

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * max(1, len(train_loader))
    warmup = max(1, int(args.warmup_frac * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim,
        lambda s: s / warmup if s < warmup else max(0.0, (steps - s) / max(1, steps - warmup)),
    )

    # bf16 autocast for the attention blocks on big configs; the head + CE loss
    # stay fp32 inside forward, same recipe as the 12B classifier.
    use_amp = args.bf16 and device.type == "cuda"
    accum = max(1, args.grad_accum)

    out_dir = Path(args.out)
    history = []
    best_auroc, best_tau = -1.0, 0.5
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        optim.zero_grad()
        for step, batch in enumerate(train_loader):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model(
                    turn_latents=batch["turn_latents"].to(device),
                    turn_logits=batch["turn_logits"].to(device),
                    role_ids=batch["role_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    policy_latent=batch["policy_latent"].to(device),
                )
                loss = per_turn_loss(
                    logits, batch["labels"].to(device), batch["onset"].to(device), args.onset_weight
                )
            (loss / accum).backward()
            if (step + 1) % accum == 0 or (step + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step()
                sched.step()
                optim.zero_grad()
            running += float(loss.detach())
        avg = running / max(1, len(train_loader))

        metrics, benign_per_turn, _ = evaluate(model, val_loader, device, positive_id)
        tau = (
            tau_for_conversation_fpr(benign_per_turn, args.target_fpr)
            if benign_per_turn else 0.5
        )
        logger.info(
            "epoch %d: loss=%.4f val_auroc=%.4f val_f1@.5=%.4f tau@%.0f%%FPR=%.4f",
            epoch, avg, metrics["auroc"], metrics["f1"], 100 * args.target_fpr, tau,
        )
        history.append({"epoch": epoch, "train_loss": avg, **metrics, "tau": tau})
        if metrics["auroc"] >= best_auroc:
            best_auroc, best_tau = metrics["auroc"], tau
            model.save(out_dir / "aggregator.pt", extra={"tau": tau, "positive_id": positive_id})

    dump_json({"history": history, "best_auroc": best_auroc, "best_tau": best_tau, "args": vars(args)},
              out_dir / "train_summary.json")
    logger.info("done: best val AUROC=%.4f, tau=%.4f -> %s", best_auroc, best_tau, out_dir)


if __name__ == "__main__":
    main()
