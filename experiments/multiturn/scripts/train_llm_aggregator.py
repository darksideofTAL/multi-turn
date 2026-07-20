#!/usr/bin/env python
"""Phase D7 (v3): train the LLM latent aggregator (frozen LM + projector + head).

Reads the SAME cached latent shards as the small aggregator; the policy text is
joined in from the conversations JSONL by conv_id. Only the projector, role
embeddings, and head train — gradients flow through the frozen LM but update
nothing in it.

Usage
  CUDA_VISIBLE_DEVICES=4 python scripts/train_llm_aggregator.py \
      --data outputs/data_v2 --train-latents /raid/.../latents_v2/train \
      --val-latents /raid/.../latents_v2/val --out outputs/llmagg_v3 \
      --lm Qwen/Qwen2.5-1.5B --epochs 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from mtlib.aggregator import per_turn_loss  # noqa: E402
from mtlib.common import dump_json, setup_logging  # noqa: E402
from mtlib.dataset import LatentConversationDataset, collate, shard_paths  # noqa: E402
from mtlib.llm_aggregator import LlmAggregatorConfig, LlmLatentAggregator  # noqa: E402
from mtlib.metrics import conversation_metrics, tau_for_conversation_fpr  # noqa: E402
from mtlib.schema import COMPOSITIONAL_SOURCES, read_conversations  # noqa: E402


def policy_index(data_dir: str) -> dict[str, str]:
    """conv_id -> policy text, across all splits present."""
    index: dict[str, str] = {}
    for split in ("train", "val", "test"):
        path = Path(data_dir) / f"{split}.jsonl"
        if path.exists():
            for conv in read_conversations(path):
                index[conv.conv_id] = conv.policy_prompt
    return index


@torch.no_grad()
def evaluate(model, loader, policies_by_conv, device, positive_id: int):
    model.eval()
    conv_scores, conv_labels, benign_per_turn = [], [], []
    for batch in loader:
        logits = model(
            policies=[policies_by_conv[cid] for cid in batch["conv_ids"]],
            turn_latents=batch["turn_latents"].to(device),
            turn_logits=batch["turn_logits"].to(device),
            role_ids=batch["role_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        probs = torch.softmax(logits.float(), dim=-1)[..., positive_id].cpu()
        for i, n in enumerate(batch["attention_mask"].sum(dim=1).tolist()):
            seq = probs[i, :n]
            conv_scores.append(float(seq.max()))
            conv_labels.append(int(batch["conv_label"][i]))
            if int(batch["conv_label"][i]) == 0:
                benign_per_turn.append(seq.tolist())
    return conversation_metrics(conv_labels, conv_scores), benign_per_turn


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir with {split}.jsonl (policy text source)")
    ap.add_argument("--train-latents", required=True)
    ap.add_argument("--val-latents", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lm", default="Qwen/Qwen3-4B")
    ap.add_argument("--head", choices=("linear", "verbalizer"), default="linear")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--warmup-frac", type=float, default=0.05)
    ap.add_argument("--projector-hidden", type=int, default=2048)
    ap.add_argument("--onset-weight", type=float, default=2.0)
    ap.add_argument("--oversample-comp", type=int, default=1)
    ap.add_argument("--no-logit-features", action="store_true")
    ap.add_argument("--target-fpr", type=float, default=0.02)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    logger = setup_logging("train_llm_aggregator")
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    policies_by_conv = policy_index(args.data)
    comp_sources = list(COMPOSITIONAL_SOURCES) + [f"recomb_{s}" for s in COMPOSITIONAL_SOURCES]
    oversample = (
        {s: args.oversample_comp for s in comp_sources} if args.oversample_comp > 1 else None
    )
    train_ds = LatentConversationDataset(shard_paths(args.train_latents), oversample=oversample)
    val_ds = LatentConversationDataset(shard_paths(args.val_latents))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    logger.info("train=%d val=%d conversations", len(train_ds), len(val_ds))

    sample = train_ds[0]
    config = LlmAggregatorConfig(
        lm_name=args.lm,
        input_dim=sample["turn_latents"].shape[1],
        num_labels=sample["turn_logits"].shape[1],
        head=args.head,
        use_logit_features=not args.no_logit_features,
        projector_hidden=args.projector_hidden,
    )
    positive_id = 1
    model = LlmLatentAggregator(config).to(device)
    trainable = model.trainable_parameters()
    n_train = sum(p.numel() for p in trainable)
    n_frozen = sum(p.numel() for p in model.lm.parameters())
    logger.info("LLM aggregator: %s frozen (%.1fB), %.1fM trainable",
                args.lm, n_frozen / 1e9, n_train / 1e6)

    optim = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    steps = args.epochs * max(1, len(train_loader))
    warmup = max(1, int(args.warmup_frac * steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: s / warmup if s < warmup else max(0.0, (steps - s) / max(1, steps - warmup))
    )

    out_dir = Path(args.out)
    history, best_auroc, best_tau = [], -1.0, 0.5
    for epoch in range(args.epochs):
        model.train()
        model.lm.eval()  # frozen LM stays in eval mode (no dropout drift)
        running = 0.0
        for batch in train_loader:
            logits = model(
                policies=[policies_by_conv[cid] for cid in batch["conv_ids"]],
                turn_latents=batch["turn_latents"].to(device),
                turn_logits=batch["turn_logits"].to(device),
                role_ids=batch["role_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )
            loss = per_turn_loss(
                logits, batch["labels"].to(device), batch["onset"].to(device), args.onset_weight
            )
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optim.step()
            sched.step()
            running += float(loss.detach())
        avg = running / max(1, len(train_loader))

        metrics, benign_per_turn = evaluate(model, val_loader, policies_by_conv, device, positive_id)
        tau = tau_for_conversation_fpr(benign_per_turn, args.target_fpr) if benign_per_turn else 0.5
        logger.info("epoch %d: loss=%.4f val_auroc=%.4f val_f1@.5=%.4f tau@%.0f%%FPR=%.4f",
                    epoch, avg, metrics["auroc"], metrics["f1"], 100 * args.target_fpr, tau)
        history.append({"epoch": epoch, "train_loss": avg, **metrics, "tau": tau})
        if metrics["auroc"] >= best_auroc:
            best_auroc, best_tau = metrics["auroc"], tau
            model.save(out_dir / "llm_aggregator.pt", extra={"tau": tau, "positive_id": positive_id})

    dump_json({"history": history, "best_auroc": best_auroc, "best_tau": best_tau, "args": vars(args)},
              out_dir / "train_summary.json")
    logger.info("done: best val AUROC=%.4f tau=%.4f -> %s", best_auroc, best_tau, out_dir)


if __name__ == "__main__":
    main()
