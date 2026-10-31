"""Fine-tune the shared encoder and all 12 heads.

Run: python -m mailroom_triage.train [--smoke]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from .config import ensure_dirs, load_config
from .data import (
    EncodedDataset,
    HeadBatchSampler,
    augment_line_order,
    collate,
    encode_examples,
    fingerprint,
    load_examples,
    make_examples,
    read_split,
)
from .heads import build_registry, save_registry
from .model import autocast_ctx, build_model, export_model


def soft_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -(targets * F.log_softmax(logits, dim=-1)).sum(dim=1).mean()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def summarize(logits_by_head: dict[str, list[torch.Tensor]], targets_by_head: dict[str, list[torch.Tensor]]) -> dict:
    result = {}
    total_rows = 0
    total_loss = 0.0
    for key in sorted(logits_by_head):
        logits = torch.cat(logits_by_head[key]).float()
        targets = torch.cat(targets_by_head[key]).float()
        loss = float(soft_ce(logits, targets))
        probs = torch.softmax(logits, dim=-1).numpy()
        truth = targets.numpy()
        rows = len(truth)
        total_rows += rows
        total_loss += loss * rows
        entry = {"rows": rows, "loss": round(loss, 5), "accuracy": round(float((probs.argmax(1) == truth.argmax(1)).mean()), 5)}
        if probs.shape[1] == 2:
            y = truth[:, 1]
            if len(np.unique(y)) == 2:
                entry["auroc"] = round(float(roc_auc_score(y, probs[:, 1])), 5)
        result[key] = entry
    result["_weighted"] = {"rows": total_rows, "loss": round(total_loss / max(1, total_rows), 5)}
    return result


@torch.no_grad()
def evaluate(model, loader, device, max_steps: int | None = None, precision: str = "fp32") -> dict:
    model.eval()
    logits_by_head: dict[str, list[torch.Tensor]] = defaultdict(list)
    targets_by_head: dict[str, list[torch.Tensor]] = defaultdict(list)
    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention = batch["attention_mask"].to(device)
        targets = batch["targets"].to(device)
        with autocast_ctx(device, precision):
            logits = model(input_ids, attention, batch["head_key"])
        logits_by_head[batch["head_key"]].append(logits.detach().float().cpu())
        targets_by_head[batch["head_key"]].append(targets.detach().float().cpu())
        if max_steps and step + 1 >= max_steps:
            break
    return summarize(logits_by_head, targets_by_head)


def train_epoch(model, loader, device, optimizer, scaler, scheduler, max_steps: int | None, precision: str = "fp32") -> dict:
    model.train()
    logits_by_head: dict[str, list[torch.Tensor]] = defaultdict(list)
    targets_by_head: dict[str, list[torch.Tensor]] = defaultdict(list)
    running = 0.0
    started = time.time()
    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention = batch["attention_mask"].to(device, non_blocking=True)
        targets = batch["targets"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx(device, precision):
            logits = model(input_ids, attention, batch["head_key"])
            loss = soft_ce(logits, targets)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        running += float(loss.detach())
        logits_by_head[batch["head_key"]].append(logits.detach().float().cpu())
        targets_by_head[batch["head_key"]].append(targets.detach().float().cpu())
        if step % 200 == 0:
            print(f"  step {step}/{len(loader)} loss={float(loss.detach()):.4f} {(time.time()-started):.0f}s", flush=True)
        if max_steps and step + 1 >= max_steps:
            break
    summary = summarize(logits_by_head, targets_by_head)
    summary["_steps"] = step + 1
    summary["_seconds"] = round(time.time() - started, 1)
    summary["_mean_batch_loss"] = round(running / (step + 1), 5)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    if args.epochs:
        cfg["epochs"] = args.epochs
    if args.batch_size:
        cfg["batch_size"] = args.batch_size
    if args.smoke:
        cfg["epochs"] = 1
        args.max_steps = args.max_steps or 30
    set_seed(cfg["seed"])

    out_dir = Path(args.out) if args.out else Path(cfg["artifacts_dir"]) / "model"
    run_dir = Path(cfg["artifacts_dir"]) / "train"
    run_dir.mkdir(parents=True, exist_ok=True)

    print("loading train rows", flush=True)
    train_rows = list(read_split(cfg["data_root"], "train"))
    registry = build_registry(train_rows)
    save_registry(registry, Path(cfg["artifacts_dir"]) / "registry.json")
    train_examples = augment_line_order(make_examples(train_rows, registry, "train"), cfg["seed"])
    val_examples = load_examples(cfg["data_root"], registry, "validation")
    print(f"train rows={len(train_examples)} val rows={len(val_examples)} heads={len(registry)}", flush=True)

    model, tokenizer = build_model(cfg["model_name"], registry)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"device={device} params={sum(p.numel() for p in model.parameters())/1e6:.1f}M", flush=True)

    train_encoded = encode_examples(train_examples, tokenizer, cfg["max_length"])
    val_encoded = encode_examples(val_examples, tokenizer, cfg["max_length"])
    train_dataset = EncodedDataset(train_encoded)
    val_dataset = EncodedDataset(val_encoded)
    collate_fn = partial(collate, pad_token_id=tokenizer.pad_token_id)
    val_sampler = HeadBatchSampler(val_encoded, cfg["eval_batch_size"], shuffle=False)
    val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, collate_fn=collate_fn, num_workers=0)

    head_params = list(model.heads.parameters())
    encoder_params = [p for name, p in model.named_parameters() if not name.startswith("heads.")]
    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": cfg["encoder_lr"]},
            {"params": head_params, "lr": cfg["head_lr"]},
        ],
        weight_decay=cfg["weight_decay"],
    )
    steps_per_epoch = min(len(HeadBatchSampler(train_encoded, cfg["batch_size"], shuffle=True)), args.max_steps or 10**9)
    total_steps = max(1, steps_per_epoch * cfg["epochs"])
    warmup = int(total_steps * cfg["warmup_ratio"])

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and cfg["precision"] == "fp16")

    best_loss = float("inf")
    best_epoch = -1
    stale = 0
    history = []
    best_path = run_dir / "best.pt"

    for epoch in range(cfg["epochs"]):
        train_sampler = HeadBatchSampler(train_encoded, cfg["batch_size"], shuffle=True, seed=cfg["seed"] + epoch)
        train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn, num_workers=0)
        print(f"epoch {epoch+1}/{cfg['epochs']}: {len(train_loader)} batches", flush=True)
        train_metrics = train_epoch(model, train_loader, device, optimizer, scaler, scheduler, args.max_steps, cfg["precision"])
        val_metrics = evaluate(model, val_loader, device, max_steps=args.max_steps, precision=cfg["precision"])
        entry = {"epoch": epoch + 1, "train": train_metrics, "validation": val_metrics}
        history.append(entry)
        val_loss = val_metrics["_weighted"]["loss"]
        print(f"epoch {epoch+1}: train={train_metrics['_weighted']['loss']:.4f} val={val_loss:.4f}", flush=True)
        if val_loss < best_loss - 1e-4:
            best_loss = val_loss
            best_epoch = epoch + 1
            stale = 0
            torch.save({"model": model.state_dict(), "epoch": best_epoch, "val_loss": best_loss}, best_path)
        else:
            stale += 1
        if stale >= 2:
            print("early stop", flush=True)
            break

    checkpoint = torch.load(best_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model"])
    export_model(model, tokenizer, out_dir, registry, cfg)

    summary = {
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "rows": {"train": len(train_examples), "validation": len(val_examples)},
        "heads": len(registry),
        "device": str(device),
        "params_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        "max_length": cfg["max_length"],
        "batch_size": cfg["batch_size"],
        "fingerprint": fingerprint(cfg["data_root"]),
        "history": history,
    }
    (run_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"exported to {out_dir}; best epoch {best_epoch} val loss {best_loss:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
