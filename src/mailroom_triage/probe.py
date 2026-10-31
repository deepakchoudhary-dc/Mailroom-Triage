"""Frozen-encoder linear probe.

The corpus is explicit-evidence synthetic data: TF-IDF already saturates it.
A linear probe on frozen MiniLM embeddings establishes the ceiling cheaply and
leaves encoder fine-tuning as an option for real inbox data, where the text is
noisier. The exported artifact is still encoder + 12 trained heads.

Run: python -m mailroom_triage.probe
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from .config import ensure_dirs, load_config
from .data import (
    EncodedDataset,
    HeadBatchSampler,
    collate,
    encode_examples,
    fingerprint,
    load_examples,
    make_examples,
    read_split,
)
from .heads import build_registry, save_registry
from .model import DecisionModel, autocast_ctx, build_model, export_model


def cache_file(cfg: dict, split: str) -> Path:
    tag = cfg["model_name"].replace("/", "_").replace("-", "_")
    return Path(cfg["artifacts_dir"]) / "cache" / f"{split}_{tag}_{cfg['max_length']}.pt"


@torch.no_grad()
def embed(model, tokenizer, examples, cfg, device) -> dict:
    encoded = encode_examples(examples, tokenizer, cfg["max_length"])
    dataset = EncodedDataset(encoded)
    sampler = HeadBatchSampler(encoded, cfg["eval_batch_size"], shuffle=False)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=partial(collate, pad_token_id=tokenizer.pad_token_id), num_workers=0)
    embeddings, targets, keys, meta = [], [], [], []
    model.eval()
    for step, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention = batch["attention_mask"].to(device)
        with autocast_ctx(device, cfg["precision"]):
            pooled = model.encode(input_ids, attention)
        embeddings.append(pooled.float().cpu())
        targets.extend(batch["targets"].tolist())
        keys.extend([batch["head_key"]] * len(batch["targets"]))
        meta.extend(batch["meta"])
        if step % 100 == 0:
            used = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
            print(f"  embed {step}/{len(loader)} gpu_mem={used:.2f}GB", flush=True)
    return {"emb": torch.cat(embeddings), "targets": targets, "keys": keys, "meta": meta}


def embed_cached(model, tokenizer, examples, cfg, device, split: str, recompute: bool) -> dict:
    path = cache_file(cfg, split)
    if path.exists() and not recompute:
        print(f"loading cached embeddings {path}", flush=True)
        return torch.load(path, map_location="cpu", weights_only=False)
    data = embed(model, tokenizer, examples, cfg, device)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(data, path)
    print(f"cached embeddings {path}", flush=True)
    return data


def soft_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -(targets * F.log_softmax(logits, dim=-1)).sum(dim=1).mean()


def targets_by_head(rows: list[list[float]], keys: list[str]) -> dict[str, torch.Tensor]:
    grouped: dict[str, list[list[float]]] = defaultdict(list)
    for row, key in zip(rows, keys):
        grouped[key].append(row)
    return {key: torch.tensor(value, dtype=torch.float32) for key, value in grouped.items()}


def evaluate_head(head: nn.Linear, x: torch.Tensor, y: torch.Tensor) -> dict:
    with torch.no_grad():
        logits = head(x)
        loss = float(soft_ce(logits, y))
        probs = torch.softmax(logits, dim=-1)
        truth = y.argmax(dim=1)
        accuracy = float((probs.argmax(dim=1) == truth).float().mean())
        result = {"loss": round(loss, 5), "accuracy": round(accuracy, 5)}
        if probs.shape[1] == 2 and len(torch.unique(y[:, 1])) == 2:
            result["auroc"] = round(float(roc_auc_score(y[:, 1].cpu().numpy(), probs[:, 1].cpu().numpy())), 5)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--recompute", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    torch.manual_seed(cfg["seed"])

    print("loading rows", flush=True)
    train_rows = list(read_split(cfg["data_root"], "train"))
    registry = build_registry(train_rows)
    save_registry(registry, Path(cfg["artifacts_dir"]) / "registry.json")
    train_examples = make_examples(train_rows, registry, "train")
    val_examples = load_examples(cfg["data_root"], registry, "validation")

    model, tokenizer = build_model(cfg["model_name"], registry)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"device={device} gpu={torch.cuda.get_device_name(0) if device.type == 'cuda' else 'none'}", flush=True)

    print("embedding train", flush=True)
    train = embed_cached(model, tokenizer, train_examples, cfg, device, "train", args.recompute)
    print("embedding validation", flush=True)
    val = embed_cached(model, tokenizer, val_examples, cfg, device, "validation", args.recompute)

    heads = nn.ModuleDict({key: nn.Linear(model.encoder.config.hidden_size, len(spec.options)) for key, spec in registry.items()}).to(device)
    optimizers = {key: torch.optim.AdamW(heads[key].parameters(), lr=1e-2, weight_decay=1e-4) for key in registry}
    train_x = train["emb"].to(device)
    train_y = {key: value.to(device) for key, value in targets_by_head(train["targets"], train["keys"]).items()}
    val_x = val["emb"].to(device)
    val_y = {key: value.to(device) for key, value in targets_by_head(val["targets"], val["keys"]).items()}
    train_mask = {key: torch.tensor([k == key for k in train["keys"]], dtype=torch.bool, device=device) for key in registry}
    val_mask = {key: torch.tensor([k == key for k in val["keys"]], dtype=torch.bool, device=device) for key in registry}
    train_head = {key: train_x[train_mask[key]] for key in registry}
    val_head = {key: val_x[val_mask[key]] for key in registry}
    best: dict[str, tuple[float, dict]] = {}
    stale = defaultdict(int)
    history = []

    for step in range(1, args.steps + 1):
        total = 0.0
        for key in registry:
            x = train_head[key]
            y = train_y[key]
            optimizers[key].zero_grad(set_to_none=True)
            loss = soft_ce(heads[key](x), y)
            loss.backward()
            optimizers[key].step()
            total += float(loss)
        for optimizer in optimizers.values():
            for group in optimizer.param_groups:
                group["lr"] = 1e-2 * 0.5 * (1.0 + math.cos(math.pi * step / args.steps))
        if step % 25 == 0 or step == 1:
            epoch_metrics = {}
            for key in registry:
                epoch_metrics[key] = evaluate_head(heads[key], val_head[key], val_y[key])
                loss = epoch_metrics[key]["loss"]
                if key not in best or loss < best[key][0] - 1e-5:
                    best[key] = (loss, {name: tensor.detach().cpu().clone() for name, tensor in heads[key].state_dict().items()})
                    stale[key] = 0
                else:
                    stale[key] += 1
            mean_loss = float(np.mean([value["loss"] for value in epoch_metrics.values()]))
            history.append({"step": step, "train_loss": round(total / len(registry), 5), "val_loss": round(mean_loss, 5)})
            print(f"step {step}: train={total/len(registry):.4f} val={mean_loss:.4f}", flush=True)
            if all(count >= 6 for count in stale.values()):
                break

    for key in registry:
        heads[key].load_state_dict(best[key][1])

    # Evaluate the reloaded best heads on validation.
    validation = {}
    for key in registry:
        validation[key] = evaluate_head(heads[key], val_head[key], val_y[key])
        print(f"{key:24s} val_loss={validation[key]['loss']:.4f} acc={validation[key]['accuracy']:.4f}")

    model.heads.load_state_dict(heads.state_dict())
    out_dir = Path(cfg["artifacts_dir"]) / "model"
    export_model(model, tokenizer, out_dir, registry, cfg)

    summary = {
        "method": "frozen_encoder_linear_probe",
        "encoder": cfg["model_name"],
        "max_length": cfg["max_length"],
        "rows": {"train": len(train_examples), "validation": len(val_examples)},
        "heads": len(registry),
        "params_millions": round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
        "fingerprint": fingerprint(cfg["data_root"]),
        "validation": validation,
        "history": history,
    }
    (Path(cfg["artifacts_dir"]) / "train" / "probe_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"exported to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

