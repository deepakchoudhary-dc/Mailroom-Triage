"""Refit individual heads on a frozen encoder.

Fine-tuning gives every head one pass over its rows. A head with 256 training
rows therefore sees eight gradient steps and underfits, while common heads see
hundreds. This freezes the (already fine-tuned) encoder, embeds only the rows
of the requested heads, and trains those heads to convergence, selecting the
checkpoint on the calibration split.

Run: python -m mailroom_triage.refit --model-dir artifacts/model-ft --heads category#7
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from functools import partial
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import ensure_dirs, load_config
from .data import EncodedDataset, HeadBatchSampler, collate, encode_examples, load_examples, make_examples, read_split
from .model import DecisionModel, autocast_ctx, export_model, load_exported


def soft_ce(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return -(targets * F.log_softmax(logits, dim=-1)).sum(dim=1).mean()


@torch.no_grad()
def embed_examples(model, tokenizer, examples, cfg, device) -> dict[str, dict]:
    encoded = encode_examples(examples, tokenizer, cfg["max_length"])
    dataset = EncodedDataset(encoded)
    sampler = HeadBatchSampler(encoded, cfg["eval_batch_size"], shuffle=False)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=partial(collate, pad_token_id=tokenizer.pad_token_id), num_workers=0)
    out: dict[str, dict] = defaultdict(lambda: {"emb": [], "targets": []})
    model.eval()
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention = batch["attention_mask"].to(device)
        with autocast_ctx(device, cfg["precision"]):
            pooled = model.encode(input_ids, attention)
        out[batch["head_key"]]["emb"].append(pooled.float().cpu())
        out[batch["head_key"]]["targets"].append(batch["targets"].float())
    return {key: {"emb": torch.cat(value["emb"]), "targets": torch.cat(value["targets"])} for key, value in out.items()}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--heads", nargs="+", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out) if args.out else model_dir
    model, tokenizer, registry = load_exported(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for head in args.heads:
        if head not in registry:
            raise SystemExit(f"unknown head {head!r}")

    wanted = {registry[head].question_id for head in args.heads}
    train_rows = [row for row in read_split(cfg["data_root"], "train") if row["metadata"]["question_id"] in wanted]
    train_examples = make_examples(train_rows, registry, "train")
    calibration_examples = [example for example in load_examples(cfg["data_root"], registry, "calibration") if example.head_key in args.heads]
    train = embed_examples(model, tokenizer, train_examples, cfg, device)
    calibration = embed_examples(model, tokenizer, calibration_examples, cfg, device)

    for head_key in args.heads:
        train_x = train[head_key]["emb"].to(device)
        train_y = train[head_key]["targets"].to(device)
        cal_x = calibration[head_key]["emb"].to(device)
        cal_y = calibration[head_key]["targets"].to(device)
        head = model.heads[head_key]
        optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        best = (float("inf"), {name: tensor.detach().cpu().clone() for name, tensor in head.state_dict().items()})
        for step in range(1, args.steps + 1):
            head.train()
            optimizer.zero_grad(set_to_none=True)
            loss = soft_ce(head(train_x), train_y)
            loss.backward()
            optimizer.step()
            for group in optimizer.param_groups:
                group["lr"] = args.lr * 0.5 * (1.0 + math.cos(math.pi * step / args.steps))
            if step % 10 == 0 or step == 1:
                head.eval()
                with torch.no_grad():
                    cal_loss = float(soft_ce(head(cal_x), cal_y))
                    accuracy = float((head(cal_x).argmax(dim=1) == cal_y.argmax(dim=1)).float().mean())
                if cal_loss < best[0] - 1e-6:
                    best = (cal_loss, {name: tensor.detach().cpu().clone() for name, tensor in head.state_dict().items()})
                if step % 100 == 0:
                    print(f"{head_key} step {step}: train={float(loss):.4f} cal={cal_loss:.4f} acc={accuracy:.3f}", flush=True)
        head.load_state_dict(best[1])
        head.eval()
        with torch.no_grad():
            accuracy = float((head(cal_x).argmax(dim=1) == cal_y.argmax(dim=1)).float().mean())
        print(f"{head_key}: best cal loss {best[0]:.4f}, accuracy {accuracy:.4f}, rows {len(train_x)}", flush=True)

    export_model(model, tokenizer, out_dir, registry, cfg)
    summary = {"model_dir": str(model_dir), "heads": args.heads, "steps": args.steps, "lr": args.lr, "rows": {key: int(len(train[key]["emb"])) for key in args.heads}}
    (out_dir / "refit.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"saved heads to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
