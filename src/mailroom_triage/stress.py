"""Robustness under realistic mail formatting.

The benchmark text is clean; real mail is not. This perturbs held-out rows in
ways that do not change the answer (HTML, quotes, signatures, typos, casing)
plus one control that does remove evidence (truncation). It measures the
preprocessing + model path, not a claim about unseen semantics.

Run: python -m mailroom_triage.stress
"""

from __future__ import annotations

import argparse
import html
import json
import random
import re
from pathlib import Path

import numpy as np
import torch

from .config import ensure_dirs, load_config
from .data import load_examples, make_examples, read_split
from .inference import collect_logits, softmax
from .model import load_exported

PERTURBATIONS = ("clean", "html", "quoted", "signature", "typos", "case", "reordered", "forwarded", "truncate")


def _html_wrap(body: str) -> str:
    paragraphs = "".join(f"<p>{html.escape(part)}&nbsp;</p>" for part in body.split("\n\n"))
    return f"<html><head><style>p{{color:#333}}</style></head><body>{paragraphs}</body></html>"


def _typos(text: str, rng: random.Random) -> str:
    words = text.split(" ")
    for _ in range(max(1, len(words) // 25)):
        i = rng.randrange(len(words))
        word = words[i]
        if len(word) < 4:
            continue
        j = rng.randrange(1, len(word) - 1)
        mode = rng.randrange(3)
        if mode == 0:
            word = word[:j] + word[j + 1 :]
        elif mode == 1:
            word = word[:j] + word[j + 1] + word[j] + word[j + 2 :]
        else:
            word = word[:j] + rng.choice("aeiouns") + word[j + 1 :]
        words[i] = word
    return " ".join(words)


def _case_noise(text: str, rng: random.Random) -> str:
    out = []
    for word in text.split(" "):
        out.append(word.upper() if rng.random() < 0.2 else word.lower() if rng.random() < 0.2 else word)
    return "   " + "\n\n".join(out) + "   "


def perturb(email: dict, kind: str, rng: random.Random) -> dict:
    if kind == "clean":
        return email
    body = email.get("body", "")
    if kind == "html":
        return {**email, "body": _html_wrap(body)}
    if kind == "quoted":
        quoted = "\n".join(f"> {line}" for line in body.splitlines())
        return {**email, "body": f"{body}\n\nOn Mon, Jan 12, 2026 at 9:14 AM Support <support@example.com> wrote:\n{quoted}"}
    if kind == "signature":
        return {**email, "body": body + "\n\n-- \nJane Doe\nBilling Team\n+1 555 0100\nThis message is confidential."}
    if kind == "typos":
        return {**email, "body": _typos(body, rng)}
    if kind == "case":
        return {**email, "body": _case_noise(body, rng)}
    if kind == "reordered":
        lines = body.splitlines()
        rng.shuffle(lines)
        return {**email, "body": "\n".join(lines)}
    if kind == "forwarded":
        header = "---------- Forwarded message ---------\nFrom: accounts@example.com\nDate: Mon, Jan 12, 2026\nSubject: Fwd: message\n\n"
        return {**email, "body": header + body}
    if kind == "truncate":
        return {**email, "body": body[: max(40, int(len(body) * 0.4))]}
    raise ValueError(kind)


def _sample(rows: list[dict], limit: int | None) -> list[dict]:
    if not limit or limit >= len(rows):
        return rows
    stride = max(1, len(rows) // limit)
    return rows[::stride][:limit]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--split", default="validation")
    parser.add_argument("--limit", type=int, default=3000)
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    model_dir = Path(args.model_dir) if args.model_dir else Path(cfg["artifacts_dir"]) / "model"

    model, tokenizer, registry = load_exported(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    temp_path = model_dir / "temperatures.json"
    temperatures = json.loads(temp_path.read_text(encoding="utf-8")) if temp_path.exists() else {}

    rows = _sample(list(read_split(cfg["data_root"], args.split)), args.limit)
    report: dict = {"model_dir": str(model_dir), "split": args.split, "rows": len(rows), "perturbations": {}}
    print(f"{args.split}: {len(rows)} rows, device={device}", flush=True)

    for kind in PERTURBATIONS:
        rng = random.Random(cfg["seed"])
        perturbed = []
        for row in rows:
            state = {**row["state"], "email": perturb(row["state"]["email"], kind, rng)}
            perturbed.append({**row, "state": state})
        examples = make_examples(perturbed, registry, args.split)
        collected = collect_logits(model, tokenizer, examples, cfg["max_length"], cfg["eval_batch_size"], device, cfg["precision"])

        head_accuracy = {}
        correct = total = 0
        for key, data in collected.items():
            probs = softmax(data["logits"], temperatures.get(key, 1.0))
            truth = data["targets"].argmax(axis=1)
            hits = int((probs.argmax(axis=1) == truth).sum())
            head_accuracy[key] = {"rows": int(len(truth)), "accuracy": round(hits / len(truth), 4)}
            correct += hits
            total += len(truth)
        report["perturbations"][kind] = {"accuracy": round(correct / total, 4), "heads": head_accuracy}
        print(f"{kind:10s} accuracy={correct/total:.4f}", flush=True)

    out = Path(cfg["reports_dir"]) / "stress.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Robustness under realistic mail formatting",
        "",
        f"Split: `{args.split}` ({report['rows']} rows), model: `{model_dir.name}`.",
        "Formatting perturbations do not change the answer; `truncate` removes evidence and is a control.",
        "",
        "| Perturbation | Accuracy | " + " | ".join(report["perturbations"]["clean"]["heads"]) + " |",
        "|---|---:|" + "---:|" * len(report["perturbations"]["clean"]["heads"]),
    ]
    for kind, info in report["perturbations"].items():
        cells = " | ".join(f"{info['heads'][key]['accuracy']:.3f}" for key in info["heads"])
        lines.append(f"| {kind} | {info['accuracy']:.4f} | {cells} |")
    Path(cfg["reports_dir"], "stress.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
