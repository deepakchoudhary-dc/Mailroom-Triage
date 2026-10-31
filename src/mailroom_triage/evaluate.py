"""Evaluate the exported model on validation, test and OOD, with slices.

Run: python -m mailroom_triage.evaluate
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score

from .config import ensure_dirs, load_config
from .data import load_examples
from .inference import collect_logits, softmax
from .metrics import evaluate_probabilities
from .model import load_exported


def head_metrics(targets: np.ndarray, probs: np.ndarray, kind: str, threshold: float) -> dict:
    core = evaluate_probabilities(targets.tolist(), probs.tolist())
    truth = targets.argmax(axis=1)
    pred = probs.argmax(axis=1)
    accepted = probs.max(axis=1) >= threshold
    result = {
        "rows": int(len(truth)),
        "nll": round(float(core["nll"]), 5),
        "brier": round(float(core["brier"]), 5),
        "ece": round(float(core["multiclass_ece"]), 5),
        "accuracy": round(float((truth == pred).mean()), 5),
        "coverage": round(float(accepted.mean()), 5),
        "precision_at_threshold": round(float((truth[accepted] == pred[accepted]).mean()), 5) if accepted.any() else None,
    }
    if len(set(truth.tolist())) > 1:
        if kind == "noul" and probs.shape[1] == 2:
            result["auroc"] = round(float(roc_auc_score(targets[:, 1], probs[:, 1])), 5)
        else:
            result["macro_f1"] = round(float(f1_score(truth, pred, average="macro", zero_division=0)), 5)
    return result


def slice_metrics(targets, probs, kind, threshold, mask):
    if not np.any(mask):
        return None
    return head_metrics(targets[mask], probs[mask], kind, threshold)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--splits", nargs="+", default=["calibration", "validation", "test", "ood"])
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    model_dir = Path(args.model_dir) if args.model_dir else Path(cfg["artifacts_dir"]) / "model"

    model, tokenizer, registry = load_exported(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    temperatures = json.loads((model_dir / "temperatures.json").read_text(encoding="utf-8"))
    thresholds = json.loads((model_dir / "thresholds.json").read_text(encoding="utf-8"))

    report: dict = {"model_dir": str(model_dir), "temperatures": temperatures, "thresholds": thresholds, "splits": {}, "language": {}, "variant": {}, "confusion": {}, "agreement": {}}
    test_collected = None

    for split in args.splits:
        examples = load_examples(cfg["data_root"], registry, split)
        collected = collect_logits(model, tokenizer, examples, cfg["max_length"], cfg["eval_batch_size"], device, cfg.get("precision", "fp32"))
        if split == "test":
            test_collected = collected
        report["splits"][split] = {}
        report["language"][split] = defaultdict(dict)
        report["variant"][split] = defaultdict(dict)
        for key, data in collected.items():
            spec = registry[key]
            threshold = thresholds.get(key, {}).get("threshold", 0.5)
            probs = softmax(data["logits"], temperatures.get(key, 1.0))
            targets = data["targets"]
            report["splits"][split][key] = head_metrics(targets, probs, spec.kind, threshold)
            langs = np.array([meta[0] for meta in data["meta"]])
            for lang in sorted(set(langs.tolist())):
                report["language"][split][lang][key] = slice_metrics(targets, probs, spec.kind, threshold, langs == lang)
            prefixes = np.array([str(meta[1]).split(":")[0] for meta in data["meta"]])
            for prefix in sorted(set(prefixes.tolist())):
                report["variant"][split][prefix][key] = slice_metrics(targets, probs, spec.kind, threshold, prefixes == prefix)

        def micro(heads: dict) -> dict:
            scored = [value for value in heads.values() if value]
            rows = sum(value["rows"] for value in scored)
            hits = sum(value["accuracy"] * value["rows"] for value in scored)
            return {"rows": rows, "micro_accuracy": round(hits / max(1, rows), 5)}

        report.setdefault("summary", {})[split] = {
            **micro(report["splits"][split]),
            "language": {lang: micro(heads) for lang, heads in report["language"][split].items()},
        }

        # Confusion matrices for the two choice heads on this split.
        for key in registry:
            if registry[key].kind != "choice" or key not in collected:
                continue
            probs = softmax(collected[key]["logits"], temperatures.get(key, 1.0))
            truth = collected[key]["targets"].argmax(axis=1)
            pred = probs.argmax(axis=1)
            matrix = confusion_matrix(truth, pred, labels=list(range(len(registry[key].options)))).tolist()
            report["confusion"][f"{split}/{key}"] = {"options": [option.split(":")[0] for option in registry[key].options], "matrix": matrix}

    # Cross-head consistency: invoice + no amount owed is valid but worth tracking.
    if test_collected is not None:
        kind_key = next((key for key in registry if registry[key].question_id == "kind"), None)
        amount_key = next((key for key in registry if registry[key].question_id == "states_amount_owed"), None)
        if kind_key and amount_key:
            kind_labels = {meta[3]: probs for meta, probs in zip(test_collected[kind_key]["meta"], softmax(test_collected[kind_key]["logits"], temperatures.get(kind_key, 1.0)))}
            amount_labels = {meta[3]: probs for meta, probs in zip(test_collected[amount_key]["meta"], softmax(test_collected[amount_key]["logits"], temperatures.get(amount_key, 1.0)))}
            kind_invoice = next(i for i, option in enumerate(registry[kind_key].options) if option.lower().startswith("invoice"))
            amount_no = next(i for i, option in enumerate(registry[amount_key].options) if option.strip().lower() == "no")
            shared = set(kind_labels) & set(amount_labels)
            flagged = [case for case in shared if kind_labels[case].argmax() == kind_invoice and amount_labels[case].argmax() == amount_no]
            report["agreement"]["invoice_without_amount_owed"] = {"cases": len(shared), "flagged": len(flagged), "rate": round(len(flagged) / max(1, len(shared)), 4)}

    out = Path(cfg["reports_dir"]) / "eval.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    def fmt(value, digits: int = 3) -> str:
        return "" if value is None else f"{value:.{digits}f}"

    lines = ["# Evaluation", "", "## Summary", "", "| Split | Rows | Micro accuracy | en | tr | zh |", "|---|---:|---:|---:|---:|---:|"]
    for split, info in report["summary"].items():
        langs = info["language"]
        cells = " | ".join(f"{langs[lang]['micro_accuracy']:.4f}" if lang in langs else "" for lang in ("en", "tr", "zh"))
        lines.append(f"| {split} | {info['rows']} | {info['micro_accuracy']:.4f} | {cells} |")
    lines += ["", "## Test", "", "| Head | Rows | NLL | Brier | ECE | Accuracy | AUROC / F1 | Coverage | Precision@thr |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key, value in report["splits"].get("test", {}).items():
        extra = value.get("auroc", value.get("macro_f1"))
        lines.append(
            f"| {key} | {value['rows']} | {value['nll']:.4f} | {value['brier']:.4f} | {value['ece']:.4f} | {value['accuracy']:.3f} | {fmt(extra)} | {value['coverage']:.3f} | {fmt(value['precision_at_threshold'])} |"
        )
    lines += ["", "## OOD", "", "| Head | Rows | NLL | Brier | ECE | Accuracy | AUROC / F1 | Coverage |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for key, value in report["splits"].get("ood", {}).items():
        extra = value.get("auroc", value.get("macro_f1"))
        lines.append(
            f"| {key} | {value['rows']} | {value['nll']:.4f} | {value['brier']:.4f} | {value['ece']:.4f} | {value['accuracy']:.3f} | {fmt(extra)} | {value['coverage']:.3f} |"
        )
    (Path(cfg["reports_dir"]) / "eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
