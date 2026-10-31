"""TF-IDF + logistic regression baseline, one model per head.

Run: python -m mailroom_triage.baseline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score

from .config import ensure_dirs, load_config
from .data import SPLITS, build_text, load_examples, read_split
from .heads import build_registry, save_registry
from .metrics import evaluate_probabilities


def fit_predict(train_texts, train_targets, texts_by_split, n_options):
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_features=60000, sublinear_tf=True)
    train_matrix = vectorizer.fit_transform(train_texts)
    y = np.asarray(train_targets).argmax(axis=1)
    classifier = LogisticRegression(max_iter=2000, C=4.0, solver="lbfgs")
    classifier.fit(train_matrix, y)
    class_to_index = {int(label): index for index, label in enumerate(classifier.classes_)}

    output = {}
    for split, texts in texts_by_split.items():
        matrix = vectorizer.transform(texts)
        raw = classifier.predict_proba(matrix)
        probs = np.zeros((raw.shape[0], n_options), dtype=np.float64)
        for column, label in enumerate(classifier.classes_):
            probs[:, class_to_index[int(label)]] = raw[:, column]
        probs = probs / probs.sum(axis=1, keepdims=True)
        output[split] = probs
    return output


def score_split(targets: list[list[float]], probs: np.ndarray, kind: str) -> dict:
    result = evaluate_probabilities(targets, probs.tolist())
    truth = np.asarray(targets).argmax(axis=1)
    pred = probs.argmax(axis=1)
    result["accuracy"] = round(float((truth == pred).mean()), 5)
    if len(set(truth.tolist())) > 1:
        if kind == "noul" and probs.shape[1] == 2:
            result["auroc"] = round(float(roc_auc_score(np.asarray(targets)[:, 1], probs[:, 1])), 5)
        else:
            result["macro_f1"] = round(float(f1_score(truth, pred, average="macro", zero_division=0)), 5)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)

    registry = build_registry(list(read_split(cfg["data_root"], "train")))
    save_registry(registry, Path(cfg["artifacts_dir"]) / "registry.json")
    examples = {split: load_examples(cfg["data_root"], registry, split) for split in SPLITS}
    texts = {split: [build_text(example) for example in rows] for split, rows in examples.items()}

    report: dict = {"splits": {}, "heads": {}}
    for key, spec in registry.items():
        train_rows = [example for example in examples["train"] if example.head_key == key]
        indices = {split: [i for i, example in enumerate(examples[split]) if example.head_key == key] for split in SPLITS}
        texts_by_split = {split: [texts[split][i] for i in indices[split]] for split in SPLITS}
        predictions = fit_predict(
            [build_text(example) for example in train_rows],
            [list(example.target) for example in train_rows],
            texts_by_split,
            len(spec.options),
        )
        report["heads"][key] = {"kind": spec.kind, "options": list(spec.options), "train_rows": len(train_rows), "splits": {}}
        for split in SPLITS:
            targets = [list(examples[split][i].target) for i in indices[split]]
            report["heads"][key]["splits"][split] = score_split(targets, predictions[split], spec.kind)
            print(f"{key:24s} {split:12s} nll={report['heads'][key]['splits'][split]['nll']:.4f}")

    out = Path(cfg["reports_dir"]) / "baseline.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = ["# TF-IDF baseline", "", "| Head | Split | NLL | Brier | ECE | Accuracy | AUROC / F1 |", "|---|---|---:|---:|---:|---:|---:|"]
    for key, info in report["heads"].items():
        for split in SPLITS:
            s = info["splits"][split]
            extra = s.get("auroc", s.get("macro_f1"))
            lines.append(
                f"| {key} | {split} | {s['nll']:.4f} | {s['brier']:.4f} | {s['multiclass_ece']:.4f} | {s['accuracy']:.3f} | {extra if extra is None else f'{extra:.3f}'} |"
            )
    (Path(cfg["reports_dir"]) / "baseline.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
