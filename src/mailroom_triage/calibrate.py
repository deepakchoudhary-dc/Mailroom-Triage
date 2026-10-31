"""Fit one temperature per head and pick operating thresholds on calibration data.

Run: python -m mailroom_triage.calibrate
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .config import ensure_dirs, load_config
from .data import load_examples
from .inference import collect_logits, softmax
from .metrics import fit_temperature
from .model import load_exported


def choose_threshold(probs: np.ndarray, correct: np.ndarray, target_precision: float, min_coverage: float) -> dict:
    best = None
    fallback = None
    for threshold in np.arange(0.50, 0.995, 0.01):
        accepted = probs.max(axis=1) >= threshold
        coverage = float(accepted.mean())
        if coverage == 0:
            continue
        precision = float(correct[accepted].mean())
        score = precision * coverage
        if fallback is None or score > fallback["score"]:
            fallback = {"threshold": round(float(threshold), 2), "coverage": round(coverage, 4), "precision": round(precision, 4), "score": score}
        if precision >= target_precision and coverage >= min_coverage:
            if best is None or coverage > best["coverage"]:
                best = {"threshold": round(float(threshold), 2), "coverage": round(coverage, 4), "precision": round(precision, 4)}
    if fallback is None:
        fallback = {"threshold": 0.0, "coverage": 1.0, "precision": round(float(correct.mean()), 4), "score": float(correct.mean())}
    chosen = best or fallback
    chosen["met_target"] = best is not None
    return chosen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--model-dir", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)
    model_dir = Path(args.model_dir) if args.model_dir else Path(cfg["artifacts_dir"]) / "model"

    model, tokenizer, registry = load_exported(model_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    examples = load_examples(cfg["data_root"], registry, "calibration")
    collected = collect_logits(model, tokenizer, examples, cfg["max_length"], cfg["eval_batch_size"], device, cfg.get("precision", "fp32"))

    temperatures = {}
    thresholds = {}
    for key, data in collected.items():
        logits, targets = data["logits"], data["targets"]
        temperature = 1.0
        if len(targets) >= 50:
            temperature = float(fit_temperature(logits.tolist(), targets.tolist()))
        temperatures[key] = round(temperature, 4)
        probs = softmax(logits, temperature)
        correct = (probs.argmax(axis=1) == targets.argmax(axis=1)).astype(np.float64)
        threshold = choose_threshold(probs, correct, cfg["target_precision"], cfg["min_coverage"])
        thresholds[key] = {
            **threshold,
            "critical": key in cfg["critical_heads"] or registry[key].question_id in cfg["critical_heads"],
            "rows": int(len(targets)),
        }
        print(f"{key:24s} T={temperature:.3f} threshold={threshold['threshold']:.2f} coverage={threshold['coverage']:.3f} precision={threshold['precision']:.3f}")

    (model_dir / "temperatures.json").write_text(json.dumps(temperatures, indent=2), encoding="utf-8")
    (model_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
    print(f"wrote {model_dir/'temperatures.json'} and thresholds.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
