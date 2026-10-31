"""Data audit: counts, splits, heads, leakage guard, duplicates.

Run: python -m mailroom_triage.audit
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

from .config import ensure_dirs, load_config
from .data import EXPECTED_COUNTS, SPLITS, build_text, load_examples, make_examples, read_split
from .heads import build_registry, save_registry


def _hash(example) -> str:
    payload = json.dumps(
        [example.state, example.question, example.options], sort_keys=True, ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    ensure_dirs(cfg)

    train_rows = list(read_split(cfg["data_root"], "train"))
    registry = build_registry(train_rows)
    save_registry(registry, Path(cfg["artifacts_dir"]) / "registry.json")
    train_examples = make_examples(train_rows, registry, "train")

    report: dict = {"splits": {}, "heads": {}, "checks": {}}
    groups: dict[str, set[str]] = {}
    duplicates: dict[str, int] = {}

    for split in SPLITS:
        examples = load_examples(cfg["data_root"], registry, split)
        counts = collections.Counter(example.head_key for example in examples)
        langs = collections.Counter(example.lang for example in examples)
        variants = collections.Counter(example.variant for example in examples)
        seen = collections.Counter(_hash(example) for example in examples)
        duplicates[split] = sum(count - 1 for count in seen.values() if count > 1)
        groups[split] = {example.group_id for example in examples}
        report["splits"][split] = {
            "rows": len(examples),
            "expected": EXPECTED_COUNTS[split],
            "matches_expected": len(examples) == EXPECTED_COUNTS[split],
            "language": dict(sorted(langs.items())),
            "heads": dict(sorted(counts.items())),
            "top_variants": dict(variants.most_common(8)),
            "duplicate_texts": duplicates[split],
        }
        if not report["splits"][split]["matches_expected"]:
            raise SystemExit(f"{split}: {len(examples)} rows, expected {EXPECTED_COUNTS[split]}")

    for spec in registry.values():
        train_subset = [example for example in train_examples if example.head_key == spec.key]
        positive = sum(example.target[1] for example in train_subset if spec.kind == "noul")
        report["heads"][spec.key] = {
            "question_id": spec.question_id,
            "kind": spec.kind,
            "options": list(spec.options),
            "default": spec.default,
            "train_rows": len(train_subset),
            "positive_rate": round(positive / len(train_subset), 4) if spec.kind == "noul" else None,
        }

    # Group separation across splits.
    overlaps = {}
    for left in SPLITS:
        for right in SPLITS:
            if left >= right:
                continue
            shared = groups[left] & groups[right]
            if shared:
                overlaps[f"{left}/{right}"] = len(shared)
    report["checks"]["group_overlap"] = overlaps
    report["checks"]["groups_disjoint"] = not overlaps

    # Leakage guard: privileged strings must not appear in model text.
    sample_text = "\n".join(build_text(example) for example in load_examples(cfg["data_root"], registry, "train")[:1000])
    report["checks"]["target_basis_in_text"] = "target_basis" in sample_text
    report["checks"]["metadata_in_text"] = "question_id" in sample_text
    report["checks"]["duplicate_texts_total"] = sum(duplicates.values())

    out_json = Path(cfg["reports_dir"]) / "data_report.json"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    lines = ["# Data report", "", f"Rows per split: { {k: v['rows'] for k, v in report['splits'].items()} }", ""]
    lines.append("| Split | Rows | Languages | Duplicate texts |")
    lines.append("|---|---:|---|---:|")
    for split, info in report["splits"].items():
        lines.append(f"| {split} | {info['rows']} | {info['language']} | {info['duplicate_texts']} |")
    lines.append("")
    lines.append("| Head | Kind | Options | Train rows | Positive rate |")
    lines.append("|---|---|---:|---:|---:|")
    for key, info in report["heads"].items():
        rate = "" if info["positive_rate"] is None else f"{info['positive_rate']:.3f}"
        lines.append(f"| {key} | {info['kind']} | {len(info['options'])} | {info['train_rows']} | {rate} |")
    lines.append("")
    lines.append(f"Groups disjoint across splits: {report['checks']['groups_disjoint']}")
    lines.append(f"Privileged string in model text: {report['checks']['target_basis_in_text']}")
    (Path(cfg["reports_dir"]) / "data_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(report["checks"], indent=2))
    print(f"registry: {len(registry)} heads")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
