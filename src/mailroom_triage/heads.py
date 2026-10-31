"""Head registry and option canonicalisation.

Every row is one decision over an ordered option set. Option order is a
presentation artefact, not a label, so options are sorted canonically and the
target is permuted with them. A head is identified by (question_id, option set),
which yields 10 fixed heads plus two category taxonomies (6 and 7 options).
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class HeadSpec:
    key: str
    question_id: str
    kind: str
    options: tuple[str, ...]
    question: str = ""
    default: bool = False


def canonicalize(options: list[str] | tuple[str, ...]) -> tuple[tuple[str, ...], list[int]]:
    """Return sorted options and the permutation old_index -> new_index order."""
    order = sorted(range(len(options)), key=lambda i: (str(options[i]).strip().casefold(), i))
    return tuple(options[i] for i in order), order


def permute_target(target: list[float], order: list[int]) -> list[float]:
    return [float(target[i]) for i in order]


def build_registry(rows: list[dict]) -> dict[str, HeadSpec]:
    """Build heads from training rows only. Rows must expose metadata.question_id."""
    grouped: dict[tuple[str, tuple[str, ...]], list[dict]] = defaultdict(list)
    for row in rows:
        options, _ = canonicalize(row["options"])
        grouped[(row["metadata"]["question_id"], options)].append(row)

    by_qid: dict[str, list[tuple[tuple[str, ...], int, str, str]]] = defaultdict(list)
    for (qid, options), items in grouped.items():
        by_qid[qid].append((options, len(items), items[0]["kind"], str(items[0].get("question", ""))))

    registry: dict[str, HeadSpec] = {}
    for qid, variants in by_qid.items():
        best = max(variants, key=lambda item: item[1])[0]
        multiple = len(variants) > 1
        for options, _count, kind, question in variants:
            key = f"{qid}#{len(options)}" if multiple else qid
            if key in registry:
                raise ValueError(f"duplicate head key {key}")
            registry[key] = HeadSpec(key, qid, kind, options, question, options == best)
    return dict(sorted(registry.items()))


def save_registry(registry: dict[str, HeadSpec], path: str | Path) -> None:
    payload = [asdict(spec) for spec in registry.values()]
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def load_registry(path: str | Path) -> dict[str, HeadSpec]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        item["key"]: HeadSpec(
            key=item["key"],
            question_id=item["question_id"],
            kind=item["kind"],
            options=tuple(item["options"]),
            question=str(item.get("question", "")),
            default=bool(item.get("default", False)),
        )
        for item in payload
    }


def resolve_head(name: str, registry: dict[str, HeadSpec]) -> str:
    """Accept either an exact key or a question_id alias (default taxonomy)."""
    if name in registry:
        return name
    matches = [spec for spec in registry.values() if spec.question_id == name and spec.default]
    if len(matches) == 1:
        return matches[0].key
    raise KeyError(f"unknown head {name!r}")
