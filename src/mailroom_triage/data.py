"""Data loading, canonicalisation and batching.

Input contract: state (subject/from/date/body) + question + options only.
metadata is used for head routing and slice reporting, never as model input.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import random
from collections import defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Iterator

import torch
from torch.utils.data import Dataset

from .heads import HeadSpec, canonicalize, permute_target
from .preprocess import clean_state

SPLITS = ("train", "calibration", "validation", "test", "ood")
EMAIL_FIELDS = ("subject", "from", "date", "body")

EXPECTED_COUNTS = {
    "train": 73472,
    "calibration": 4018,
    "validation": 6027,
    "test": 8323,
    "ood": 22960,
}


@dataclass
class Example:
    head_key: str
    question: str
    options: tuple[str, ...]
    target: tuple[float, ...]
    kind: str
    lang: str
    variant: str
    group_id: str
    case_id: str
    state: dict
    split: str


@dataclass
class Encoded:
    input_ids: list[int]
    attention_mask: list[int]
    head_key: str
    target: tuple[float, ...]
    lang: str
    variant: str
    group_id: str
    case_id: str


def read_split(data_root: str | Path, split: str) -> Iterator[dict]:
    path = Path(data_root) / f"{split}.jsonl.gz"
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def lookup_table(registry: dict[str, HeadSpec]) -> dict[tuple[str, tuple[str, ...]], str]:
    return {(spec.question_id, spec.options): spec.key for spec in registry.values()}


def make_examples(rows: Iterable[dict], registry: dict[str, HeadSpec], split: str) -> list[Example]:
    lookup = lookup_table(registry)
    out: list[Example] = []
    for row in rows:
        state = row["state"]["email"]
        missing = [name for name in EMAIL_FIELDS if name not in state]
        if missing:
            raise ValueError(f"{split}: email missing {missing}")
        qid = row["metadata"]["question_id"]
        options, order = canonicalize(row["options"])
        key = lookup.get((qid, options))
        if key is None:
            raise ValueError(f"{split}: unknown head {qid} with {len(options)} options")
        target = permute_target(row["target"], order)
        if len(target) != len(options):
            raise ValueError(f"{split}: target/option length mismatch")
        if abs(sum(target) - 1.0) > 1e-6 or any(value < 0 for value in target):
            raise ValueError(f"{split}: target is not a distribution")
        meta = row["metadata"]
        out.append(
            Example(
                head_key=key,
                question=row["question"],
                options=options,
                target=tuple(target),
                kind=row["kind"],
                lang=str(meta.get("language", "unknown")),
                variant=str(meta.get("provenance", {}).get("variant", "unknown")),
                group_id=str(row.get("group_id", "")),
                case_id=str(meta.get("case_id", "")),
                state=dict(state),
                split=split,
            )
        )
    return out


def build_text(example: Example) -> str:
    return render_text(example.state, example.question)


def render_text(state: dict, question: str) -> str:
    """Single source of truth for the model input, shared with serving."""
    email = clean_state(state)
    return (
        f"Subject: {email['subject']}\n"
        f"From: {email['from']}\n"
        f"Date: {email['date']}\n\n"
        f"{email['body']}\n\n"
        f"Question: {question}"
    )


def render_text_with_limit(state: dict, question: str, tokenizer, max_length: int) -> str:
    """Keep the question when a long body would be truncated.

    The question sits after the body, so plain right-truncation can delete the
    question itself. Trim the body instead; short mail is returned unchanged.
    """
    text = render_text(state, question)
    if len(tokenizer(text, add_special_tokens=True, truncation=True, max_length=max_length + 1)["input_ids"]) <= max_length:
        return text
    email = clean_state(state)
    header = f"Subject: {email['subject']}\nFrom: {email['from']}\nDate: {email['date']}\n\n"
    tail = f"\n\nQuestion: {question}"
    overhead = len(tokenizer(header + tail, add_special_tokens=True)["input_ids"])
    body_ids = tokenizer(email["body"], add_special_tokens=False, truncation=True, max_length=max_length)["input_ids"]
    budget = max(8, max_length - overhead)
    email["body"] = tokenizer.decode(body_ids[:budget], skip_special_tokens=True)
    return render_text(email, question)


def augment_line_order(examples: list[Example], seed: int, probability: float = 0.5) -> list[Example]:
    """Shuffle body lines so evidence position cannot become the learned cue.

    Real threads and the corpus OOD split move paragraphs around; without this
    the fine-tune learns where the evidence sits, not what it says.
    """
    rng = random.Random(seed)
    out: list[Example] = []
    for example in examples:
        lines = str(example.state.get("body", "")).splitlines()
        if len(lines) > 1 and rng.random() < probability:
            rng.shuffle(lines)
            out.append(replace(example, state={**example.state, "body": "\n".join(lines)}))
        else:
            out.append(example)
    return out


def encode_examples(examples: list[Example], tokenizer, max_length: int) -> list[Encoded]:
    texts = []
    for example in examples:
        text = build_text(example)
        if len(tokenizer(text, add_special_tokens=True, truncation=True, max_length=max_length + 1)["input_ids"]) > max_length:
            text = render_text_with_limit(example.state, example.question, tokenizer, max_length)
        texts.append(text)
    batch = tokenizer(texts, truncation=True, max_length=max_length, padding=False)
    return [
        Encoded(
            input_ids=list(ids),
            attention_mask=list(mask),
            head_key=example.head_key,
            target=example.target,
            lang=example.lang,
            variant=example.variant,
            group_id=example.group_id,
            case_id=example.case_id,
        )
        for ids, mask, example in zip(batch["input_ids"], batch["attention_mask"], examples)
    ]


class EncodedDataset(Dataset):
    def __init__(self, items: list[Encoded]):
        self.items = items

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Encoded:
        return self.items[index]


class HeadBatchSampler:
    """Homogeneous batches per head so one linear head runs per batch."""

    def __init__(self, items: list[Encoded], batch_size: int, shuffle: bool, seed: int = 42):
        groups: dict[str, list[int]] = defaultdict(list)
        for index, item in enumerate(items):
            groups[item.head_key].append(index)
        rng = random.Random(seed)
        self.batches: list[list[int]] = []
        for key in sorted(groups):
            indices = groups[key][:]
            if shuffle:
                rng.shuffle(indices)
            for start in range(0, len(indices), batch_size):
                self.batches.append(indices[start : start + batch_size])
        if shuffle:
            rng.shuffle(self.batches)

    def __iter__(self):
        return iter(self.batches)

    def __len__(self) -> int:
        return len(self.batches)


def collate(batch: list[Encoded], pad_token_id: int) -> dict:
    head_keys = {item.head_key for item in batch}
    if len(head_keys) != 1:
        raise ValueError(f"mixed heads in a batch: {head_keys}")
    width = max(len(item.input_ids) for item in batch)
    input_ids = torch.full((len(batch), width), pad_token_id, dtype=torch.long)
    attention = torch.zeros((len(batch), width), dtype=torch.long)
    for row, item in enumerate(batch):
        length = len(item.input_ids)
        input_ids[row, :length] = torch.tensor(item.input_ids, dtype=torch.long)
        attention[row, :length] = 1
    targets = torch.tensor([item.target for item in batch], dtype=torch.float32)
    return {
        "input_ids": input_ids,
        "attention_mask": attention,
        "targets": targets,
        "head_key": batch[0].head_key,
        "meta": [(item.lang, item.variant, item.group_id, item.case_id) for item in batch],
    }


def load_examples(data_root: str | Path, registry: dict[str, HeadSpec], split: str) -> list[Example]:
    return make_examples(read_split(data_root, split), registry, split)


def fingerprint(data_root: str | Path) -> str:
    digest = hashlib.sha256()
    for split in SPLITS:
        path = Path(data_root) / f"{split}.jsonl.gz"
        digest.update(path.name.encode())
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()
