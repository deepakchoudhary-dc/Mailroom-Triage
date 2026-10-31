"""Shared inference helpers for calibration, evaluation and serving."""

from __future__ import annotations

from collections import defaultdict
from functools import partial

import numpy as np
import torch
from torch.utils.data import DataLoader

from .data import EncodedDataset, HeadBatchSampler, collate, encode_examples
from .model import autocast_ctx


@torch.no_grad()
def collect_logits(model, tokenizer, examples, max_length: int, batch_size: int, device, precision: str = "fp32") -> dict:
    encoded = encode_examples(examples, tokenizer, max_length)
    dataset = EncodedDataset(encoded)
    sampler = HeadBatchSampler(encoded, batch_size, shuffle=False)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=partial(collate, pad_token_id=tokenizer.pad_token_id), num_workers=0)
    out: dict = defaultdict(lambda: {"logits": [], "targets": [], "meta": []})
    model.eval()
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention = batch["attention_mask"].to(device)
        with autocast_ctx(device, precision):
            logits = model(input_ids, attention, batch["head_key"])
        entry = out[batch["head_key"]]
        entry["logits"].append(logits.detach().float().cpu())
        entry["targets"].append(batch["targets"].float().cpu())
        entry["meta"].extend(batch["meta"])
    return {
        key: {
            "logits": torch.cat(value["logits"]).numpy(),
            "targets": torch.cat(value["targets"]).numpy(),
            "meta": value["meta"],
        }
        for key, value in out.items()
    }


@torch.no_grad()
def predict_texts(model, tokenizer, texts: list[str], head_key: str, max_length: int, device, precision: str = "fp32") -> np.ndarray:
    batch = tokenizer(texts, truncation=True, max_length=max_length, padding=True, return_tensors="pt")
    input_ids = batch["input_ids"].to(device)
    attention = batch["attention_mask"].to(device)
    with autocast_ctx(device, precision):
        logits = model(input_ids, attention, head_key)
    return logits.float().cpu().numpy()


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / float(temperature)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)
