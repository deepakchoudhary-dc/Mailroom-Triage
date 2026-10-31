"""Encoder with one linear head per (question, option set)."""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from .heads import HeadSpec, load_registry, save_registry


def autocast_ctx(device, precision: str = "fp32"):
    """fp16 only helps on tensor-core GPUs; fp32 is faster on TU117/GTX 16xx."""
    if device.type == "cuda" and precision == "fp16":
        return torch.amp.autocast("cuda", dtype=torch.float16)
    if device.type == "cuda" and precision == "bf16":
        return torch.amp.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


class DecisionModel(nn.Module):
    def __init__(self, encoder: nn.Module, registry: dict[str, HeadSpec], dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        hidden = int(encoder.config.hidden_size)
        self.heads = nn.ModuleDict(
            {key: nn.Linear(hidden, len(spec.options)) for key, spec in registry.items()}
        )

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        return self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, head_key: str) -> torch.Tensor:
        return self.heads[head_key](self.dropout(self.encode(input_ids, attention_mask)))


def build_model(model_name: str, registry: dict[str, HeadSpec]) -> tuple[DecisionModel, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoder = AutoModel.from_pretrained(model_name)
    return DecisionModel(encoder, registry), tokenizer


def export_model(model: DecisionModel, tokenizer, out_dir: str | Path, registry: dict[str, HeadSpec], cfg: dict) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model.encoder.save_pretrained(out / "encoder")
    tokenizer.save_pretrained(out / "tokenizer")
    torch.save({key: value.detach().cpu() for key, value in model.heads.state_dict().items()}, out / "heads.pt")
    save_registry(registry, out / "registry.json")
    (out / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")


def load_exported(out_dir: str | Path) -> tuple[DecisionModel, AutoTokenizer, dict[str, HeadSpec]]:
    out = Path(out_dir)
    registry = load_registry(out / "registry.json")
    encoder = AutoModel.from_pretrained(out / "encoder", local_files_only=True)
    model = DecisionModel(encoder, registry)
    model.heads.load_state_dict(torch.load(out / "heads.pt", map_location="cpu", weights_only=True))
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(out / "tokenizer", local_files_only=True)
    return model, tokenizer, registry
