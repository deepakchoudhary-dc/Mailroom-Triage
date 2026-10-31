"""FastAPI service for the 12-head triage model.

The service performs the same preprocessing as training/evaluation, encodes the
email once per requested head (the question text differs per head), and returns
calibrated probabilities plus an acceptance decision from the fitted threshold.

Run: python -m mailroom_triage.serve
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from .config import load_config
from .data import render_text_with_limit
from .heads import HeadSpec, resolve_head
from .inference import softmax
from .model import DecisionModel, autocast_ctx, load_exported


@dataclass
class Bundle:
    model: DecisionModel
    tokenizer: object
    registry: dict[str, HeadSpec]
    temperatures: dict[str, float]
    thresholds: dict[str, dict]
    device: torch.device
    precision: str
    max_length: int
    model_name: str


def load_bundle(model_dir: str | Path | None = None) -> Bundle:
    cfg = load_config()
    directory = Path(model_dir or os.getenv("MODEL_DIR") or Path(cfg["artifacts_dir"]) / "model")
    model, tokenizer, registry = load_exported(directory)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    def _read(name: str) -> dict:
        path = directory / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    return Bundle(
        model=model,
        tokenizer=tokenizer,
        registry=registry,
        temperatures=_read("temperatures.json"),
        thresholds=_read("thresholds.json"),
        device=device,
        precision=cfg.get("precision", "fp32"),
        max_length=int(cfg["max_length"]),
        model_name=cfg["model_name"],
    )


class Email(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    subject: str = ""
    sender: str = Field(default="", alias="from")
    date: str = ""
    body: str = ""
    heads: list[str] | None = None


class Prediction(BaseModel):
    head: str
    question_id: str
    label: str
    name: str
    confidence: float
    accepted: bool
    threshold: float
    probabilities: dict[str, float]


def predict(bundle: Bundle, email: Email) -> list[Prediction]:
    requested = email.heads or list(bundle.registry)
    keys = []
    for name in requested:
        try:
            keys.append(resolve_head(name, bundle.registry))
        except KeyError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    state = {"subject": email.subject, "from": email.sender, "date": email.date, "body": email.body}
    texts = [
        render_text_with_limit(state, bundle.registry[key].question, bundle.tokenizer, bundle.max_length)
        for key in keys
    ]
    batch = bundle.tokenizer(
        texts, truncation=True, max_length=bundle.max_length, padding=True, return_tensors="pt"
    )
    input_ids = batch["input_ids"].to(bundle.device)
    attention = batch["attention_mask"].to(bundle.device)
    with torch.inference_mode(), autocast_ctx(bundle.device, bundle.precision):
        embeddings = bundle.model.encode(input_ids, attention)

    results = []
    for index, key in enumerate(keys):
        spec = bundle.registry[key]
        with torch.inference_mode():
            logits = bundle.model.heads[key](embeddings[index : index + 1]).float().cpu().numpy()
        probs = softmax(logits, bundle.temperatures.get(key, 1.0))[0]
        names = [
            option.split(":")[0].strip() if spec.kind == "choice" else option.strip()
            for option in spec.options
        ]
        best = int(probs.argmax())
        threshold = float(bundle.thresholds.get(key, {}).get("threshold", 0.5))
        results.append(
            Prediction(
                head=key,
                question_id=spec.question_id,
                label=spec.options[best],
                name=names[best],
                confidence=round(float(probs[best]), 6),
                accepted=bool(probs[best] >= threshold),
                threshold=round(threshold, 4),
                probabilities={name: round(float(p), 6) for name, p in zip(names, probs)},
            )
        )
    return results


def create_app(bundle: Bundle | None = None) -> FastAPI:
    state: dict[str, Bundle | None] = {"bundle": bundle}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if state["bundle"] is None:
            state["bundle"] = load_bundle()
        yield

    app = FastAPI(title="mailroom-triage", version="0.1.0", lifespan=lifespan)

    def current() -> Bundle:
        if state["bundle"] is None:
            state["bundle"] = load_bundle()
        return state["bundle"]

    @app.get("/health")
    def health() -> dict:
        active = current()
        return {
            "status": "ok",
            "model": active.model_name,
            "device": str(active.device),
            "heads": len(active.registry),
            "calibrated": bool(active.temperatures and active.thresholds),
        }

    @app.get("/heads")
    def heads() -> list[dict]:
        return [
            {
                "key": spec.key,
                "question_id": spec.question_id,
                "kind": spec.kind,
                "options": list(spec.options),
                "default": spec.default,
            }
            for spec in current().registry.values()
        ]

    @app.post("/predict", response_model=list[Prediction])
    def predict_one(email: Email) -> list[Prediction]:
        return predict(current(), email)

    @app.post("/predict/batch", response_model=list[list[Prediction]])
    def predict_many(emails: list[Email]) -> list[list[Prediction]]:
        active = current()
        if len(emails) > 256:
            raise HTTPException(status_code=413, detail="at most 256 emails per batch")
        return [predict(active, email) for email in emails]

    return app


app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run(
        "mailroom_triage.serve:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
    )


if __name__ == "__main__":
    main()
