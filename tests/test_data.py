from pathlib import Path

import pytest
from transformers import AutoTokenizer

from mailroom_triage.data import (
    Example,
    augment_line_order,
    render_text,
    render_text_with_limit,
)


def _tokenizer():
    root = Path(__file__).resolve().parents[1] / "artifacts" / "model" / "tokenizer"
    if not root.exists():
        pytest.skip("model tokenizer not present")
    return AutoTokenizer.from_pretrained(root, local_files_only=True)


def _example(body: str) -> Example:
    return Example(
        head_key="from_billing_entity",
        question="Is this sender a provider?",
        options=("no", "yes"),
        target=(0.0, 1.0),
        kind="noul",
        lang="en",
        variant="invoice:en:base",
        group_id="g",
        case_id="c",
        state={"subject": "s", "from": "a@b.c", "date": "d", "body": body},
        split="train",
    )


def test_augment_line_order_keeps_content_and_is_deterministic():
    body = "line one\nline two\nline three"
    first = augment_line_order([_example(body)], seed=7, probability=1.0)[0]
    second = augment_line_order([_example(body)], seed=7, probability=1.0)[0]
    assert first.state["body"] == second.state["body"]
    assert sorted(first.state["body"].splitlines()) == sorted(body.splitlines())


def test_augment_line_order_leaves_single_line_untouched():
    example = _example("only one line")
    out = augment_line_order([example], seed=7, probability=1.0)[0]
    assert out.state["body"] == "only one line"


def test_long_body_keeps_question_within_limit():
    tokenizer = _tokenizer()
    state = {
        "subject": "Invoice INV-1",
        "from": "billing@luma.example",
        "date": "Mon",
        "body": "Amount now owed: EUR 5.00. " * 200,
    }
    question = "Is this sender a provider issuing its own charges?"
    text = render_text_with_limit(state, question, tokenizer, 320)
    ids = tokenizer(text, add_special_tokens=True)["input_ids"]
    assert len(ids) <= 320
    assert "issuing its own charges" in tokenizer.decode(ids).lower()


def test_short_mail_is_unchanged_by_the_limit():
    tokenizer = _tokenizer()
    state = {"subject": "Invoice", "from": "a@b.c", "date": "Mon", "body": "Amount owed: EUR 5."}
    question = "Is this an invoice?"
    assert render_text_with_limit(state, question, tokenizer, 320) == render_text(state, question)
