from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mailroom_triage.serve import create_app

MODEL_DIR = Path(__file__).resolve().parents[1] / "artifacts" / "model"
pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "registry.json").exists(),
    reason="model artifact not present",
)

EMAIL = {
    "subject": "Invoice INV-1234 from Luma",
    "from": "billing@luma.example",
    "date": "Mon, 12 Jan 2026 09:14:00 +0000",
    "body": (
        "We are the provider and issue our own charges. Your invoice is attached. "
        "Amount now owed: EUR 42.00. Reference number: INV-1234."
    ),
}


@pytest.fixture(scope="module")
def client():
    with TestClient(create_app()) as test_client:
        yield test_client


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["heads"] == 12


def test_predict_all_heads(client):
    response = client.post("/predict", json=EMAIL)
    assert response.status_code == 200
    predictions = response.json()
    assert len(predictions) == 12
    for prediction in predictions:
        assert 0.0 <= prediction["confidence"] <= 1.0
        assert abs(sum(prediction["probabilities"].values()) - 1.0) < 1e-3
        assert isinstance(prediction["accepted"], bool)


def test_predict_single_head_alias(client):
    response = client.post("/predict", json={**EMAIL, "heads": ["category"]})
    assert response.status_code == 200
    predictions = response.json()
    assert len(predictions) == 1
    assert predictions[0]["head"].startswith("category#")


def test_unknown_head_is_rejected(client):
    response = client.post("/predict", json={**EMAIL, "heads": ["not_a_head"]})
    assert response.status_code == 400


def test_batch_endpoint(client):
    response = client.post("/predict/batch", json=[EMAIL, EMAIL])
    assert response.status_code == 200
    assert len(response.json()) == 2


def test_html_email_is_cleaned(client):
    html_email = {
        **EMAIL,
        "body": "<html><body><style>p{color:red}</style><p>Amount now owed: EUR 42.00</p></body></html>",
    }
    response = client.post("/predict", json=html_email)
    assert response.status_code == 200
    assert len(response.json()) == 12
