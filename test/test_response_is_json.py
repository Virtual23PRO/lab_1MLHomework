from typing import Any
from fastapi.testclient import TestClient
from app import app
from inference import ID2LABEL

client = TestClient(app)
ALLOWED = set(ID2LABEL.values())


def test_predict_returns_valid_json_response():
    resp = client.post("/predict", json={"text": "What a great MLOps lecture!"})

    assert resp.status_code == 200

    assert resp.headers.get("content-type", "").startswith("application/json")
    body: dict[str, Any] = resp.json()
    assert isinstance(body, dict)

    assert set(body.keys()) == {"prediction"}
    assert isinstance(body["prediction"], str)

    assert body["prediction"] in ALLOWED
