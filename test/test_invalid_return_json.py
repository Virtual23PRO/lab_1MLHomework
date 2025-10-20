from typing import Any
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)


def test_invalid_input_returns_json_with_explanation():
    resp = client.post("/predict", json={"text": ""})
    assert resp.status_code == 422
    assert resp.headers["content-type"].startswith("application/json")
    body: dict[str, Any] = resp.json()
    assert (
        "detail" in body
        and isinstance(body["detail"], list)
        and len(body["detail"]) >= 1
    )
    err = body["detail"][0]
    assert "msg" in err and "loc" in err and "type" in err
