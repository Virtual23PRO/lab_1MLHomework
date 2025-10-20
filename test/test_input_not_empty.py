from fastapi.testclient import TestClient
from app import app
import pytest

client = TestClient(app)


@pytest.mark.parametrize("bad", ["", "   "])
def test_predict_rejects_empty_or_whitespace_input(bad: str):
    resp = client.post("/predict", json={"text": bad})
    assert resp.status_code == 422
