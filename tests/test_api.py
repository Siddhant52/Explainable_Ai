import io
import os
import tempfile

import numpy as np
import pytest
from PIL import Image


@pytest.fixture(scope="module")
def client():
    tmp = tempfile.mkdtemp()
    os.environ["MODELS_DIR"] = os.path.join(tmp, "models")
    os.environ["DB_PATH"] = os.path.join(tmp, "m.db")
    from scripts.make_dummy_models import main
    main(os.environ["MODELS_DIR"])
    from fastapi.testclient import TestClient
    from backend.app import app
    return TestClient(app)


def png(size=(200, 150)):
    b = io.BytesIO()
    Image.fromarray(np.random.default_rng(1).integers(0, 255, (*size[::-1], 3), dtype=np.uint8)).save(b, format="PNG")
    return b.getvalue()


def test_health_and_cancers(client):
    assert client.get("/api/health").json()["ok"]
    j = client.get("/api/cancers").json()
    assert j["demo_models"] and len(j["cancers"]) == 8


def test_frontend_served(client):
    assert "Multi-Cancer" in client.get("/").text


def test_predict_with_monitoring(client):
    f = {"file": ("a.png", png(), "image/png")}
    r = client.post("/api/predict", files=f, data={"cancer": "Oral Cancer", "patient_id": "P-1", "lime": "true"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert abs(sum(j["probabilities"].values()) - 1) < 1e-4
    assert j["gradcam_png"] and j["lime_png"] and j["change_vs_previous"] is None
    r2 = client.post("/api/predict", files={"file": ("a.png", png(), "image/png")}, data={"cancer": "Oral Cancer", "patient_id": "P-1"})
    assert r2.json()["change_vs_previous"]["class_changed"] in (True, False)
    assert len(client.get("/api/history", params={"patient_id": "P-1"}).json()["records"]) == 2
    assert client.delete("/api/history/P-1").json()["deleted"] == 2


def test_validation_errors(client):
    ok = {"file": ("a.png", png(), "image/png")}
    assert client.post("/api/predict", files=ok, data={"cancer": "Nope"}).status_code == 400
    assert client.post("/api/predict", files=ok, data={"cancer": "Oral Cancer", "patient_id": "bad id!"}).status_code == 400
    bad = {"file": ("a.txt", b"not an image", "text/plain")}
    assert client.post("/api/predict", files=bad, data={"cancer": "Oral Cancer"}).status_code == 400
