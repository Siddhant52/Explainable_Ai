"""FastAPI service: prediction + Grad-CAM + LIME + monitoring log. Run: uvicorn backend.app:app"""
import io
import os
import threading
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

from . import db
from .model_def import IMG_SIZE
from .registry import Registry
from .xai import gradcam, lime_overlay, overlay, png_b64

MODELS_DIR = os.getenv("MODELS_DIR", "models")
DB_PATH = os.getenv("DB_PATH", "data/monitoring.db")
MAX_BYTES = 10 * 1024 * 1024
DISCLAIMER = "Research prototype. Not a medical device and not for clinical decisions."

Path(DB_PATH).parent.mkdir(parents=True, exist_ok=True)
registry = Registry(MODELS_DIR)
con = db.connect(DB_PATH)
infer_lock = threading.Lock()
app = FastAPI(title="Multi-Cancer Explainable Screening", version="1.0")


@app.get("/api/health")
def health():
    return {"ok": True, "demo_models": registry.demo, "cancers": len(registry.cancers), "disclaimer": DISCLAIMER}


@app.get("/api/cancers")
def cancers():
    return {"demo_models": registry.demo, "cancers": registry.info()}


@app.post("/api/predict")
def predict(file: UploadFile = File(...), cancer: str = Form(...), patient_id: str = Form(""),
            lime: bool = Form(False)):
    if cancer not in registry.cancers:
        raise HTTPException(400, f"Unknown cancer type: {cancer}")
    pid = patient_id.strip()
    if pid and not db.PID_RE.match(pid):
        raise HTTPException(400, "patient_id must be 1-40 characters: letters, digits, '_' or '-' (use a pseudonymous ID).")
    data = file.file.read(MAX_BYTES + 1)
    if len(data) > MAX_BYTES:
        raise HTTPException(413, "Image larger than 10 MB.")
    try:
        img = Image.open(io.BytesIO(data)).convert("RGB")
    except (UnidentifiedImageError, OSError):
        raise HTTPException(400, "File is not a readable image.")
    arr = np.asarray(img.resize(IMG_SIZE, Image.BILINEAR), dtype="float32")
    try:
        model = registry.get(cancer)
    except FileNotFoundError as e:
        raise HTTPException(503, f"Model file missing: {e}")

    with infer_lock:
        probs = model.predict(arr[None], verbose=0)[0]
        idx = int(np.argmax(probs))
        cam, _ = gradcam(model, arr, idx)
        lime_png = png_b64(lime_overlay(model, arr, idx)) if lime else None
    over, cam_full = overlay(np.uint8(arr), cam)
    classes = registry.cancers[cancer]["classes"]
    conf, area = float(probs[idx]), float((cam_full > 0.5).mean())

    change = None
    if pid:
        prev = db.last_record(con, pid, cancer)
        if prev:
            change = {"previous_time": prev["created_at"], "previous_class": prev["pred_class"],
                      "class_changed": prev["pred_class"] != classes[idx],
                      "confidence_delta": round(conf - prev["confidence"], 4),
                      "attention_area_delta": round(area - prev["attn_area"], 4)}
        db.add(con, pid, cancer, classes[idx], conf, area)

    return {"cancer": cancer, "predicted": classes[idx], "confidence": conf,
            "probabilities": {c: float(p) for c, p in zip(classes, probs)},
            "attention_area": area, "input_png": png_b64(arr), "gradcam_png": png_b64(over),
            "lime_png": lime_png, "change_vs_previous": change, "demo_models": registry.demo,
            "disclaimer": DISCLAIMER}


@app.get("/api/history")
def history(patient_id: str = Query(...), cancer: str = Query(None), limit: int = Query(50, le=200)):
    if not db.PID_RE.match(patient_id):
        raise HTTPException(400, "Invalid patient_id.")
    return {"records": db.history(con, patient_id, cancer, limit)}


@app.delete("/api/history/{patient_id}")
def delete_history(patient_id: str):
    if not db.PID_RE.match(patient_id):
        raise HTTPException(400, "Invalid patient_id.")
    return {"deleted": db.delete(con, patient_id)}


app.mount("/", StaticFiles(directory=Path(__file__).resolve().parent.parent / "frontend", html=True), name="ui")
