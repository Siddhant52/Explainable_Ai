# Multi-Cancer Explainable Screening (project implementation)

Implements the paper "Explainable AI-Based Cancer Risk Prediction and Monitoring System": eight per-cancer
MobileNetV2 classifiers, each with Grad-CAM and LIME explanations, behind a web app with a simple monitoring log.

**Research prototype. Not a medical device. Not for clinical decisions.**

## Layout
```
backend/    FastAPI service (app.py), model definition, Grad-CAM/LIME (xai.py), SQLite log (db.py)
frontend/   index.html, served by the backend at /
training/   train_all.py (CLI) and multi_cancer_improved.ipynb (Kaggle notebook)
scripts/    make_dummy_models.py (random-weight models to try the UI)
tests/      API tests (pytest)
models/     put <Cancer_Type>.keras and class_names.json here
```

## 1. Install
```
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
Use the same TensorFlow/Keras major version that trained the models (the `.keras` format).

## 2. Get models (choose one)
**A. Real models (Kaggle).** Run `training/multi_cancer_improved.ipynb` on Kaggle with the Multi Cancer Dataset and a GPU.
Download `/kaggle/working/models/` and copy its contents into `models/` (it must contain the eight `.keras` files and `class_names.json`).

**B. Real models (local/CLI).**
```
python -m training.train_all --data-root "/path/to/Multi Cancer/Multi Cancer" --out models
```
Add `--no-fine-tune` for a frozen backbone only. Results are written to `models/results.json`.

**C. Demo only.** `python -m scripts.make_dummy_models` creates random models so you can try the interface.
The app shows a banner and the predictions are meaningless.

## 3. Run
```
uvicorn backend.app:app --host 127.0.0.1 --port 8000
```
Open http://127.0.0.1:8000 . Environment variables: `MODELS_DIR` (default `models`), `DB_PATH` (default `data/monitoring.db`).

## API
| Method | Path | Purpose |
|---|---|---|
| GET | /api/health | status, demo flag |
| GET | /api/cancers | cancer types, classes, validation accuracy |
| POST | /api/predict | form fields: `file`, `cancer`, optional `patient_id`, `lime` (true/false). Returns probabilities, Grad-CAM overlay, optional LIME, change vs previous visit |
| GET | /api/history?patient_id=... | monitoring records |
| DELETE | /api/history/{patient_id} | delete a patient's records |

## Tests
`pytest -q tests` (uses random demo models; checks the API, explanations, monitoring log and input validation).

## What is and is not verified
- Tested here on dummy data: model building, two-stage training, Grad-CAM, LIME, API, monitoring log, training script output loading in the API.
- **Not** tested here: training on the real dataset, so real accuracy is unknown until you run it.
- The monitoring log compares confidence and Grad-CAM attention area between visits. It is a prototype of the paper's future-work idea and has not been validated on longitudinal data (the dataset has none).
- The Multi Cancer Dataset contains augmented copies of a smaller set of originals, so a random 80/20 split can leak near-duplicates into validation and inflate accuracy. Check this before quoting results.

## Privacy and security notes
- Use pseudonymous patient IDs only. Images are processed in memory and are not stored.
- There is no login. Before any real deployment add authentication, HTTPS, access control and audit logs, and follow the data-protection rules that apply to you.
