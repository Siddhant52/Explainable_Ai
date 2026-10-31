"""Create 8 RANDOM-weight models so the UI/API can be tried before real training.
Their predictions are meaningless. Usage: python -m scripts.make_dummy_models [models_dir]"""
import json
import sys
from pathlib import Path

from backend.model_def import build_model

CLASS_COUNTS = {"ALL": 4, "Brain Cancer": 3, "Breast Cancer": 2, "Cervical Cancer": 5,
                "Kidney Cancer": 2, "Lung and Colon Cancer": 5, "Lymphoma": 3, "Oral Cancer": 2}


def main(out="models"):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    meta = {"demo": True, "cancers": {}}
    for name, n in CLASS_COUNTS.items():
        model, _ = build_model(n, weights=None)
        model.save(out / (name.replace(" ", "_") + ".keras"))
        meta["cancers"][name] = {"classes": [f"class_{i + 1}" for i in range(n)], "val_acc": None}
    (out / "class_names.json").write_text(json.dumps(meta, indent=1))
    print("Wrote demo models to", out)


if __name__ == "__main__":
    main(*sys.argv[1:2])
