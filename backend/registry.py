"""Loads per-cancer models lazily from MODELS_DIR using class_names.json."""
import json
import threading
from pathlib import Path

from tensorflow import keras


class Registry:
    def __init__(self, models_dir):
        self.dir = Path(models_dir)
        self._models = {}
        self._lock = threading.Lock()
        meta_path = self.dir / "class_names.json"
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"demo": False, "cancers": {}}
        self.demo = bool(meta.get("demo"))
        self.cancers = meta.get("cancers", {})

    @staticmethod
    def filename(cancer):
        return cancer.replace(" ", "_") + ".keras"

    def info(self):
        return [{"name": c, "classes": v["classes"], "val_acc": v.get("val_acc"),
                 "available": (self.dir / self.filename(c)).exists()} for c, v in self.cancers.items()]

    def get(self, cancer):
        with self._lock:
            if cancer not in self._models:
                path = self.dir / self.filename(cancer)
                if not path.exists():
                    raise FileNotFoundError(path.name)
                self._models[cancer] = keras.models.load_model(path, compile=False)
            return self._models[cancer]
