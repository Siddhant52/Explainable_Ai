"""Train one model per cancer type and export everything the API needs.

Usage:
  python -m training.train_all --data-root "/path/Multi Cancer/Multi Cancer" --out models
Options: --no-fine-tune (frozen backbone only, as in the first paper draft), --only "Oral Cancer"
Output: models/<Cancer_Type>.keras, models/class_names.json, models/results.json
"""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow import keras
from tensorflow.keras import layers

from backend.model_def import IMG_SIZE, build_model

SEED, BATCH, VAL_SPLIT = 42, 64, 0.2


def datasets(folder):
    kw = dict(validation_split=VAL_SPLIT, seed=SEED, image_size=IMG_SIZE, batch_size=BATCH, label_mode="categorical")
    tr = keras.utils.image_dataset_from_directory(folder, subset="training", **kw)
    va = keras.utils.image_dataset_from_directory(folder, subset="validation", **kw)
    return tr.cache().prefetch(tf.data.AUTOTUNE), va.cache().prefetch(tf.data.AUTOTUNE), tr.class_names


def cbs(patience):
    return [keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, min_lr=1e-7)]


def fit(tr, va, n, args):
    model, base = build_model(n, weights=args.weights)
    model.compile(keras.optimizers.Adam(1e-3), "categorical_crossentropy", metrics=["accuracy"])
    h = model.fit(tr, validation_data=va, epochs=args.epochs_head, callbacks=cbs(2), verbose=2)
    hist = {k: list(v) for k, v in h.history.items()}
    if args.fine_tune:
        base.trainable = True
        for l in base.layers[:args.unfreeze_from]:
            l.trainable = False
        for l in base.layers:
            if isinstance(l, layers.BatchNormalization):
                l.trainable = False
        model.compile(keras.optimizers.Adam(args.ft_lr), "categorical_crossentropy", metrics=["accuracy"])
        h2 = model.fit(tr, validation_data=va, epochs=args.epochs_ft, callbacks=cbs(3), verbose=2)
        for k, v in h2.history.items():
            hist.setdefault(k, []).extend(v)
    return model, hist


def evaluate(model, va, names):
    yt, yp = [], []
    for x, y in va:
        yp.extend(np.argmax(model.predict(x, verbose=0), 1)); yt.extend(np.argmax(y.numpy(), 1))
    yt, yp = np.array(yt), np.array(yp)
    return dict(acc=float((yt == yp).mean()), classes=list(names), cm=confusion_matrix(yt, yp).tolist(),
                report=classification_report(yt, yp, target_names=names, output_dict=True))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True); p.add_argument("--out", default="models")
    p.add_argument("--only", default=None); p.add_argument("--weights", default="imagenet")
    p.add_argument("--epochs-head", type=int, default=8); p.add_argument("--epochs-ft", type=int, default=12)
    p.add_argument("--unfreeze-from", type=int, default=100); p.add_argument("--ft-lr", type=float, default=5e-5)
    p.add_argument("--no-fine-tune", dest="fine_tune", action="store_false")
    args = p.parse_args()
    args.weights = None if args.weights == "none" else args.weights
    random.seed(SEED); np.random.seed(SEED); tf.random.set_seed(SEED)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    res_path, meta_path = out / "results.json", out / "class_names.json"
    results = json.loads(res_path.read_text()) if res_path.exists() else {}
    types = sorted(d for d in os.listdir(args.data_root) if (Path(args.data_root) / d).is_dir())
    if args.only:
        types = [t for t in types if t == args.only]
    for cancer in types:
        if cancer in results:
            print("skip (done):", cancer); continue
        tr, va, names = datasets(str(Path(args.data_root) / cancer))
        model, hist = fit(tr, va, len(names), args)
        r = evaluate(model, va, names); r["history"] = hist
        results[cancer] = r
        model.save(out / (cancer.replace(" ", "_") + ".keras"))
        res_path.write_text(json.dumps(results))
        meta = {"demo": False, "cancers": {c: {"classes": v["classes"], "val_acc": v["acc"]} for c, v in results.items()}}
        meta_path.write_text(json.dumps(meta, indent=1))
        print(f"{cancer}: validation accuracy {r['acc'] * 100:.2f}%")


if __name__ == "__main__":
    main()
