"""Grad-CAM and LIME helpers. All functions take a raw 0-255 RGB array of shape (160, 160, 3)."""
import base64
import io

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

from .model_def import HEAD_LAYERS


def gradcam(model, img, class_idx=None):
    """Return (cam in [0,1] at feature-map resolution, class index)."""
    base = next(l for l in model.layers if l.name.startswith("mobilenetv2"))
    conv_model = keras.Model(base.inputs, [base.get_layer("out_relu").output, base.output])
    head = [model.get_layer(n) for n in HEAD_LAYERS]
    x = preprocess_input(tf.convert_to_tensor(img[None].astype("float32")))
    with tf.GradientTape() as tape:
        conv, feat = conv_model(x, training=False)
        y = feat
        for layer in head:
            y = layer(y, training=False)
        if class_idx is None:
            class_idx = int(tf.argmax(y[0]))
        score = y[:, class_idx]
    grads = tape.gradient(score, conv)
    w = tf.reduce_mean(grads, axis=(0, 1, 2))
    cam = tf.nn.relu(tf.reduce_sum(conv[0] * w, axis=-1)).numpy()
    return cam / (cam.max() + 1e-12), class_idx


def overlay(img, cam, alpha=0.4):
    """Blend a jet heatmap on the image. Returns (overlay uint8, full-size cam)."""
    h = cv2.resize(cam, (img.shape[1], img.shape[0]))
    color = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * h), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return np.uint8(color * alpha + img * (1 - alpha)), h


def lime_overlay(model, img, class_idx, num_samples=300):
    """LIME superpixel boundaries for the given class. Slower than Grad-CAM."""
    from lime import lime_image
    from skimage.segmentation import mark_boundaries

    exp = lime_image.LimeImageExplainer().explain_instance(
        np.uint8(img).astype("double"),
        lambda b: model.predict(b.astype("float32"), verbose=0),
        labels=(class_idx,), hide_color=0, num_samples=num_samples, batch_size=64, random_seed=42)
    tmp, mask = exp.get_image_and_mask(class_idx, positive_only=True, num_features=5, hide_rest=False)
    return np.uint8(mark_boundaries(tmp / 255.0, mask) * 255)


def png_b64(arr):
    buf = io.BytesIO()
    Image.fromarray(np.uint8(arr)).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()
