"""Model definition shared by training and serving (must stay identical in both)."""
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import MobileNetV2

IMG_SIZE = (160, 160)
HEAD_LAYERS = ("gap", "drop1", "fc128", "drop2", "probs")


def build_model(num_classes: int, weights="imagenet"):
    """MobileNetV2 (frozen by default) + small head. Input is a raw 0-255 RGB image."""
    base = MobileNetV2(include_top=False, weights=weights, input_shape=(*IMG_SIZE, 3))
    base.trainable = False
    aug = keras.Sequential(
        [layers.RandomFlip("horizontal"), layers.RandomRotation(0.05), layers.RandomZoom(0.1)], name="aug")
    inp = keras.Input((*IMG_SIZE, 3))
    x = aug(inp)  # active during training only
    x = layers.Rescaling(1 / 127.5, offset=-1, name="prep")(x)  # same as MobileNetV2 preprocess_input
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dropout(0.3, name="drop1")(x)
    x = layers.Dense(128, activation="relu", name="fc128")(x)
    x = layers.Dropout(0.2, name="drop2")(x)
    out = layers.Dense(num_classes, activation="softmax", name="probs")(x)
    return keras.Model(inp, out), base
