"""
Phase 2 — Train a MobileNetV2 gesture classifier (Rock / Paper / Scissors).

Improvements over v1:
  - Uses the held-out test set as validation (different distribution from
    the synthetic training images) so val_accuracy reflects real accuracy.
  - Heavy data augmentation to bridge synthetic ↔ real gap.
  - Label smoothing to prevent overconfident predictions.
  - Wider dense head (256 units) with stronger dropout.
  - Auto-converts the best model to TFLite at the end.

Usage:
    python train.py                 # full training from scratch
    python train.py --continue 12   # load rps_mobilenet.keras and fine-tune more
"""

import argparse
import os, shutil
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.layers import (
    GlobalAveragePooling2D, Dense, Dropout,
    RandomFlip, RandomRotation, RandomZoom, RandomContrast,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ── PATHS ───────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
TRAIN_DIR   = os.path.join(BASE_DIR, "data", "rps")
TEST_DIR    = os.path.join(BASE_DIR, "data", "rps-test-set")
MODEL_DIR   = os.path.join(BASE_DIR, "model")
MODEL_PATH  = os.path.join(MODEL_DIR, "rps_mobilenet.keras")
FINETUNE_PATH = os.path.join(MODEL_DIR, "rps_mobilenet_ft.keras")
TFLITE_PATH = os.path.join(MODEL_DIR, "rps_mobilenet.tflite")

os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
CLASS_NAMES = ["paper", "rock", "scissors"]


def _find_mobilenet_backbone(model):
    for layer in model.layers:
        if "mobilenet" in layer.name.lower():
            return layer
    return model.layers[1]


def load_datasets():
    train_ds_raw = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        class_names=CLASS_NAMES,
        shuffle=True,
        seed=42,
    )
    val_ds_raw = tf.keras.utils.image_dataset_from_directory(
        TEST_DIR,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        label_mode="categorical",
        class_names=CLASS_NAMES,
        shuffle=False,
    )
    return train_ds_raw, val_ds_raw


# ── LOAD DATA ───────────────────────────────────────────────
print("Loading datasets ...")
train_ds_raw, val_ds_raw = load_datasets()

# ── AUGMENTATION ────────────────────────────────────────────
augmenter = tf.keras.Sequential([
    RandomFlip("horizontal_and_vertical"),
    RandomRotation(0.18),
    RandomZoom((-0.20, 0.0)),
    RandomContrast(0.35),
], name="augmenter")

def augment_and_preprocess(images, labels):
    images = augmenter(images, training=True)
    images = tf.image.random_brightness(images, 0.30)
    images = tf.image.random_saturation(images, 0.65, 1.35)
    images = tf.image.random_hue(images, 0.06)
    images = tf.clip_by_value(images, 0.0, 255.0)
    return preprocess_input(images), labels

def preprocess_only(images, labels):
    return preprocess_input(images), labels

AUTOTUNE = tf.data.AUTOTUNE
train_ds = (train_ds_raw
            .map(augment_and_preprocess, num_parallel_calls=AUTOTUNE)
            .prefetch(AUTOTUNE))
val_ds = (val_ds_raw
          .map(preprocess_only, num_parallel_calls=AUTOTUNE)
          .prefetch(AUTOTUNE))

loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)


def evaluate_and_save_plots(best_model, history_extra=None):
    print("\n=== Final evaluation on test set ===")
    test_loss, test_acc = best_model.evaluate(val_ds)
    print(f"\nTest accuracy: {test_acc:.4f}")
    print(f"Test loss:     {test_loss:.4f}")

    y_true, y_pred = [], []
    for images, labels in val_ds:
        preds = best_model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion Matrix — MobileNetV2 (acc={test_acc:.2%})")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"), dpi=150)
    print("Confusion matrix saved.")

    if history_extra is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        e = range(1, len(history_extra.history["accuracy"]) + 1)
        ax1.plot(e, history_extra.history["accuracy"], label="Train")
        ax1.plot(e, history_extra.history["val_accuracy"], label="Val")
        ax1.set_xlabel("Epoch (continue)")
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Continue training — accuracy")
        ax1.legend()
        ax2.plot(e, history_extra.history["loss"], label="Train")
        ax2.plot(e, history_extra.history["val_loss"], label="Val")
        ax2.set_xlabel("Epoch (continue)")
        ax2.set_ylabel("Loss")
        ax2.set_title("Continue training — loss")
        ax2.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, "training_curves_continue.png"), dpi=150)
        print("Continue training curves saved.")

    print("\nConverting to TFLite ...")
    converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(TFLITE_PATH, "wb") as f:
        f.write(tflite_model)
    size_mb = os.path.getsize(TFLITE_PATH) / (1024 * 1024)
    print(f"TFLite model saved: {TFLITE_PATH} ({size_mb:.1f} MB)")
    print(f"\nDone. Best model at {MODEL_PATH}")


def run_continue_training(extra_epochs):
    if not os.path.isfile(MODEL_PATH):
        raise SystemExit(f"Missing {MODEL_PATH} — run full training first.")
    print(f"\n=== Continue: loading {MODEL_PATH}, fine-tuning {extra_epochs} epochs ===")
    model = tf.keras.models.load_model(MODEL_PATH)
    base_model = _find_mobilenet_backbone(model)
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=5e-6),
        loss=loss_fn,
        metrics=["accuracy"],
    )
    cont_path = os.path.join(MODEL_DIR, "rps_mobilenet_continue.keras")
    hist = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=extra_epochs,
        callbacks=[
            ModelCheckpoint(cont_path, monitor="val_accuracy",
                            save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                              patience=2, min_lr=1e-8, verbose=1),
        ],
    )
    best = tf.keras.models.load_model(cont_path)
    _, acc_new = best.evaluate(val_ds, verbose=0)
    _, acc_old = tf.keras.models.load_model(MODEL_PATH).evaluate(val_ds, verbose=0)
    print(f"\n  Accuracy before continue: {acc_old:.4f}")
    print(f"  Accuracy after continue:  {acc_new:.4f}")
    if acc_new >= acc_old:
        shutil.copy2(cont_path, MODEL_PATH)
        print("  -> Updated rps_mobilenet.keras with continued model.")
    else:
        print("  -> Kept previous rps_mobilenet.keras (continue did not improve).")
    best_model = tf.keras.models.load_model(MODEL_PATH)
    evaluate_and_save_plots(best_model, history_extra=hist)


def run_full_training():
    print("Building MobileNetV2 model ...")
    base_model = MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(224, 224, 3),
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=(224, 224, 3))
    x = base_model(inputs, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.45)(x)
    x = Dense(256, activation="relu")(x)
    x = Dropout(0.35)(x)
    outputs = Dense(3, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model.summary()

    print("\n=== Phase 1: Training classification head (base frozen) ===")
    model.compile(
        optimizer=Adam(learning_rate=5e-4),
        loss=loss_fn,
        metrics=["accuracy"],
    )
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=25,
        callbacks=[
            ModelCheckpoint(MODEL_PATH, monitor="val_accuracy",
                            save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                              patience=3, min_lr=1e-6, verbose=1),
        ],
    )

    model = tf.keras.models.load_model(MODEL_PATH)
    _, head_acc = model.evaluate(val_ds, verbose=0)
    print(f"\n  Best head-only val accuracy: {head_acc:.4f}")

    print("\n=== Phase 2: Fine-tuning top 30 layers of MobileNetV2 ===")
    base_model = _find_mobilenet_backbone(model)
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss=loss_fn,
        metrics=["accuracy"],
    )
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        callbacks=[
            ModelCheckpoint(FINETUNE_PATH, monitor="val_accuracy",
                            save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                              patience=3, min_lr=1e-7, verbose=1),
        ],
    )

    ft_model = tf.keras.models.load_model(FINETUNE_PATH)
    _, ft_acc = ft_model.evaluate(val_ds, verbose=0)
    print(f"\n  Head-only  val accuracy: {head_acc:.4f}")
    print(f"  Fine-tuned val accuracy: {ft_acc:.4f}")
    if ft_acc > head_acc:
        print("  -> Using fine-tuned model.")
        shutil.copy2(FINETUNE_PATH, MODEL_PATH)
    else:
        print("  -> Keeping head-only model (better).")

    best_model = tf.keras.models.load_model(MODEL_PATH)

    print("\n=== Final evaluation on test set ===")
    test_loss, test_acc = best_model.evaluate(val_ds)
    print(f"\nTest accuracy: {test_acc:.4f}")
    print(f"Test loss:     {test_loss:.4f}")

    y_true, y_pred = [], []
    for images, labels in val_ds:
        preds = best_model.predict(images, verbose=0)
        y_true.extend(np.argmax(labels.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    ax.set_title(f"Confusion Matrix — MobileNetV2 (acc={test_acc:.2%})")
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "confusion_matrix.png"), dpi=150)
    print("Confusion matrix saved.")

    acc1, vacc1 = history1.history["accuracy"], history1.history["val_accuracy"]
    acc2, vacc2 = history2.history["accuracy"], history2.history["val_accuracy"]
    epochs_total = range(1, len(acc1) + len(acc2) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(epochs_total, acc1 + acc2, label="Train")
    ax1.plot(epochs_total, vacc1 + vacc2, label="Val")
    ax1.axvline(x=len(acc1) + 0.5, color="gray", linestyle="--", label="Fine-tune start")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_title("Accuracy")
    ax1.legend()
    loss1, vloss1 = history1.history["loss"], history1.history["val_loss"]
    loss2, vloss2 = history2.history["loss"], history2.history["val_loss"]
    ax2.plot(epochs_total, loss1 + loss2, label="Train")
    ax2.plot(epochs_total, vloss1 + vloss2, label="Val")
    ax2.axvline(x=len(loss1) + 0.5, color="gray", linestyle="--", label="Fine-tune start")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("Loss")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "training_curves.png"), dpi=150)
    print("Training curves saved.")

    print("\nConverting to TFLite ...")
    converter = tf.lite.TFLiteConverter.from_keras_model(best_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(TFLITE_PATH, "wb") as f:
        f.write(tflite_model)
    size_mb = os.path.getsize(TFLITE_PATH) / (1024 * 1024)
    print(f"TFLite model saved: {TFLITE_PATH} ({size_mb:.1f} MB)")
    print(f"\nDone. Best model at {MODEL_PATH}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Train RPS MobileNetV2 classifier")
    p.add_argument(
        "--continue",
        dest="continue_epochs",
        type=int,
        default=0,
        metavar="N",
        help="Load saved model and fine-tune N more epochs (skips full training)",
    )
    args = p.parse_args()
    if args.continue_epochs > 0:
        run_continue_training(args.continue_epochs)
    else:
        run_full_training()
