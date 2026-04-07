"""
Phase 2 — Compare Traditional CV Pipeline vs CNN (MobileNetV2) on the test set.

Runs every test image through both classifiers and prints:
  - Per-class accuracy
  - Overall accuracy
  - Per-class F1 score
  - Side-by-side confusion matrices

Usage:
    python compare.py
"""

import os
import sys
import cv2
import math
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
)
import matplotlib.pyplot as plt

# ── PATHS ───────────────────────────────────────────────────
BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
TEST_DIR  = os.path.join(BASE_DIR, "data", "rps-test-set")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "rps_mobilenet.keras")

CLASS_NAMES = ["paper", "rock", "scissors"]
IMG_SIZE = (224, 224)

# ════════════════════════════════════════════════════════════
#  TRADITIONAL CV PIPELINE  (same logic as phase1_cv)
# ════════════════════════════════════════════════════════════

def build_skin_mask(roi_bgr):
    blur = cv2.GaussianBlur(roi_bgr, (5, 5), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, (0, 30, 50), (25, 255, 255))
    mask2 = cv2.inRange(hsv, (160, 30, 50), (179, 255, 255))
    mask = cv2.bitwise_or(mask1, mask2)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=1)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    return mask


def count_defects(contour, hull_indices):
    if hull_indices is None or len(hull_indices) < 3:
        return 0
    defects = cv2.convexityDefects(contour, hull_indices)
    if defects is None:
        return 0
    peri = cv2.arcLength(contour, True)
    if peri < 1:
        return 0
    min_depth = max(12.0, peri * 0.02)
    count = 0
    for i in range(defects.shape[0]):
        s, e, f, d = defects[i, 0]
        depth = d / 256.0
        if depth < min_depth:
            continue
        start = contour[s][0]
        end   = contour[e][0]
        far   = contour[f][0]
        a = np.hypot(end[0] - start[0], end[1] - start[1])
        b = np.hypot(far[0] - start[0], far[1] - start[1])
        c = np.hypot(end[0] - far[0],   end[1] - far[1])
        if b < 1 or c < 1:
            continue
        cos_angle = np.clip((b*b + c*c - a*a) / (2*b*c), -1.0, 1.0)
        angle_deg = math.degrees(math.acos(cos_angle))
        if angle_deg < 100:
            count += 1
    return count


def classify_traditional(image_bgr):
    """Classify a single BGR image using the traditional CV pipeline.
    Returns predicted class name (lowercase) or 'unknown'."""
    mask = build_skin_mask(image_bgr)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return "unknown"
    hand = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(hand)
    h, w = image_bgr.shape[:2]
    if area < max(1500, h * w * 0.03):
        return "unknown"
    hull_idx = cv2.convexHull(hand, returnPoints=False)
    defect_count = count_defects(hand, hull_idx)
    if defect_count == 0:
        return "rock"
    if defect_count <= 3:
        return "scissors"
    return "paper"

# ════════════════════════════════════════════════════════════
#  CNN PIPELINE
# ════════════════════════════════════════════════════════════

print("Loading CNN model...")
cnn_model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded.\n")


def classify_cnn(image_bgr):
    """Classify a single BGR image using the MobileNetV2 CNN.
    Returns predicted class name (lowercase)."""
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, IMG_SIZE)
    img_array = np.expand_dims(resized.astype(np.float32), axis=0)
    preds = cnn_model(img_array, training=False).numpy()[0]
    idx = int(np.argmax(preds))
    return CLASS_NAMES[idx]

# ════════════════════════════════════════════════════════════
#  RUN COMPARISON
# ════════════════════════════════════════════════════════════

def load_test_set():
    images = []
    labels = []
    for cls in CLASS_NAMES:
        cls_dir = os.path.join(TEST_DIR, cls)
        if not os.path.isdir(cls_dir):
            print(f"Warning: {cls_dir} not found, skipping.")
            continue
        for fname in sorted(os.listdir(cls_dir)):
            fpath = os.path.join(cls_dir, fname)
            img = cv2.imread(fpath)
            if img is not None:
                images.append(img)
                labels.append(cls)
    return images, labels


def main():
    images, y_true = load_test_set()
    n = len(images)
    print(f"Loaded {n} test images across {len(CLASS_NAMES)} classes.\n")

    y_pred_cv  = []
    y_pred_cnn = []

    for i, img in enumerate(images):
        y_pred_cv.append(classify_traditional(img))
        y_pred_cnn.append(classify_cnn(img))
        if (i + 1) % 50 == 0 or i == n - 1:
            print(f"  Processed {i + 1}/{n}")

    # Filter out "unknown" for traditional CV (it has no matching label)
    # For fair comparison, mark unknowns as wrong by keeping them
    # but we'll note how many there were.
    n_unknown = y_pred_cv.count("unknown")

    print(f"\n{'=' * 60}")
    print("TRADITIONAL CV PIPELINE")
    print(f"{'=' * 60}")
    print(f"  Unknown / no detection: {n_unknown}/{n} images")
    # Replace unknowns with a wrong class for metrics
    y_cv_clean = [p if p != "unknown" else "none" for p in y_pred_cv]
    all_labels = CLASS_NAMES + (["none"] if n_unknown > 0 else [])
    print(classification_report(y_true, y_cv_clean, labels=CLASS_NAMES,
                                target_names=CLASS_NAMES, zero_division=0))
    acc_cv = accuracy_score(y_true, y_cv_clean)
    print(f"  Overall accuracy: {acc_cv:.4f}\n")

    print(f"{'=' * 60}")
    print("CNN (MobileNetV2)")
    print(f"{'=' * 60}")
    print(classification_report(y_true, y_pred_cnn, target_names=CLASS_NAMES,
                                zero_division=0))
    acc_cnn = accuracy_score(y_true, y_pred_cnn)
    print(f"  Overall accuracy: {acc_cnn:.4f}\n")

    # ── SIDE-BY-SIDE CONFUSION MATRICES ─────────────────────
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    cm_cv = confusion_matrix(y_true, y_cv_clean, labels=CLASS_NAMES)
    ConfusionMatrixDisplay(cm_cv, display_labels=CLASS_NAMES).plot(
        ax=ax1, cmap="Oranges", values_format="d")
    ax1.set_title(f"Traditional CV  (acc={acc_cv:.2%})")

    cm_cnn = confusion_matrix(y_true, y_pred_cnn, labels=CLASS_NAMES)
    ConfusionMatrixDisplay(cm_cnn, display_labels=CLASS_NAMES).plot(
        ax=ax2, cmap="Blues", values_format="d")
    ax2.set_title(f"CNN / MobileNetV2  (acc={acc_cnn:.2%})")

    out_path = os.path.join(MODEL_DIR, "comparison.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Comparison chart saved to {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
