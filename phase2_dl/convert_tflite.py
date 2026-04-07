"""Convert the trained Keras model to TFLite for fast CPU inference."""

import os
import tensorflow as tf

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
KERAS_PATH = os.path.join(BASE_DIR, "model", "rps_mobilenet.keras")
TFLITE_PATH = os.path.join(BASE_DIR, "model", "rps_mobilenet.tflite")

print("Loading Keras model...")
model = tf.keras.models.load_model(KERAS_PATH)

print("Converting to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open(TFLITE_PATH, "wb") as f:
    f.write(tflite_model)

size_mb = os.path.getsize(TFLITE_PATH) / (1024 * 1024)
print(f"Saved to {TFLITE_PATH} ({size_mb:.1f} MB)")
print("Done.")
