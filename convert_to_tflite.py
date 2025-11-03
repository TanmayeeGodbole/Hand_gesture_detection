import tensorflow as tf

# Convert SavedModel to TFLite
converter = tf.lite.TFLiteConverter.from_saved_model("gesture_classifier")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save compressed model
with open("gesture_classifier.tflite", "wb") as f:
    f.write(tflite_model)

print("Compressed model saved as gesture_classifier.tflite")
