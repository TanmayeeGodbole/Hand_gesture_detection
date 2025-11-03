import tensorflow as tf

# Load your .h5 model
model = tf.keras.models.load_model(
    "Real-Time-Sign-Language-Detection-Complete-Machine-Learning-Project-Tutorial-main/model.h5"
)

# Create converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable TensorFlow ops to handle LSTM dynamic operations
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # Default TFLite ops
    tf.lite.OpsSet.SELECT_TF_OPS     # Allow TF ops for unsupported ops like dynamic LSTM
]

# Optional: disable lowering tensor list ops
converter._experimental_lower_tensor_list_ops = False

# Convert to TFLite
tflite_model = converter.convert()

# Save the TFLite model
tflite_file = "Real-Time-Sign-Language-Detection-Complete-Machine-Learning-Project-Tutorial-main.tflite"
with open(tflite_file, "wb") as f:
    f.write(tflite_model)

print(f" TFLite model saved at '{tflite_file}'")
