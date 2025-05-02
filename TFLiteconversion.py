import tensorflow as tf

# Load your Keras model
model = tf.keras.models.load_model('keras_lstm_model.h5')  # or 'custom_lstm_model.h5'

# Convert the model to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# ✅ Add these lines:
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS, 
    tf.lite.OpsSet.SELECT_TF_OPS
]
converter._experimental_lower_tensor_list_ops = False
converter.experimental_enable_resource_variables = True

# Perform the conversion
tflite_model = converter.convert()

# Save the converted model
with open('compatible_keras_lstm_model.tflite', 'wb') as f:
    f.write(tflite_model)


