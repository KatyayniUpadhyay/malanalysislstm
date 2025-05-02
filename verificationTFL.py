import numpy as np
import tensorflow as tf

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="keras_lstm_model.tflite")
interpreter.allocate_tensors()

# Check input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("Input details:", input_details)
print("Output details:", output_details)

# Test input data (ensure shape matches input_details)
input_shape = input_details[0]['shape']
test_input = np.random.rand(*input_shape).astype(np.float32)

# Run inference
interpreter.set_tensor(input_details[0]['index'], test_input)
interpreter.invoke()
output = interpreter.get_tensor(output_details[0]['index'])
print("Output from TensorFlow Lite model:", output)
