import tensorflow as tf
import numpy as np
from transformers import WhisperProcessor, TFWhisperForConditionalGeneration
import os

# --- Configuration ---
# The original model name on Hugging Face
MODEL_NAME = "openai/whisper-tiny"
# The local directory where the original model and the TFLite model will be saved.
# Using a raw string (r"...") is good practice for Windows paths.
LOCAL_MODEL_DIR = r"C:\Users\elish\Desktop\whisper-Tensorflowlite"
# The final path for the converted TFLite file
TFLITE_MODEL_PATH = os.path.join(LOCAL_MODEL_DIR, "whisper_tiny_quantized.tflite")

# --- Step 0: Create the directory if it doesn't exist ---
print(f"--- 0. Ensuring directory exists ---")
os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
print(f"Files will be saved in: {LOCAL_MODEL_DIR}")


print("\n--- 1. Downloading and Saving Original Model ---")
# This block will run once to download and save the model.
# If the files already exist in LOCAL_MODEL_DIR, you could comment this part out.
try:
    # Load processor and model from the Hugging Face Hub
    print(f"Downloading '{MODEL_NAME}' from Hugging Face Hub...")
    processor_hub = WhisperProcessor.from_pretrained(MODEL_NAME)
    model_hub = TFWhisperForConditionalGeneration.from_pretrained(MODEL_NAME, from_pt=True)

    # Save the processor and model files to your local directory
    processor_hub.save_pretrained(LOCAL_MODEL_DIR)
    model_hub.save_pretrained(LOCAL_MODEL_DIR)
    print(f"✅ Model and processor saved successfully to {LOCAL_MODEL_DIR}")
except Exception as e:
    print(f"Could not download model. Maybe you are offline? Error: {e}")


print("\n--- 2. Loading Model from Local Directory for Conversion ---")
# Now, we load the model from the local path we just saved it to.
processor = WhisperProcessor.from_pretrained(LOCAL_MODEL_DIR)
model = TFWhisperForConditionalGeneration.from_pretrained(LOCAL_MODEL_DIR)
print("✅ Model loaded successfully from local files.")


print("\n--- 3. Creating a Dummy Input for Tracing ---")
# Create a dummy input with the correct shape for the model
dummy_input_features = np.zeros((1, 80, 3000), dtype=np.float32)
print(f"Dummy input shape: {dummy_input_features.shape}")


print("\n--- 4. Creating the TensorFlow Module for Conversion ---")
class WhisperModelModule(tf.Module):
    def __init__(self, model_to_wrap):
        super(WhisperModelModule, self).__init__()
        self.model = model_to_wrap

    @tf.function(input_signature=[tf.TensorSpec(shape=(1, 80, 3000), dtype=tf.float32)])
    def serving_default(self, input_features):
        return self.model.generate(input_features)

tf_module = WhisperModelModule(model)


print("\n--- 5. Converting the Model to TensorFlow Lite ---")
concrete_func = tf_module.serving_default.get_concrete_function()
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

# Save the TFLite model to the specified path
with open(TFLITE_MODEL_PATH, 'wb') as f:
    f.write(tflite_model)
print(f"\n--- Conversion Complete! ---")
print(f"✅ TFLite model saved to: {TFLITE_MODEL_PATH}")


print("\n--- 6. Verifying the TFLite Model ---")
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
interpreter.set_tensor(input_details[0]['index'], dummy_input_features)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])
transcription = processor.batch_decode(output_data, skip_special_tokens=True)

print(f"\n--- Verification Result ---")
print(f"Decoded output from silent dummy input: {transcription}")
print("\n🚀 Script finished successfully!")