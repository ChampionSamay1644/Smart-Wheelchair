from transformers import pipeline, AutoModelForSpeechSeq2Seq, AutoProcessor, AutoConfig, GenerationConfig
import torch
import torch.quantization # Import the quantization module
import os
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav


# Define the model ID and the local cache directory
model_id = "openai/whisper-small"
cache_directory = "./models"


# Force CPU usage for Raspberry Pi
device = "cpu"
torch_dtype = torch.float32


# --- Model Loading and Quantization Setup ---

# 1. Load the base model configuration and the generation configuration
# The generation_config is crucial for tasks like timestamp generation.
optimized_dir = "./optimized_models"
weights_path = os.path.join(optimized_dir, "whisper_small_weights_cpu_int8.pt")
config = AutoConfig.from_pretrained(model_id, cache_dir=cache_directory)
generation_config = GenerationConfig.from_pretrained(model_id, cache_dir=cache_directory)

# 2. Create the model from the configuration and attach the generation config
model = AutoModelForSpeechSeq2Seq.from_config(config)
model.generation_config = generation_config

# 3. Apply dynamic quantization to the model structure.
# This converts the standard nn.Linear layers to their quantized counterparts,
# making the model's structure match your saved weights file.
print("Applying quantization structure to the model...")
model_quantized = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)

# 4. Now, load your pre-quantized weights into the correctly structured model.
# The keys in the state_dict should now perfectly match the layers in `model_quantized`.
print(f"Loading quantized model weights from {weights_path}...")
model_quantized.load_state_dict(torch.load(weights_path, map_location="cpu"))
model = model_quantized # Use the quantized model for the pipeline
model.eval()
print("Successfully loaded quantized model.")

# 5. Load the processor (which includes the feature extractor and tokenizer)
# This will also be saved to the cache directory
processor = AutoProcessor.from_pretrained(model_id, cache_dir=cache_directory)


# 6. Now, create the pipeline using the loaded quantized model
# The model now has the correct generation_config, so timestamps will work.
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=4,  # Lower batch size for Pi
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device, # Explicitly set the device for the pipeline
)



# Transcribe a given audio file
def transcribe_audio_file(audio_path):
    # Check if the file exists before trying to transcribe
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        # Create a dummy wav file for demonstration if it doesn't exist
        print("Creating a dummy 'voicemodel.wav' for testing purposes.")
        sample_rate = 16000  # Whisper expects 16kHz
        duration = 5  # seconds
        frequency = 440  # Hz
        t = np.linspace(0., duration, int(sample_rate * duration), endpoint=False)
        amplitude = np.iinfo(np.int16).max * 0.5
        data = amplitude * np.sin(2. * np.pi * frequency * t)
        wav.write(audio_path, sample_rate, data.astype(np.int16))
        print(f"Dummy file created. Please replace it with your actual audio.")


    print(f"Transcribing file: {audio_path}")
    # The pipeline expects the raw audio data, not just the path for this setup
    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    # --- FIX APPLIED HERE ---
    # To get the output translated to English, we need to set the task to "translate".
    # This tells the Whisper model to translate the audio from any detected language into English.
    result = pipe(audio_bytes, generate_kwargs={"task": "translate"})
    # result = pipe(audio_bytes, generate_kwargs={"task": "transcribe"})
    
    print("Transcription:", result["text"])

if __name__ == "__main__":
    # Ensure the target directory exists
    if not os.path.exists(optimized_dir):
        os.makedirs(optimized_dir)

    # Note: This script assumes 'whisper_small_weights_cpu_int8.pt' exists.
    # You can now test with your non-English audio files.
    # For example:
    # transcribe_audio_file("voicemodel2.wav")
    transcribe_audio_file("voicemix.wav")

