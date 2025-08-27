# ==============================================================================
# Piper TTS - Multi-Voice Download and Synthesize Script (Final)
#
# Description:
# This script automates setting up and running Piper TTS for multiple voices
# across different languages. It performs the following steps:
# 1. Downloads specified Piper TTS models (.onnx and .json files) into a
#    './models' directory.
# 2. Runs a text-to-speech synthesis test for each voice using the
#    downloaded models directly on the CPU.
# 3. Saves the output .wav files to './tts_audio_outputs'.
#
# NOTE: This version removes the model quantization step, as it was creating
# models incompatible with the Piper runtime, causing errors. The original
# Piper models are already well-optimized for CPU usage.
#
# Author: Gemini
# ==============================================================================

import os
import urllib.request
import subprocess
import sys

# --- Configuration ---

# Define the models to be downloaded and processed.
# Added one male and one female voice for both English and Hindi.
MODELS_TO_PROCESS = [
    {
        "name": "en_US-lessac-medium",
        "language": "en",
        "gender": "female",
        "onnx_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
        "json_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
    },
    {
        "name": "en_US-ryan-medium",
        "language": "en",
        "gender": "male",
        "onnx_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx",
        "json_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json"
    },
    {
        "name": "hi_IN-priyamvada-medium",
        "language": "hi",
        "gender": "female",
        "onnx_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/priyamvada/medium/hi_IN-priyamvada-medium.onnx",
        "json_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/priyamvada/medium/hi_IN-priyamvada-medium.onnx.json"
    },
    {
        "name": "hi_IN-rohan-medium",
        "language": "hi",
        "gender": "male",
        "onnx_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/rohan/medium/hi_IN-rohan-medium.onnx",
        "json_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/rohan/medium/hi_IN-rohan-medium.onnx.json"
    },
]

# Define test sentences for each language.
TEST_SENTENCES = {
    "en": "The afternoon sun filtered through the dense canopy of trees, creating shifting patterns of light and shadow on the forest floor. A gentle breeze rustled the leaves, creating a soft, whispering sound that was the only interruption to the peaceful silence. The air was filled with the rich, earthy smell of damp soil and wild ferns, a perfume unique to the woods. With every step on the winding path, the worries of the busy world seemed to fade away, replaced by a profound sense of calm. It’s in these moments of quiet solitude that you can truly hear your own thoughts and appreciate the simple beauty of the world around you.",
    "hi": "शाम का वक्त था, सूरज धीरे-धीरे क्षितिज के नीचे जा रहा था। आसमान में नारंगी, गुलाबी और बैंगनी रंगों की एक खूबसूरत चादर फैल गई थी। दिनभर की भागदौड़ अब शांत हो रही थी, और पंछियों के चहचहाने की आवाज़ें घर लौटते लोगों के शोर में मिल रही थीं। बालकनी में खड़े होकर इस दृश्य को देखना एक सुकून भरा अनुभव था। हाथ में चाय का गर्म कप इस पल को और भी खास बना रहा था। कभी-कभी, जीवन की सबसे बड़ी खुशियाँ ऐसे ही छोटे-छोटे, शांत लम्हों में छिपी होती हैं।",
    "mr": "आज सकाळपासूनच आभाळ भरून आलं होतं. ढगांची गर्दी आणि थंडगार वारा, वातावरणात एक वेगळाच शांतपणा घेऊन आला होता. दुपारच्या सुमारास पावसाच्या सरींनी जोर धरला. खिडकीवर पडणाऱ्या थेंबांचा तो लयबद्ध आवाज मनाला सुखावत होता आणि ओल्या मातीचा सुगंध तर थेट बालपणीच्या आठवणींमध्ये घेऊन जात होता. अशा वेळी गरमागरम चहा आणि सोबत गरमागरम कांदा भजी खाण्याची मजाच काही और असते. पाऊस म्हणजे फक्त पाणी नाही, तर तो आपल्यासोबत अनेक आठवणी आणि भावना घेऊन येतो."
}

# Define directory paths.
MODELS_DIR = "./models"
OUTPUT_DIR = "./tts_audio_outputs"

# --- Helper Functions ---

def setup_directories():
    """Creates the necessary directories if they don't exist."""
    print("--- Setting up directories ---")
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Directories '{MODELS_DIR}' and '{OUTPUT_DIR}' are ready.")

def download_file(url, save_path):
    """Downloads a file from a URL to a specified path if it doesn't already exist."""
    if not os.path.exists(save_path):
        print(f"Downloading {os.path.basename(save_path)}...")
        try:
            urllib.request.urlretrieve(url, save_path)
            print("Download complete.")
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            return False
    else:
        print(f"{os.path.basename(save_path)} already exists. Skipping download.")
    return True

def synthesize_speech(model_name, language, gender, text):
    """
    Generates audio from text using the original Piper TTS model.
    """
    print(f"\n--- Synthesizing speech for {language.upper()} ({gender}) ---")
    model_path = os.path.join(MODELS_DIR, f"{model_name}.onnx")
    config_path = os.path.join(MODELS_DIR, f"{model_name}.onnx.json")
    output_wav_path = os.path.join(OUTPUT_DIR, f"output_{language}_{gender}.wav")

    if not os.path.exists(model_path) or not os.path.exists(config_path):
        print(f"Error: Model or config file not found for {model_name}. Cannot synthesize.")
        return

    print(f"Using model: {model_path}")
    print(f"Text: {text}")

    # Construct the command to pipe the text into Piper.
    # The piper command automatically looks for the .json file in the same
    # directory as the .onnx model, so we only need to specify the model path.
    command = (
        f'echo "{text}" | '
        f'piper --model "{model_path}" '
        f'--output_file "{output_wav_path}"'
    )

    try:
        # Execute the command in the shell.
        subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"Successfully generated audio file: {output_wav_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error running Piper for {language} ({gender}):")
        print(f"Command failed with exit code {e.returncode}")
        print(f"Stderr: {e.stderr.strip()}")
    except FileNotFoundError:
        print("Error: 'piper' command not found. Please ensure 'piper-tts' is installed.")

# --- Main Execution ---

def main():
    """
    Main function to run the entire download and test pipeline.
    """
    setup_directories()

    # Process each model defined in the configuration.
    for model_info in MODELS_TO_PROCESS:
        model_name = model_info["name"]
        print(f"\n--- Processing model: {model_name} ---")

        # Download model and config files.
        onnx_path = os.path.join(MODELS_DIR, f"{model_name}.onnx")
        json_path = os.path.join(MODELS_DIR, f"{model_name}.onnx.json")
        download_success = download_file(model_info["onnx_url"], onnx_path)
        if download_success:
            download_file(model_info["json_url"], json_path)

    # Run synthesis tests for each downloaded voice.
    for model_info in MODELS_TO_PROCESS:
        lang = model_info["language"]
        if lang in TEST_SENTENCES:
            model_path = os.path.join(MODELS_DIR, f"{model_info['name']}.onnx")
            if os.path.exists(model_path):
                synthesize_speech(
                    model_info["name"],
                    lang,
                    model_info["gender"],
                    TEST_SENTENCES[lang]
                    # TEST_SENTENCES["mr"]
                )
            else:
                print(f"\nSkipping synthesis for {model_info['name']} because model files were not found.")

    
    # synthesize_speech(
    #     "hi_IN-priyamvada-medium",
    #     "hi",
    #     "female",
    #     TEST_SENTENCES["mr"]

    print("\n\nScript finished. Check 'tts_audio_outputs' for the generated .wav files.")

if __name__ == "__main__":
    main()
