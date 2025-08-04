# ==============================================================================
# Text-to-Speech with Voice Cloning using Coqui XTTS
#
# Description:
# This script demonstrates how to use the Coqui Text-to-Speech (TTS) library
# with the advanced XTTS model to clone voices from local audio files.
#
# Requirements:
# 1. Install the Coqui TTS library: `pip install TTS`
# 2. Create two audio files in the same directory as this script:
#    - `male_voice.wav`
#    - `female_voice.wav`
#    These should be clear, high-quality recordings of speech, at least
#    6-10 seconds long.
#
# ==============================================================================

import os
from TTS.api import TTS

def main():
    """
    Main function to demonstrate TTS and voice cloning capabilities.
    """
    print("Initializing the TTS model. This may take a moment...")
    print("On first run, it will download the model files (approx. 2-3 GB).")

    # --- 1. Initialize the TTS model ---
    # Model name for the multilingual, voice-cloning capable XTTS v2 model.
    model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    
    # Check if a CUDA-enabled GPU is available and use it.
    try:
        # Set gpu=True to use GPU, False to use CPU
        tts = TTS(model_name, gpu=True)
        print("TTS model loaded successfully on GPU.")
    except Exception as e:
        print(f"GPU not available or error occurred: {e}")
        print("Falling back to CPU. This will be slower.")
        tts = TTS(model_name, gpu=False)
        print("TTS model loaded successfully on CPU.")


    # --- 2. Define Speaker Audio Files and Output Directory ---
    output_dir = "tts_audio_outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Audio files will be saved in the '{output_dir}' directory.")

    # IMPORTANT: Define paths to your local speaker files.
    # These files MUST exist in the same directory as the script.
    speaker_wav_male = "male_voice.wav"
    speaker_wav_female = "male_voice.wav"

    # Check if speaker files exist before proceeding
    if not os.path.exists(speaker_wav_male) or not os.path.exists(speaker_wav_female):
        print("\n" + "="*50)
        print("!!! IMPORTANT WARNING !!!")
        print(f"Speaker file '{speaker_wav_male}' or '{speaker_wav_female}' not found.")
        print("Please create these .wav files with your own voice recordings.")
        print("The script will not be able to perform voice cloning without them.")
        print("="*50 + "\n")
        # You might want to exit here if the files are essential
        # return 
        
    
    # --- 3. Voice Cloning with Local Speaker Files ---
    
    # --- Male Voice Examples ---
    if os.path.exists(speaker_wav_male):
        print(f"\n--- Generating Speech using Male Voice: '{speaker_wav_male}' ---")
        
        # English Example (Male)
        text_english = "Hello, this is a test of the text to speech model in English, using a male voice."
        output_path_english_male = os.path.join(output_dir, "output_english_male.wav")
        print(f"Generating English speech for: '{text_english}'")
        tts.tts_to_file(
            text=text_english, 
            speaker_wav=speaker_wav_male, 
            language="en", 
            file_path=output_path_english_male
        )
        print(f"Saved to: {output_path_english_male}")

        # Hindi Example (Male)
        text_hindi = "नमस्ते, यह हिंदी में टेक्स्ट टू स्पीच मॉडल का परीक्षण है, एक पुरुष की आवाज का उपयोग करते हुए।"
        output_path_hindi_male = os.path.join(output_dir, "output_hindi_male.wav")
        print(f"Generating Hindi speech for: '{text_hindi}'")
        tts.tts_to_file(
            text=text_hindi, 
            speaker_wav=speaker_wav_male, 
            language="hi", 
            file_path=output_path_hindi_male
        )
        print(f"Saved to: {output_path_hindi_male}")

    # --- Female Voice Examples ---
    if os.path.exists(speaker_wav_female):
        print(f"\n--- Generating Speech using Female Voice: '{speaker_wav_female}' ---")

        # English Example (Female)
        text_english_female = "A giant horse cock weighs over 11 pounds"
        output_path_english_female = os.path.join(output_dir, "output_english_female.wav")
        print(f"Generating English speech for: '{text_english_female}'")
        tts.tts_to_file(
            text=text_english_female, 
            speaker_wav=speaker_wav_female, 
            language="en", 
            file_path=output_path_english_female
        )
        print(f"Saved to: {output_path_english_female}")

        # Hinglish Example (Female)
        text_hinglish = "Hello friends, chai pee lo. This is a Hinglish test with a female voice."
        output_path_hinglish_female = os.path.join(output_dir, "output_hinglish_female.wav")
        print(f"Generating Hinglish speech for: '{text_hinglish}'")
        tts.tts_to_file(
            text=text_hinglish, 
            speaker_wav=speaker_wav_female, 
            language="en", 
            file_path=output_path_hinglish_female
        )
        print(f"Saved to: {output_path_hinglish_female}")


    print("\nScript finished. Check the 'tts_audio_outputs' folder for the .wav files.")


if __name__ == '__main__':
    main()
