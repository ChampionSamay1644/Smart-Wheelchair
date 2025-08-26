import os
import shutil
import numpy as np
import scipy.io.wavfile as wav
import sounddevice as sd
from pathlib import Path

try:
    import noisereduce as nr
except ImportError:
    print("NoiseReduce not found. For better results, install it using: pip install noisereduce")
    nr = None

try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    from resemblyzer.audio import sampling_rate
except ImportError:
    print("Resemblyzer not found. Please install it: pip install resemblyzer")
    exit(1)

try:
    from pydub import AudioSegment
except ImportError:
    print("Pydub not found. Please install it: pip install pydub")
    AudioSegment = None

# --- Configuration ---
# Directories for storing voice data
VOICE_DB_PROCESSED_DIR = Path("./voice_db_processed")
VOICE_DB_EMBEDDINGS_DIR = Path("./voice_db_embeddings")

# Audio recording settings
SAMPLE_RATE = sampling_rate  # Use Resemblyzer's default sample rate
DURATION = 5  # seconds for live recording

# --- Voice Database Management ---

def noise_reduce_audio(data, rate):
    """Applies noise reduction to audio data if the library is available."""
    if nr is not None:
        # The noisereduce library works on float audio data
        return nr.reduce_noise(y=data, sr=rate)
    else:
        print("Skipping noise reduction as 'noisereduce' is not installed.")
        return data

def convert_mp3_to_wav(mp3_path, wav_path):
    """Converts an MP3 file to WAV format with the same sample rate and mono channel."""
    if AudioSegment is None:
        print("Pydub not installed. Cannot convert MP3 to WAV.")
        return False
    audio = AudioSegment.from_mp3(mp3_path)
    audio = audio.set_frame_rate(SAMPLE_RATE).set_channels(1)
    audio.export(wav_path, format="wav")
    return True

def store_voice(encoder):
    """
    Asks for an audio file, processes it, and stores its embedding in the database.
    """
    # Ensure the necessary directories exist before processing.
    VOICE_DB_PROCESSED_DIR.mkdir(exist_ok=True)
    VOICE_DB_EMBEDDINGS_DIR.mkdir(exist_ok=True)

    print("\nChoose how to add to database:")
    print("1. Provide a WAV/MP3 file")
    print("2. Live record and store")
    choice = input("Enter 1 or 2: ").strip()
    if choice == "1":
        filepath_str = input("Enter the path to the audio file to store in the database: ").strip()
        filepath = Path(filepath_str)

        if not filepath.is_file():
            print(f"Error: File not found at '{filepath_str}'")
            return

        ext = filepath.suffix.lower()
        speaker_name = input(f"Enter a name for this speaker (e.g., 'john_smith'): ").strip().lower()
        if not speaker_name:
            print("Error: Speaker name cannot be empty.")
            return

        # Convert MP3 to WAV if needed
        if ext == ".mp3":
            wav_path = VOICE_DB_PROCESSED_DIR / f"{speaker_name}_converted.wav"
            print("Converting MP3 to WAV...")
            if not convert_mp3_to_wav(str(filepath), str(wav_path)):
                print("MP3 conversion failed.")
                return
            filepath = wav_path
            ext = ".wav"
        print(f"  - Processing {filepath.name} for speaker '{speaker_name}'...")
        try:
            # preprocess_wav handles loading different audio formats and resampling.
            original_wav_data = preprocess_wav(filepath)

            # Apply noise reduction to the loaded audio data.
            reduced_wav_data = noise_reduce_audio(original_wav_data, SAMPLE_RATE)
            
            # Save the noise-reduced version as a WAV file for inspection.
            processed_path = VOICE_DB_PROCESSED_DIR / f"{speaker_name}_processed.wav"
            wav.write(processed_path, SAMPLE_RATE, (reduced_wav_data * 32767).astype(np.int16))

            # Generate the embedding from the noise-reduced audio data.
            embedding = encoder.embed_utterance(reduced_wav_data)
            embedding_path = VOICE_DB_EMBEDDINGS_DIR / f"{speaker_name}.npy"
            np.save(embedding_path, embedding)
            print(f"    - Saved processed audio to {processed_path}")
            print(f"    - Saved embedding for '{speaker_name}' to {embedding_path}")
            print("Voice stored successfully.")

        except Exception as e:
            print(f"    - Could not process file {filepath.name}. Error: {e}")
    elif choice == "2":
        speaker_name = input(f"Enter a name for this speaker (e.g., 'john_smith'): ").strip().lower()
        if not speaker_name:
            print("Error: Speaker name cannot be empty.")
            return
        temp_wav_path = VOICE_DB_PROCESSED_DIR / f"{speaker_name}_live.wav"
        print(f"\nPress Enter to start live recording for {DURATION} seconds...")
        print("Please speak clearly and consistently.")
        input()
        print("Recording...")
        recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
        sd.wait()
        wav.write(temp_wav_path, SAMPLE_RATE, recording)
        print(f"Saved recording to {temp_wav_path}")
        try:
            original_wav_data = preprocess_wav(temp_wav_path)

            # Apply noise reduction to the live audio data
            reduced_live_data = noise_reduce_audio(original_wav_data, SAMPLE_RATE)

            processed_path = VOICE_DB_PROCESSED_DIR / f"{speaker_name}_processed.wav"
            wav.write(processed_path, SAMPLE_RATE, (reduced_live_data * 32767).astype(np.int16))

            # Generate the embedding from the noise-reduced audio data.
            embedding = encoder.embed_utterance(reduced_live_data)
            embedding_path = VOICE_DB_EMBEDDINGS_DIR / f"{speaker_name}.npy"
            np.save(embedding_path, embedding)
            print(f"    - Saved processed audio to {processed_path}")
            print(f"    - Saved embedding for '{speaker_name}' to {embedding_path}")
            print("Voice stored successfully.")
            if temp_wav_path.exists():
                temp_wav_path.unlink()
        except Exception as e:
            print(f"    - Could not process live recording. Error: {e}")
    else:
        print("Invalid choice.")

# --- Live Voice Testing ---

def record_live_audio(filename):
    """Records live audio from the microphone and saves it to a file."""
    print(f"\nPress Enter to start recording for {DURATION} seconds...")
    print("Please speak clearly and consistently.")
    input()
    print("Recording...")
    recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()
    wav.write(filename, SAMPLE_RATE, recording)
    print(f"Saved recording to {filename}")

def get_all_speaker_scores(test_embed):
    """
    Compares a test embedding against all voice embeddings in the database.
    Returns a list of all speakers and their similarity scores, sorted by the best match.
    """
    all_matches = []
    try:
        # Compare against each embedding in the database
        for embed_path in VOICE_DB_EMBEDDINGS_DIR.glob("*.npy"):
            db_embed = np.load(embed_path)
            # Calculate cosine similarity (dot product of normalized vectors)
            similarity = np.dot(test_embed, db_embed)
            speaker_name = embed_path.stem
            all_matches.append((speaker_name, similarity))

    except Exception as e:
        print(f"Could not perform speaker match. Error: {e}")
    
    # Sort matches by similarity score in descending order
    return sorted(all_matches, key=lambda x: x[1], reverse=True)

def test_voice(encoder):
    """
    Orchestrates the live voice test: records, processes, and matches.
    """
    if not VOICE_DB_EMBEDDINGS_DIR.exists() or not any(VOICE_DB_EMBEDDINGS_DIR.iterdir()):
        print("\nVoice database is empty. Please use the 'store' command to add a voice first.")
        return

    temp_wav_path = Path("test_voice_live.wav")
    record_live_audio(temp_wav_path)

    # Preprocess the live recording (loads, resamples to float32 array)
    live_wav_data = preprocess_wav(temp_wav_path)
    
    # Apply noise reduction to the live audio data
    reduced_live_data = noise_reduce_audio(live_wav_data, SAMPLE_RATE)

    print("Generating embedding for your voice...")
    test_embed = encoder.embed_utterance(reduced_live_data)

    print("Comparing your voice against the database...")
    all_scores = get_all_speaker_scores(test_embed)

    if not all_scores:
        print("\n--- Could not compare against the database. ---")
        return

    # For debugging, show the best match regardless of the threshold
    best_speaker, best_sim = all_scores[0]
    print(f"\nBest match: Speaker '{best_speaker}' with similarity: {best_sim:.2f}")

    # Define a more lenient threshold and check if the best score meets it
    threshold = 0.75
    if best_sim >= threshold:
        print("\n--- Match Found! ---")
        print(f"  - Speaker: {best_speaker} (Similarity: {best_sim:.2f})")
    else:
        print(f"\n--- No matching speaker found above the threshold of {threshold}. ---")


    # Clean up temporary file
    if os.path.exists(temp_wav_path):
        os.remove(temp_wav_path)

# --- Main Application Logic ---

def main():
    """Main function to run the speaker verification tool."""
    print("--- Speaker Verification Tool ---")
    print("Loading the voice encoder model... (This may take a moment)")
    # Use the default VoiceEncoder, which handles its own model downloading/caching.
    encoder = VoiceEncoder()
    print("Model loaded.")
            
    # Main loop for user actions
    while True:
        action = input("\nType 'store' to add a voice to the database, 'test' for a live test, or 'exit': ").strip().lower()
        if action == "store":
            store_voice(encoder)
        elif action == "test":
            test_voice(encoder)
        elif action == "exit":
            print("Exiting.")
            break
        else:
            print("Invalid input. Please try again.")

if __name__ == "__main__":
    main()



## My name is Ramendranath Pandey and I am 50 years old. I am in India and I love cricket.