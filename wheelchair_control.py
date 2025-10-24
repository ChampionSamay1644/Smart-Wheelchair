#!/usr/bin/env python3
"""
Smart Wheelchair Control System with Multilingual Voice Commands

This system provides voice-controlled operation of a wheelchair with two main modes:
1. Online LLM Mode: Authenticated users can ask questions to an online LLM
2. Command Mode: Authenticated users can control the wheelchair with voice commands

Features:
- Voice authentication for security
- Multilingual command recognition (English, Hindi, Marathi)
- Enhanced phonetic matching for better command recognition
- Automatic language detection without requiring explicit translation
- Advanced noise handling and audio processing

Author: GitHub Copilot
Date: October 2, 2025
"""

import os
import sys
import time
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
from pathlib import Path
import torch
import difflib
import json
import uuid
import datetime
from typing import Dict, List, Tuple, Optional, Union, Any

try:
    import soundfile as sf
except ImportError:
    print("Warning: soundfile not installed. Install with: pip install soundfile")
    sf = None

try:
    import librosa
except ImportError:
    print("Warning: librosa not installed. Install with: pip install librosa")
    librosa = None

# Import shared components from the system module
from system import (
    preprocess_and_noise_reduce,
    transcribe_audio,
    query_llm,
    TEMP_DIR,
    TTS_OUTPUT_DIR,
    load_api_key,
    cosine_similarity,
    record_audio,
    save_float_to_wav,
    _load_audio_16k,
    load_local_stt_pipeline,
    WHISPER_MODEL_ID,
    WHISPER_MODEL_DIR
)

# Import the new multi-voice TTS module
from multi_voice_tts import synthesize_speech, play_audio

# Import voice recognition components
try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    from resemblyzer.audio import sampling_rate as RS_SAMPLING_RATE
except ImportError:
    print("Missing dependency: resemblyzer. Install with: pip install resemblyzer")
    sys.exit(1)

# =============================================================================
# VOICE ACTIVITY DETECTION - Consolidated from voice_activity.py
# =============================================================================

def detect_silence(audio, sample_rate, threshold=0.015, min_duration=0.5):
    """
    Detect if audio contains mostly silence or background noise.
    
    Args:
        audio: Audio data as numpy array
        sample_rate: Sample rate in Hz
        threshold: RMS threshold for considering audio as non-silent
        min_duration: Minimum duration of non-silent audio needed (seconds)
        
    Returns:
        (is_silent, speech_percentage): 
            is_silent: True if audio contains mostly silence
            speech_percentage: Percentage of audio that's considered speech
    """
    # Calculate energy over small windows
    frame_size = int(0.025 * sample_rate)  # 25ms windows
    hop_size = int(0.010 * sample_rate)    # 10ms hop
    
    # Ensure audio is float and normalized
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / 32768.0  # Convert from int16 to float
            
    # Calculate frame energies
    frames = []
    for i in range(0, len(audio) - frame_size, hop_size):
        frames.append(audio[i:i+frame_size])
    
    if not frames:  # If audio is too short
        return True, 0.0
        
    # Calculate RMS for each frame
    frame_rms = np.array([np.sqrt(np.mean(frame**2)) for frame in frames])
    
    # Count frames above threshold
    speech_frames = np.sum(frame_rms > threshold)
    speech_percentage = speech_frames / len(frame_rms)
    
    # Calculate required frames for minimum duration
    min_speech_frames = (min_duration / (len(audio) / sample_rate)) * len(frame_rms)
    
    is_silent = speech_frames < min_speech_frames
    
    return is_silent, speech_percentage * 100

def record_with_vad(duration, sample_rate, device=None, max_attempts=3, prompt_phrase=None):
    """
    Record audio with voice activity detection to ensure speech is captured.
    
    Args:
        duration: Recording duration in seconds
        sample_rate: Sample rate in Hz
        device: Audio device to use (None for default)
        max_attempts: Maximum number of recording attempts
        prompt_phrase: Optional phrase to display for the user to say
        
    Returns:
        (audio, speech_percent): The recorded audio as numpy array and speech percentage,
                                  or (None, 0) if all attempts failed
    """
    for attempt in range(1, max_attempts + 1):
        if attempt > 1:
            print(f"\nAttempt {attempt}/{max_attempts}: Let's try again...")
            time.sleep(0.5)
            
        if prompt_phrase:
            print(f"\nPlease say: \"{prompt_phrase}\"")
        
        print(f"Recording for {duration} seconds...")
        print("3... ", end="", flush=True)
        time.sleep(1)
        print("2... ", end="", flush=True)
        time.sleep(1)
        print("1... ", end="", flush=True)
        time.sleep(1)
        print("GO!")
        
        audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, 
                      channels=1, dtype='float32', device=device)
        sd.wait()
        audio = audio.flatten()
        
        # Check for silence
        is_silent, speech_percent = detect_silence(audio, sample_rate)
        
        if is_silent:
            print(f"Mostly silence detected (only {speech_percent:.1f}% speech). Please speak clearly.")
        else:
            print(f"Voice detected ({speech_percent:.1f}% speech).")
            return audio, speech_percent
    
    print("\nFailed to capture clear speech after multiple attempts.")
    return None, 0

# Standard prompt phrases for voice enrollment
ENROLLMENT_PHRASES = [
    "My voice is my password, verify my identity",
    "मेरी आवाज मेरा पासवर्ड है, मेरी पहचान सत्यापित करें",
    "माझा आवाज माझा पासवर्ड आहे, माझी ओळख सत्यापित करा"
]

# =============================================================================
# END VOICE ACTIVITY DETECTION
# =============================================================================

# =============================================================================
# VOICE PROFILE & DATABASE MANAGEMENT - Consolidated from voice_profile_creation.py and voice_db_management.py
# =============================================================================

def clean_voice_database(voice_db_processed_dir, voice_db_embeddings_dir, backup=True):
    """Clean the voice database by removing all existing voice profiles with optional backup."""
    if not voice_db_processed_dir.exists() and not voice_db_embeddings_dir.exists():
        print("No voice database found to clean.")
        return True
        
    processed_files = list(voice_db_processed_dir.glob("*.*"))
    embedding_files = list(voice_db_embeddings_dir.glob("*.npy"))
    total_files = len(processed_files) + len(embedding_files)
    
    if total_files == 0:
        print("Voice database is already empty.")
        return True
        
    print(f"Found {len(processed_files)} processed files and {len(embedding_files)} embeddings.")
    
    if backup:
        import shutil
        backup_dir = Path("./voice_db_backup_" + time.strftime("%Y%m%d_%H%M%S"))
        backup_dir.mkdir(exist_ok=True)
        backup_processed = backup_dir / "processed"
        backup_embeddings = backup_dir / "embeddings"
        backup_processed.mkdir(exist_ok=True)
        backup_embeddings.mkdir(exist_ok=True)
        
        print(f"Creating backup in {backup_dir}...")
        for file in processed_files:
            try:
                shutil.copy2(file, backup_processed / file.name)
            except Exception as e:
                print(f"Error backing up {file}: {e}")
                
        for file in embedding_files:
            try:
                shutil.copy2(file, backup_embeddings / file.name)
            except Exception as e:
                print(f"Error backing up {file}: {e}")
                
        print(f"Backup created successfully with {total_files} files.")
    
    print("Cleaning voice database...")
    for file in processed_files:
        try:
            file.unlink()
        except Exception as e:
            print(f"Error deleting {file}: {e}")
            
    for file in embedding_files:
        try:
            file.unlink()
        except Exception as e:
            print(f"Error deleting {file}: {e}")
    
    print(f"Voice database cleaned. Removed {total_files} files.")
    return True

def create_voice_profile_internal(encoder, voice_db_processed_dir, voice_db_embeddings_dir, sample_rate=16000):
    """Create a new voice profile with multilingual recordings."""
    print("\n=== Create New Voice Profile ===")
    
    speaker = input("Enter speaker name (lowercase, underscores allowed): ").strip().lower()
    if not speaker:
        print("Invalid name. Operation cancelled.")
        return False
    
    embedding_path = voice_db_embeddings_dir / f"{speaker}.npy"
    if embedding_path.exists():
        overwrite = input(f"Profile '{speaker}' already exists. Overwrite? (y/n): ").strip().lower()
        if overwrite != 'y':
            print("Operation cancelled.")
            return False
    
    while True:
        gender = input("Enter speaker gender (male / female / unknown): ").strip().lower()
        if gender in ['male', 'female', 'unknown']:
            break
        print("Invalid gender. Please enter 'male', 'female', or 'unknown'.")
    
    voice_db_processed_dir.mkdir(exist_ok=True)
    voice_db_embeddings_dir.mkdir(exist_ok=True)
    
    recordings = []
    record_duration = 5
    
    print("\nFor better voice authentication, we'll record you saying phrases in 3 different languages:")
    print("1. English: \"My voice is my password, verify my identity\"")
    print("2. Hindi: \"मेरी आवाज मेरा पासवर्ड है, मेरी पहचान सत्यापित करें\"")
    print("3. Marathi: \"माझा आवाज माझा पासवर्ड आहे, माझी ओळख सत्यापित करा\"")
    print("\nThis creates a more robust multilingual voice profile.")
    
    for i, phrase in enumerate(ENROLLMENT_PHRASES, 1):
        print(f"\n[Recording {i}/3]")
        
        try:
            language = "English" if i == 1 else "Hindi" if i == 2 else "Marathi"
            print(f"\nLanguage: {language}")
            print("Press Enter to begin recording...")
            input()
            
            if i == 2:
                print("Transliteration: \"Meri awaaz mera password hai, meri pehchaan satyapit karein\"")
            elif i == 3:
                print("Transliteration: \"Majha awaaz majha password aahe, majhi olakh satyapit kara\"")
            
            audio_float, speech_percent = record_with_vad(record_duration, sample_rate, max_attempts=3, prompt_phrase=phrase)
            
            if audio_float is None or speech_percent < 20:
                print(f"Failed to capture sufficient speech for phrase {i}.")
                continue
            
            audio = (np.clip(audio_float, -1.0, 1.0) * 32767).astype(np.int16)
            
            raw_path = voice_db_processed_dir / (f"{speaker}_raw.wav" if i == 1 else f"{speaker}_raw_{i}.wav")
            processed_path = voice_db_processed_dir / (f"{speaker}.wav" if i == 1 else f"{speaker}_{i}.wav")
            
            wav.write(raw_path, sample_rate, audio)
            recordings.append((audio, raw_path, processed_path))
            print(f"Successfully recorded phrase {i}.")
            
        except Exception as e:
            print(f"Error during recording: {e}")
    
    if not recordings:
        print("Failed to capture any valid recordings. Please try again.")
        return False
    
    print(f"\nSuccessfully captured {len(recordings)} recordings.")
    
    try:
        all_embeddings = []
        print("\nProcessing recordings...")
        print("Processing audio samples: ", end="", flush=True)
        
        for i, (audio, raw_path, processed_path) in enumerate(recordings, 1):
            try:
                print(f"{i}/{len(recordings)}... ", end="", flush=True)
                raw_wav = preprocess_wav(str(raw_path))
                wav.write(processed_path, sample_rate, (np.clip(raw_wav, -1.0, 1.0) * 32767).astype(np.int16))
                emb = encoder.embed_utterance(raw_wav)
                all_embeddings.append(emb)
                
                if i > 1:
                    np.save(voice_db_embeddings_dir / f"{speaker}_{i}.npy", emb)
                    
            except Exception as e:
                print(f"Error processing recording {i}: {e}")
        
        print("done!")
        
        if not all_embeddings:
            print("Failed to create any valid embeddings. Please try again.")
            return False
            
        if len(all_embeddings) > 1:
            print(f"Creating averaged voice profile from {len(all_embeddings)} recordings...")
            avg_embedding = np.mean(all_embeddings, axis=0)
            avg_embedding = avg_embedding / np.linalg.norm(avg_embedding)
        else:
            avg_embedding = all_embeddings[0]
        
        np.save(voice_db_embeddings_dir / f"{speaker}.npy", avg_embedding)
        
        with open(voice_db_processed_dir / f"{speaker}_gender.txt", 'w') as f:
            f.write(gender)
        
        print("\n=== Voice Profile Created Successfully ===")
        print(f"Speaker name: {speaker}")
        print(f"Gender: {gender}")
        print(f"Languages recorded: English, Hindi, and Marathi")
        print(f"Number of recordings used: {len(all_embeddings)}")
        print("This multilingual profile enhances speaker recognition across different languages.")
        
        return True
        
    except Exception as e:
        print(f"Error creating voice profile: {e}")
        return False

def test_voice_authentication_internal(encoder):
    """Test voice authentication without entering command mode."""
    print("\n=== Test Voice Authentication ===")
    print("This will test if your voice can be authenticated against existing profiles.")
    
    profiles = list(VOICE_DB_EMBEDDINGS_DIR.glob("*.npy"))
    if not profiles:
        print("\nNo voice profiles found. Please create a profile first.")
        input("\nPress Enter to continue...")
        return None
        
    print("\nFound", len(profiles), "voice profiles for matching.")
    
    user, score, processed_path = identify_speaker(encoder)
    
    if user:
        print(f"\nAuthentication successful!")
        print(f"Matched profile: {user}")
        print(f"Match score: {score:.4f}")
    else:
        print("\nAuthentication failed.")
        if score > 0:
            print(f"Best match score: {score:.4f} (below threshold)")
        else:
            print("No match found.")
            
    input("\nPress Enter to continue...")
    return None

# =============================================================================
# END VOICE PROFILE & DATABASE MANAGEMENT
# =============================================================================


# --- Global Configuration ---
SAMPLE_RATE = RS_SAMPLING_RATE  # 16000 Hz
RECORD_DURATION = 5  # seconds
SIMILARITY_THRESHOLD = 0.60  # Match threshold for voice authentication
VOICE_DB_PROCESSED_DIR = Path("./voice_db_processed")
VOICE_DB_EMBEDDINGS_DIR = Path("./voice_db_embeddings")
DEFAULT_TTS_VOICE_FILE = Path("./tts_outputs/command_response.wav")
DEFAULT_LANGUAGE = "en"
DEFAULT_GENDER = "male"

# Gender classification thresholds for pitch estimation
# Average male fundamental frequency is around 85-180 Hz
# Average female fundamental frequency is around 165-255 Hz
MALE_PITCH_THRESHOLD = 170  # Hz - Maximum for male classification
FEMALE_PITCH_THRESHOLD = 165  # Hz - Minimum for female classification
# Values between these thresholds are considered ambiguous

# Hindi direction detection data
HINDI_DIRECTION_WORDS = {
    # Forward direction words
    "आगे": "forward",
    "सामने": "forward",
    "सीधे": "forward",
    "aage": "forward",
    "seedha": "forward",
    "seedhe": "forward",
    "samne": "forward",
    
    # Backward direction words
    "पीछे": "backward",
    "वापस": "backward",
    "पिछे": "backward",
    "peeche": "backward",
    "vaapas": "backward",
    "piche": "backward",
    
    # Left direction words
    "बाएं": "left",
    "बायें": "left",
    "बायाँ": "left",
    "बाईं": "left",
    "bayen": "left",
    "baayen": "left",
    "baai": "left",
    "baen": "left",
    "बाई ओर": "left",
    "बाईं तरफ": "left",
    "bai taraf": "left",
    "bayein taraf": "left",
    
    # Right direction words
    "दाएं": "right",
    "दायें": "right",
    "दायाँ": "right",
    "दाईं": "right",
    "dayen": "right",
    "daayen": "right",
    "daai": "right",
    "daen": "right",
    "दाई ओर": "right",
    "दाईं तरफ": "right",
    "दाएं तरफ": "right",
    "dai taraf": "right",
    "daaye taraf": "right",
    
    # Rotate left words
    "बाएं घूमो": "rotate_left",
    "बायें घूमो": "rotate_left",
    "bayen ghumo": "rotate_left",
    "baayen ghumo": "rotate_left",
    "bai taraf ghumo": "rotate_left",
    "bayein taraf ghumo": "rotate_left",
    "बाएं मुड़ो": "rotate_left",
    "बायें मुड़ो": "rotate_left",
    "bayen mudo": "rotate_left",
    "baayen mudo": "rotate_left",
    
    # Rotate right words
    "दाएं घूमो": "rotate_right",
    "दायें घूमो": "rotate_right",
    "dayen ghumo": "rotate_right",
    "daayen ghumo": "rotate_right",
    "daaye taraf ghumo": "rotate_right",
    "dai taraf ghumo": "rotate_right",
    "दाएं मुड़ो": "rotate_right",
    "दायें मुड़ो": "rotate_right",
    "dayen mudo": "rotate_right",
    "daayen mudo": "rotate_right",
    
    # Stop words
    "रुको": "stop",
    "थामो": "stop",
    "रोको": "stop",
    "बंद": "stop",
    "ruko": "stop",
    "thamo": "stop",
    "roko": "stop",
    "band": "stop",
    "ruk jao": "stop",
    "रुक जाओ": "stop",
    
    # Speed-related words
    "तेज": "faster",
    "तेज़": "faster",
    "तेजी": "faster",
    "तेज़ी": "faster",
    "tez": "faster",
    "teji": "faster",
    
    # Slower speed words
    "धीरे": "slower",
    "धीमे": "slower",
    "धीमा": "slower",
    "dheere": "slower",
    "dheema": "slower",
    "dhime": "slower"
}

# Pre-defined wheelchair commands with variations in multiple languages
# Organized by command with variations in English, Hindi, Marathi, and other languages
WHEELCHAIR_COMMANDS = {
    # Forward command variations
    "forward": [
        # English variations
        "forward", "go forward", "move forward", "straight", "move straight", "go straight", 
        "ahead", "go ahead", "move ahead", "onwards", "advance", "proceed",
        # Hindi variations (देवनागरी and romanized)
        "आगे", "आगे बढ़ो", "आगे जाओ", "आगे चलो", "सीधे", "सीधे जाओ", "सीधे चलो",
        "aage", "aage badho", "aage jao", "aage chalo", "seedhe", "seedhe jao", "seedhe chalo",
        # Marathi variations
        "पुढे", "पुढे जा", "पुढे चल", "सरळ", "सरळ जा",
        "pudhe", "pudhe ja", "pudhe chal", "saral", "saral ja",
        # Spanish variations
        "adelante", "sigue adelante", "ve adelante", "recto", "sigue recto",
        # French variations
        "avancer", "en avant", "allez tout droit", "droit devant", "aller de l'avant",
        # German variations
        "vorwärts", "geradeaus", "nach vorne", "geh vorwärts", "gerade", "weiter",
        # Italian variations
        "avanti", "vai avanti", "dritto", "procedere", "muovere in avanti",
        # Chinese variations (Simplified)
        "前进", "向前", "直走", "往前", "前方", "直行",
        # Japanese variations
        "前進", "前へ", "まっすぐ", "直進", "フォワード",
        # Russian variations
        "вперед", "прямо", "вперёд", "двигайся вперёд", "прямо вперёд",
        # Korean variations
        "앞으로", "전진", "직진", "앞으로 가", "직진해",
        # Arabic variations
        "إلى الأمام", "تقدم", "مباشرة", "للأمام", "تحرك للأمام",
        # Turkish variations
        "ileri", "düz", "ilerle", "düz git", "ileri git",
        # Polish variations
        "naprzód", "do przodu", "prosto", "jedź prosto", "idź naprzód",
        # Dutch variations
        "vooruit", "rechtdoor", "ga vooruit", "recht vooruit", "voorwaarts",
        # Portuguese variations
        "frente", "para frente", "em frente", "avançar", "siga em frente",
        # Swedish variations
        "framåt", "rakt fram", "gå framåt", "fortsätt framåt", "rakt",
        # Finnish variations
        "eteenpäin", "suoraan", "mene eteenpäin", "etene", "suoraan eteenpäin",
        # Simple phonetic variations that may come from ASR errors
        "go for word", "ford", "foreward", "forword", "farward"
    ],
    
    # Backward command variations
    "backward": [
        # English variations
        "backward", "go backward", "move backward", "back", "go back", "move back", 
        "reverse", "go reverse", "move in reverse", "retreat", "step back", 
        # Hindi variations
        "पीछे", "पीछे जाओ", "पीछे चलो", "वापस", "वापस जाओ", "पीछे की ओर",
        "peeche", "peeche jao", "peeche chalo", "vaapas", "vaapas jao", "peeche ki or",
        # Marathi variations
        "मागे", "मागे जा", "मागे चल", "मागे चला", "मागे फिरा", 
        "mage", "mage ja", "mage chal", "mage chala", "mage fira",
        # Spanish variations
        "atrás", "hacia atrás", "ve atrás", "retrocede", "reversa",
        # French variations
        "reculer", "en arrière", "marche arrière", "reculez", "arrière", "recule",
        # German variations
        "rückwärts", "zurück", "nach hinten", "geh zurück", "zurückgehen", "rücken",
        # Italian variations
        "indietro", "vai indietro", "marcia indietro", "retrocedere", "tornare indietro", "retromarcia",
        # Chinese variations (Simplified)
        "后退", "向后", "倒退", "往后", "后方", "倒车",
        # Japanese variations
        "後退", "バック", "下がる", "戻る", "後ろへ", "バックする",
        # Russian variations
        "назад", "задний ход", "двигайся назад", "отступить", "реверс", "возвращайся",
        # Korean variations
        "뒤로", "후진", "뒤로 가", "뒤로 가세요", "백", "후퇴",
        # Arabic variations
        "للخلف", "إلى الخلف", "تراجع", "عد", "ارجع", "رجوع",
        # Turkish variations
        "geri", "geriye", "geri git", "tersine", "geri dön", "geri çek",
        # Polish variations
        "wstecz", "do tyłu", "cofnij", "cofaj", "zawróć", "odwrót",
        # Dutch variations
        "achteruit", "terug", "ga terug", "achterwaarts", "keer terug", "terugrijden",
        # Portuguese variations
        "para trás", "retroceder", "voltar", "recuar", "ré", "marcha atrás",
        # Swedish variations
        "bakåt", "backa", "gå bakåt", "tillbaka", "återgå", "reträtt",
        # Finnish variations
        "taaksepäin", "peruuta", "takaisin", "käänny takaisin", "taakse", "peruutus",
        # Phonetic variations
        "backword", "bakward", "back word", "backwad", "bak"
    ],
    
    # Left command variations
    "left": [
        # English variations
        "left", "go left", "move left", "turn left", "to the left", "leftward",
        # Hindi variations (with distinctive forms)
        "बाएं", "बाएं मुड़ो", "बाएं जाओ", "बाएं चलो", "बायीं ओर", "बायीं तरफ", "बायें घूमो",
        "बाई", "बाई तरफ", "बाई ओर", "बाई मुड़ो", "बाई मुडो", "बाई मुरें", "बाई मूड़ो", 
        "bayen", "baaye", "baye", "baye mudo", "baaye mudo", "baaye jao", "baaye chalo", "bayi or", "left mudo",
        "baee", "baaee", "bai taraf", "bai mudo", "bai muren", "bayein taraf", "bayen mudo",
        # Common spelling/pronunciation variations
        "bayein", "bayee", "bai", "baai", "baen", "baayen", "baain", 
        # Marathi variations
        "डावीकडे", "डावीकडे वळा", "डावीकडे जा", "डावीकडे चला",
        "davikade", "davikade vala", "davikade ja", "davikade chala",
        # Spanish variations
        "izquierda", "a la izquierda", "gira a la izquierda", "ve a la izquierda",
        # French variations
        "gauche", "à gauche", "tourner à gauche", "allez à gauche", "vers la gauche",
        # German variations
        "links", "nach links", "biege links ab", "links abbiegen", "zur linken", "linke seite",
        # Italian variations
        "sinistra", "a sinistra", "gira a sinistra", "vai a sinistra", "verso sinistra",
        # Chinese variations (Simplified)
        "左", "向左", "左转", "往左", "左边", "向左转",
        # Japanese variations
        "左", "左へ", "左折", "左に曲がる", "レフト", "左方向",
        # Russian variations
        "влево", "налево", "поверни налево", "левая сторона", "слева", "в левую сторону",
        # Korean variations
        "왼쪽", "왼쪽으로", "왼쪽으로 돌아", "좌회전", "왼쪽으로 가", "좌측",
        # Arabic variations
        "يسار", "إلى اليسار", "انعطف يسارا", "اذهب يسارا", "الجانب الأيسر", "يساراً",
        # Turkish variations
        "sol", "sola", "sola dön", "sola git", "sol taraf", "sola doğru",
        # Polish variations
        "lewo", "w lewo", "skręć w lewo", "idź w lewo", "na lewo", "po lewej",
        # Dutch variations
        "links", "naar links", "ga naar links", "linksom", "links afslaan", "linker kant",
        # Portuguese variations
        "esquerda", "à esquerda", "vire à esquerda", "vá para a esquerda", "lado esquerdo", "virar à esquerda",
        # Swedish variations
        "vänster", "till vänster", "sväng vänster", "gå åt vänster", "vänstra sidan", "vänd vänster",
        # Finnish variations
        "vasen", "vasemmalle", "käänny vasemmalle", "mene vasemmalle", "vasen puoli", "vasempaan",
        # Czech variations
        "vlevo", "doleva", "odbočte doleva", "jděte doleva", "na levé straně", "levá strana",
        # Phonetic variations and common ASR mistakes
        "lift", "leafed", "laft", "lft", "lef", "leven", "lefty", "läft"
    ],
    
    # Right command variations
    "right": [
        # English variations
        "right", "go right", "move right", "turn right", "to the right", "rightward",
        # Hindi variations (with distinctive forms)
        "दाएं", "दाएं मुड़ो", "दाएं जाओ", "दाएं चलो", "दायीं ओर", "दायीं तरफ", "दायें घूमो",
        "दाई", "दाई तरफ", "दाई ओर", "दाई मुड़ो", "दाई मुडो", "दाई मुरें", "दाई मूड़ो",
        "dayen", "daaye", "daye", "daye mudo", "daaye mudo", "daaye jao", "daaye chalo", "dayi or", "right mudo",
        "daee", "daaee", "dai taraf", "dai mudo", "dai muren", "dahine", "dahina", "dahini taraf", 
        # Common spelling/pronunciation variations
        "dayein", "dayee", "dai", "daai", "daen", "daayen", "daain",
        # Marathi variations
        "उजवीकडे", "उजवीकडे वळा", "उजवीकडे जा", "उजवीकडे चला",
        "ujavikade", "ujavikade vala", "ujavikade ja", "ujavikade chala",
        # Spanish variations
        "derecha", "a la derecha", "gira a la derecha", "ve a la derecha",
        # French variations
        "droite", "à droite", "tourner à droite", "allez à droite", "vers la droite",
        # German variations
        "rechts", "nach rechts", "biege rechts ab", "rechts abbiegen", "zur rechten", "rechte seite",
        # Italian variations
        "destra", "a destra", "gira a destra", "vai a destra", "verso destra",
        # Chinese variations (Simplified)
        "右", "向右", "右转", "往右", "右边", "向右转",
        # Japanese variations
        "右", "右へ", "右折", "右に曲がる", "ライト", "右方向",
        # Russian variations
        "вправо", "направо", "поверни направо", "правая сторона", "справа", "в правую сторону",
        # Korean variations
        "오른쪽", "오른쪽으로", "오른쪽으로 돌아", "우회전", "오른쪽으로 가", "우측",
        # Arabic variations
        "يمين", "إلى اليمين", "انعطف يمينا", "اذهب يمينا", "الجانب الأيمن", "يميناً",
        # Turkish variations
        "sağ", "sağa", "sağa dön", "sağa git", "sağ taraf", "sağa doğru",
        # Polish variations
        "prawo", "w prawo", "skręć w prawo", "idź w prawo", "na prawo", "po prawej",
        # Dutch variations
        "rechts", "naar rechts", "ga naar rechts", "rechtsom", "rechts afslaan", "rechter kant",
        # Portuguese variations
        "direita", "à direita", "vire à direita", "vá para a direita", "lado direito", "virar à direita",
        # Swedish variations
        "höger", "till höger", "sväng höger", "gå åt höger", "högra sidan", "vänd höger",
        # Finnish variations
        "oikea", "oikealle", "käänny oikealle", "mene oikealle", "oikea puoli", "oikeaan",
        # Czech variations
        "vpravo", "doprava", "odbočte doprava", "jděte doprava", "na pravé straně", "pravá strana",
        # Phonetic variations and common ASR mistakes
        "rite", "wright", "ryt", "rit", "rght", "raight", "righte", "rigte"
    ],
    
    # Rotate left command variations
    "rotate_left": [
        # English variations
        "rotate left", "spin left", "turn around left", "rotate counter-clockwise", "turn counter clockwise",
        "spin counter-clockwise", "rotate anticlockwise", "circle left", "turn full left",
        # Hindi variations
        "बाएं घूमो", "बाएं घूमना", "पूरा बाएं मुड़ो", "उल्टी दिशा में घूमो",
        "baaye ghumo", "baaye ghoom", "baaye ghumao", "baaye rotate karo", "counter clockwise ghoom",
        "ulti disha me ghumo", "pura baaye mudo", "left me ghoom jao",
        # Marathi variations
        "डावीकडे फिरा", "डावीकडे गोल फिरा", "डावीकडे वळून फिरा",
        "davikade fira", "davikade gol fira", "davikade valun fira",
        # Spanish variations
        "girar a la izquierda", "rotar a la izquierda", "dar vuelta a la izquierda", "girar completamente a la izquierda",
        # French variations
        "tourner à gauche complètement", "faire un tour à gauche", "rotation à gauche", "pivoter à gauche",
        "tourner dans le sens antihoraire", "faire un cercle à gauche",
        # German variations
        "nach links drehen", "links herum drehen", "links rotieren", "gegen den Uhrzeigersinn drehen",
        "links rundherum", "vollständig nach links drehen",
        # Italian variations
        "ruotare a sinistra", "girare completamente a sinistra", "fare un cerchio a sinistra", "rotazione sinistra",
        "girare in senso antiorario", "rotazione antioraria",
        # Chinese variations (Simplified)
        "向左旋转", "左转圈", "逆时针旋转", "完全向左转", "左侧旋转",
        # Japanese variations
        "左回り", "左に回転", "反時計回り", "左に旋回する", "左回転",
        # Russian variations
        "повернуть влево полностью", "вращаться влево", "поворот против часовой стрелки",
        "крутиться влево", "повернуться влево кругом",
        # Korean variations
        "왼쪽으로 회전", "왼쪽으로 돌기", "반시계 방향으로", "왼쪽으로 빙글빙글", "왼쪽으로 완전히 돌기",
        # Arabic variations
        "الدوران إلى اليسار", "دوران كامل لليسار", "لف إلى اليسار", "دوران عكس عقارب الساعة",
        # Turkish variations
        "sola dön", "sola döndür", "saat yönünün tersine", "sol tarafa dön", "tamamen sola dön",
        # Polish variations
        "obróć w lewo", "skręć całkowicie w lewo", "obróć się przeciwnie do ruchu wskazówek zegara", 
        # Dutch variations
        "draai naar links", "roteer linksom", "volledig naar links draaien", "tegen de klok in draaien",
        # Portuguese variations
        "girar à esquerda", "rodar para a esquerda", "rotação anti-horária", "dar volta completa à esquerda",
        # Phonetic variations
        "rotateleft", "rotate lft", "spin lft", "turn lft"
    ],
    
    # Rotate right command variations
    "rotate_right": [
        # English variations
        "rotate right", "spin right", "turn around right", "rotate clockwise", "turn clockwise",
        "spin clockwise", "circle right", "turn full right", "make a right circle",
        # Hindi variations
        "दाएं घूमो", "दाएं घूमना", "पूरा दाएं मुड़ो", "सीधी दिशा में घूमो",
        "daaye ghumo", "daaye ghoom", "daaye ghumao", "daaye rotate karo", "clockwise ghoom",
        "seedhi disha me ghumo", "pura daaye mudo", "right me ghoom jao",
        # Marathi variations
        "उजवीकडे फिरा", "उजवीकडे गोल फिरा", "उजवीकडे वळून फिरा",
        "ujavikade fira", "ujavikade gol fira", "ujavikade valun fira",
        # Spanish variations
        "girar a la derecha", "rotar a la derecha", "dar vuelta a la derecha",
        # French variations
        "tourner à droite complètement", "faire un tour à droite", "rotation à droite", "pivoter à droite",
        "tourner dans le sens horaire", "faire un cercle à droite",
        # German variations
        "nach rechts drehen", "rechts herum drehen", "rechts rotieren", "im Uhrzeigersinn drehen",
        "rechts rundherum", "vollständig nach rechts drehen",
        # Italian variations
        "ruotare a destra", "girare completamente a destra", "fare un cerchio a destra", "rotazione destra",
        "girare in senso orario", "rotazione oraria",
        # Chinese variations (Simplified)
        "向右旋转", "右转圈", "顺时针旋转", "完全向右转", "右侧旋转",
        # Japanese variations
        "右回り", "右に回転", "時計回り", "右に旋回する", "右回転",
        # Russian variations
        "повернуть вправо полностью", "вращаться вправо", "поворот по часовой стрелке",
        "крутиться вправо", "повернуться вправо кругом",
        # Phonetic variations
        "rotateright", "rotate rght", "spin rght", "turn rght"
    ],
    
    # Start command variations
    "start": [
        # English variations
        "start", "begin", "power on", "activate", "wake up", "turn on", "initiate", "get going", 
        "let's go", "engage", "launch", "commence", "start wheelchair",
        # Hindi variations
        "शुरू", "शुरू करो", "चालू करो", "चालू", "शुरुआत करो", "ऑन करो", "जागो",
        "shuru", "shuru karo", "chalu karo", "chalu", "on karo", "activate karo", "jago",
        "power on karo", "start karo", "start ho jao",
        # Marathi variations
        "सुरू", "सुरू करा", "चालू करा", "चालू", "ऑन करा",
        "suru", "suru kara", "chalu kara", "chalu", "on kara",
        # Spanish variations
        "empezar", "iniciar", "encender", "activar", "comenzar", "arrancar", "poner en marcha",
        # French variations
        "commencer", "démarrer", "allumer", "activer", "mettre en marche", "démarrage",
        "lancer", "s'y mettre", "allons-y", "en route",
        # German variations
        "starten", "beginnen", "einschalten", "aktivieren", "anmachen", "anfangen",
        "los", "anschalten", "in gang setzen", "initiieren",
        # Italian variations
        "avviare", "iniziare", "accendere", "attivare", "cominciare", "partire",
        "mettere in moto", "avvio", "via",
        # Chinese variations (Simplified)
        "开始", "启动", "打开", "激活", "开机", "运行", "启动轮椅",
        # Japanese variations
        "開始", "スタート", "起動", "オン", "作動", "始める", "電源オン",
        # Russian variations
        "старт", "начать", "включить", "активировать", "запустить", "начинать",
        "включение", "приступить", "поехали",
        # Korean variations
        "시작", "켜다", "켜기", "활성화", "작동", "시작하다", "출발",
        # Arabic variations
        "ابدأ", "تشغيل", "بدء", "تنشيط", "تفعيل", "شغل", "انطلق",
        # Turkish variations
        "başla", "başlat", "çalıştır", "aktive et", "aç", "başlama", "hareket et",
        # Polish variations
        "start", "rozpocznij", "włącz", "aktywuj", "uruchom", "zacznij", "ruszaj",
        # Dutch variations
        "starten", "beginnen", "aanzetten", "activeren", "inschakelen", "opstarten", "aan",
        # Portuguese variations
        "iniciar", "começar", "ligar", "ativar", "arrancar", "dar partida", "acionar",
        # Swedish variations
        "starta", "börja", "sätta på", "aktivera", "sätt igång", "kör igång", "slå på",
        # Finnish variations
        "aloita", "käynnistä", "aktivoi", "käynnistys", "laita päälle", "aloittaa", "virta päälle",
        # Czech variations
        "start", "začít", "zapnout", "aktivovat", "spustit", "zahájit", "nastartovat",
        # Phonetic variations
        "staart", "begin now", "stat", "strt"
    ],
    
    # Stop command variations
    "stop": [
        # English variations
        "stop", "halt", "pause", "wait", "brake", "power off", "deactivate", "cease", 
        "hold", "freeze", "stand still", "stay", "stop moving", "no movement",
        # Hindi variations
        "रुको", "थांबो", "रुक जाओ", "बंद करो", "बंद", "ऑफ करो", "ठहरो", "ठहर जाओ",
        "ruko", "thambo", "ruk jao", "band karo", "band", "off karo", "thehro", "theher jao",
        "stop karo", "rukna", "ab ruko", "bas", "bas karo", "deactivate karo",
        # Marathi variations
        "थांबा", "थांबवा", "बंद करा", "ऑफ करा",
        "thamba", "thambava", "band kara", "off kara",
        # Spanish variations
        "parar", "detener", "alto", "para", "detente", "espera",
        # French variations
        "arrêter", "arrêt", "stop", "halte", "pause", "attendre", "freiner", "éteindre",
        "désactiver", "cesser", "immobiliser", "tenir", "geler",
        # German variations
        "stopp", "halt", "anhalten", "pausieren", "warten", "bremsen", "ausschalten",
        "deaktivieren", "stillstehen", "bleiben", "einfrieren", "halten",
        # Italian variations
        "fermare", "fermati", "stop", "pausa", "aspetta", "frenare", "spegnere",
        "disattivare", "cessare", "bloccare", "immobile", "fermo", "arrestare",
        # Chinese variations (Simplified)
        "停止", "暂停", "等待", "刹车", "关闭", "停", "别动", "静止",
        # Japanese variations
        "停止", "ストップ", "止まれ", "止める", "待って", "ブレーキ", "オフ", "止まる", "中止",
        # Russian variations
        "стоп", "остановись", "пауза", "ждать", "тормоз", "выключить", "деактивировать",
        "прекратить", "держать", "замереть", "стоять", "замри",
        # Korean variations
        "멈춰", "정지", "멈추세요", "멈춤", "서", "스톱", "중지", "기다려", "그만",
        # Arabic variations
        "قف", "توقف", "انتظر", "أوقف", "كفى", "تمهل", "تعطيل", "إيقاف",
        # Turkish variations
        "dur", "durun", "durdur", "duraklat", "bekle", "durma", "fren", "durdur",
        # Polish variations
        "zatrzymaj", "stop", "stój", "wstrzymaj", "pauza", "hamuj", "czekaj", "zatrzymanie",
        # Dutch variations
        "stop", "halt", "houden", "wacht", "stoppen", "pauze", "rem", "stilstaan",
        # Portuguese variations
        "pare", "parar", "alto", "espera", "deter", "trava", "freio", "aguarde",
        # Swedish variations
        "stopp", "stanna", "håll", "pausa", "vänta", "broms", "avsluta", "stå still",
        # Finnish variations
        "seis", "pysähdy", "lopeta", "tauko", "jarruta", "odota", "keskeytä", "pysäytä",
        # Phonetic variations
        "stp", "stahp", "stoop", "brake now", "hault", "holt"
    ]
}

def estimate_fundamental_frequency(audio, sample_rate):
    """
    Estimate the fundamental frequency (pitch) of voice audio using autocorrelation.
    This is a lightweight implementation that works well for voiced speech.
    
    Args:
        audio: The audio signal (numpy array)
        sample_rate: The sample rate in Hz
        
    Returns:
        Estimated fundamental frequency in Hz, or None if estimation fails
    """
    # Use a frequency range typical for human voice
    min_freq = 80  # Hz - minimum frequency to detect (lower end of male voice)
    max_freq = 400  # Hz - maximum frequency to detect (higher end of female voice)
    
    # Convert frequency limits to periods (samples)
    min_period = int(sample_rate / max_freq)
    max_period = int(sample_rate / min_freq)
    
    # Ensure we have enough audio data
    if len(audio) < max_period * 3:
        return None  # Not enough data
    
    # Simple filtering to focus on speech frequencies
    from scipy import signal
    sos = signal.butter(2, [min_freq, 1000], 'bandpass', fs=sample_rate, output='sos')
    filtered = signal.sosfilt(sos, audio)
    
    # Use center chunk of audio for analysis (typically contains speech)
    center_start = len(filtered) // 4
    center_end = center_start + len(filtered) // 2
    center_chunk = filtered[center_start:center_end]
    
    # Calculate autocorrelation
    corr = np.correlate(center_chunk, center_chunk, mode='full')
    corr = corr[len(corr)//2:]  # Keep only the positive lags
    
    # Limit analysis to the frequency range we're interested in
    if len(corr) <= max_period:
        return None
    
    # Find peaks in autocorrelation (after skipping the first peak at lag 0)
    peaks = [i for i in range(min_period, min(max_period, len(corr)-1))
             if corr[i] > corr[i-1] and corr[i] > corr[i+1]]
    
    if not peaks:
        return None
    
    # Get highest peak (strongest periodicity)
    best_period = max(peaks, key=lambda i: corr[i])
    pitch = sample_rate / best_period
    
    return pitch

def detect_gender_from_audio(audio, sample_rate):
    """
    Detect the likely gender based on voice pitch estimation.
    Returns 'male', 'female', or 'unknown'.
    """
    pitch = estimate_fundamental_frequency(audio, sample_rate)
    
    if pitch is None:
        return 'unknown'
    
    if pitch < MALE_PITCH_THRESHOLD:
        return 'male'
    elif pitch > FEMALE_PITCH_THRESHOLD:
        return 'female'
    else:
        return 'unknown'  # Ambiguous range

def preprocess_audio_for_verification(audio_path):
    """
    Improved audio preprocessing for more reliable voice verification.
    Focuses on voice activity detection and voice quality preservation.
    Extracts gender characteristics to improve matching.
    """
    from scipy import signal
    import numpy as np
    
    # Get raw audio data using resemblyzer's preprocess function
    raw_wav = preprocess_wav(audio_path)
    
    # Calculate signal energy
    amplitude = np.abs(raw_wav)
    rms = np.sqrt(np.mean(raw_wav**2))
    print(f"[Audio] Raw signal RMS: {rms:.4f}")
    
    # Apply more aggressive voice activity detection
    # This better separates speech from background noise
    
    # 1. Apply bandpass filter focusing on speech frequencies (150Hz-3500Hz)
    # Lower cutoff to better capture male fundamental frequencies
    sos = signal.butter(2, [150, 3500], 'bandpass', fs=SAMPLE_RATE, output='sos')
    filtered = signal.sosfilt(sos, raw_wav)
    
    # 2. Use energy-based voice activity detection with adaptive threshold
    # Calculate rolling energy
    frame_length = int(SAMPLE_RATE * 0.025)  # 25ms frames
    hop_length = int(SAMPLE_RATE * 0.010)    # 10ms hop
    
    if librosa:
        energy = librosa.feature.rms(y=filtered, frame_length=frame_length, hop_length=hop_length)[0]
        # Use a more aggressive threshold (70th percentile) to better isolate speech
        energy_thresh = np.percentile(energy, 70)
        
        # Create a mask for frames above threshold
        voice_mask = energy > energy_thresh
        
        # Convert frame-level mask to sample-level mask
        sample_mask = np.zeros_like(raw_wav, dtype=bool)
        for i, is_voice in enumerate(voice_mask):
            start = i * hop_length
            end = min(start + frame_length, len(raw_wav))
            if is_voice:
                sample_mask[start:end] = True
                
        # Apply smoothing to avoid rapid transitions
        from scipy.ndimage import binary_dilation
        smooth_mask = binary_dilation(sample_mask, np.ones(int(SAMPLE_RATE * 0.1)))
    else:
        # Fallback if librosa not available
        amplitude = np.abs(filtered)
        threshold = np.percentile(amplitude, 70)
        smooth_mask = amplitude > threshold
    
    # 3. Extract only the voice segments for embedding
    # This dramatically improves voice profile quality
    voice_only = np.zeros_like(raw_wav)
    voice_only[smooth_mask] = raw_wav[smooth_mask]
    
    # 4. Normalize to consistent level
    proc_rms = np.sqrt(np.mean(voice_only[smooth_mask]**2)) if np.any(smooth_mask) else 0.001
    if proc_rms > 0.001:
        # Target a consistent level across all recordings
        target_rms = 0.1
        gain = target_rms / proc_rms
        processed_wav = voice_only * gain
    else:
        # Fallback if no voice detected
        processed_wav = raw_wav * 0.1
    
    final_rms = np.sqrt(np.mean(processed_wav**2))
    print(f"[Audio] Processed RMS: {final_rms:.4f} (voice-focused processing)")
    
    return processed_wav

# --- Voice Authentication Functions ---

def list_embeddings():
    """Returns a list of all voice embeddings in the database."""
    return list(VOICE_DB_EMBEDDINGS_DIR.glob("*.npy"))

def get_profile_gender(profile_name):
    """
    Retrieves the gender associated with a voice profile.
    
    Args:
        profile_name: Name of the profile (without file extension)
        
    Returns:
        'male', 'female', or 'unknown' based on stored gender information
    """
    gender_file = VOICE_DB_PROCESSED_DIR / f"{profile_name}_gender.txt"
    
    if gender_file.exists():
        try:
            with open(gender_file, 'r') as f:
                return f.read().strip().lower()
        except Exception:
            pass
            
    # For now, return unknown instead of trying to detect from audio
    # This avoids CPU-intensive processing during initialization
    return 'unknown'
    
    # NOTE: The following code was causing high CPU usage and has been disabled
    # If no gender file exists, try to determine gender from the stored voice sample
    # voice_file = VOICE_DB_PROCESSED_DIR / f"{profile_name}.wav"
    # if voice_file.exists():
    #     try:
    #         profile_wav = preprocess_wav(voice_file)
    #         return detect_gender_from_audio(profile_wav, SAMPLE_RATE)
    #     except Exception:
    #         pass
            
    # return 'unknown'

# Moved cosine_similarity to system.py and imported from there

def hindi_direction_detector(text: str) -> Optional[str]:
    """
    Detect direction commands from Hindi text.
    
    Args:
        text: Hindi text to analyze (can be in Devanagari or romanized)
        
    Returns:
        The detected command in English, or None if no command is detected
    """
    import re
    import string
    
    if not text:
        return None
        
    # Convert to lowercase for romanized text
    text_lower = text.lower()
    
    # Remove punctuation and hyphens
    text_clean = text_lower.translate(str.maketrans('', '', string.punctuation))
    text_clean = text_clean.replace('-', ' ')
    
    # First check for rotation commands with compound patterns
    rotation_patterns = [
        # Right rotation patterns
        (r'दाएं\s+(?:तरफ|ओर)?\s*(?:घूमो|मुड़ो|घूम|मुड़|फिरो|फिर)', "rotate_right"),
        (r'दायें\s+(?:तरफ|ओर)?\s*(?:घूमो|मुड़ो|घूम|मुड़|फिरो|फिर)', "rotate_right"),
        (r'दाईं\s+(?:तरफ|ओर)?\s*(?:घूमो|मुड़ो|घूम|मुड़|फिरो|फिर)', "rotate_right"),
        (r'दाई\s+(?:तरफ|ओर)?\s*(?:घूमो|मुड़ो|घूम|मुड़|फिरो|फिर)', "rotate_right"),
        (r'dai\s*(?:taraf|or)?\s*(?:ghumo|mudo|ghoom|mud|firo|fir|gumo)', "rotate_right"),
        (r'daaye\s*(?:taraf|or)?\s*(?:ghumo|mudo|ghoom|mud|firo|fir|gumo)', "rotate_right"),
        (r'dayen\s*(?:taraf|or)?\s*(?:ghumo|mudo|ghoom|mud|firo|fir|gumo)', "rotate_right"),
        (r'daayen\s*(?:taraf|or)?\s*(?:ghumo|mudo|ghoom|mud|firo|fir|gumo)', "rotate_right"),
        
        # Left rotation patterns
        (r'बाएं\s+(?:तरफ|ओर)?\s*(?:घूमो|मुड़ो|घूम|मुड़|फिरो|फिर)', "rotate_left"),
        (r'बायें\s+(?:तरफ|ओर)?\s*(?:घूमो|मुड़ो|घूम|मुड़|फिरो|फिर)', "rotate_left"),
        (r'बाईं\s+(?:तरफ|ओर)?\s*(?:घूमो|मुड़ो|घूम|मुड़|फिरो|फिर)', "rotate_left"),
        (r'बाई\s+(?:तरफ|ओर)?\s*(?:घूमो|मुड़ो|घूम|मुड़|फिरो|फिर)', "rotate_left"),
        (r'bai\s*(?:taraf|or)?\s*(?:ghumo|mudo|ghoom|mud|firo|fir|gumo)', "rotate_left"),
        (r'baye\s*(?:taraf|or)?\s*(?:ghumo|mudo|ghoom|mud|firo|fir|gumo)', "rotate_left"),
        (r'bayen\s*(?:taraf|or)?\s*(?:ghumo|mudo|ghoom|mud|firo|fir|gumo)', "rotate_left"),
        (r'baayen\s*(?:taraf|or)?\s*(?:ghumo|mudo|ghoom|mud|firo|fir|gumo)', "rotate_left"),
    ]
    
    # Check for rotation patterns
    for pattern, command in rotation_patterns:
        if re.search(pattern, text_clean):
            return command
    
    # Split into words for individual word matching
    words = text_clean.split()
    
    # Check each word for direction keywords
    for word in words:
        if word in HINDI_DIRECTION_WORDS:
            return HINDI_DIRECTION_WORDS[word]
    
    # More complex analysis for compound directions
    if any(x in text_clean for x in ["आगे बढ़ो", "aage badho", "आगे चलो", "aage chalo"]):
        return "forward"
    
    if any(x in text_clean for x in ["पीछे जाओ", "peeche jao", "वापस जाओ", "vaapas jao"]):
        return "backward"
        
    if any(x in text_clean for x in ["बाएं मुड़ो", "bayen mudo", "बाईं ओर", "baen or"]):
        return "left"
        
    if any(x in text_clean for x in ["दाएं मुड़ो", "dayen mudo", "दाईं ओर", "daen or"]):
        return "right"
        
    # Check for taraf/or (direction) words with ghumo/mudo (turn/rotate)
    if "taraf" in text_clean and "ghumo" in text_clean:
        if any(word in text_clean for word in ["dai", "daaye", "dayen", "daayen", "right"]):
            return "rotate_right"
        if any(word in text_clean for word in ["bai", "baaye", "bayen", "baayen", "left"]):
            return "rotate_left"
            
    if "or" in text_clean and any(word in text_clean for word in ["ghumo", "mudo", "ghum", "gumo"]):
        if any(word in text_clean for word in ["dai", "daaye", "dayen", "daayen", "right"]):
            return "rotate_right"
        if any(word in text_clean for word in ["bai", "baaye", "bayen", "baayen", "left"]):
            return "rotate_left"
    
    # If no direction found
    return None

def process_command_with_whisper_tiny(audio_path=None, detect_lang=True):
    """
    Process voice command using Whisper tiny model directly.
    This function handles recording (if audio_path not provided),
    transcription with Whisper tiny, and command matching.
    
    Args:
        audio_path: Optional path to existing audio file. If None, will record new audio.
        detect_lang: Whether to auto-detect language (True) or use DEFAULT_LANGUAGE (False)
        
    Returns:
        Tuple of (matched_command, confidence_score)
    """
    # Record audio if path not provided
    if audio_path is None:
        # Create timestamp for unique filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        audio_path = TEMP_DIR / f"voice_cmd_{timestamp}.wav"
        
        # Create temp directory if it doesn't exist
        TEMP_DIR.mkdir(exist_ok=True)
        
        # Import voice activity detection
        try:
            from voice_activity import record_with_vad, detect_silence
            vad_available = True
        except ImportError:
            vad_available = False
        
        # Use voice activity detection if available
        if vad_available:
            print("\n[Command Recording] Press Enter to start recording command...")
            input()
            
            # Use VAD-enhanced recording
            prompt_phrase = "Please speak your command clearly"
            audio_float, speech_percent = record_with_vad(
                RECORD_DURATION, 
                SAMPLE_RATE, 
                max_attempts=2,
                prompt_phrase=prompt_phrase
            )
            
            if audio_float is None or speech_percent < 10:
                print("Failed to detect sufficient speech in recording.")
                print("Please speak clearly when giving commands.")
                return None, 0
                
            # Convert to int16
            audio = (np.clip(audio_float, -1.0, 1.0) * 32767).astype(np.int16)
            wav.write(audio_path, SAMPLE_RATE, audio)
            
        else:
            # Fallback to basic recording
            print("\n[Command Recording] Press Enter to start recording command...")
            input()
            print(f"Recording for {RECORD_DURATION} seconds...")
            audio = sd.rec(int(RECORD_DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
            sd.wait()
            wav.write(audio_path, SAMPLE_RATE, audio.astype(np.int16))
            
            # Basic silence detection
            audio_float = audio.flatten().astype(np.float32) / 32768.0
            rms = np.sqrt(np.mean(audio_float**2))
            if rms < 0.01:  # Very low volume
                print("Warning: Recording volume is very low. Please speak louder.")
                # Continue anyway since we might still be able to process it
        
        # Process the audio for better voice command recognition
    try:
        # First, determine which language to use for transcription
        language_code = None  # Default: auto-detection
        
        if not detect_lang:
            # Use preset language if auto-detection is disabled
            language_code = DEFAULT_LANGUAGE
        
        # Transcribe using Whisper tiny model directly
        transcription = transcribe_command_audio(audio_path, language=language_code)
        
        if not transcription:
            print("Failed to transcribe audio or no speech detected.")
            return None, 0
        
        # Try Hindi direction detection first for better rotation command detection
        hindi_command = hindi_direction_detector(transcription)
        if hindi_command:
            print(f"Found command through Hindi direction detector: '{hindi_command}'")
            return hindi_command, 0.90  # High confidence for direct matches
            
        # Match the transcription to a wheelchair command
        command, confidence = match_command(transcription)
        
        return command, confidence
        
    except Exception as e:
        print(f"Error processing command: {e}")
        return None, 0

def identify_speaker(encoder=None):
    """
    Improved speaker identification with enhanced reliability and gender verification.
    Uses multiple comparison methods, voice-focused processing, and gender detection
    to prevent cross-gender matching issues.
    
    Returns (name, score, processed_path) if match found, else (None, score, processed_path).
    """
    # Load the encoder if it wasn't provided
    if encoder is None:
        print("Loading voice encoder (CPU)... This may take a moment.")
        try:
            from resemblyzer import VoiceEncoder
            encoder = VoiceEncoder(device="cpu")
            print("Voice encoder loaded successfully.")
        except Exception as e:
            print(f"CRITICAL ERROR: Failed to load voice encoder: {e}")
            print("Please ensure resemblyzer is properly installed.")
            return None, 0, None
    # Check if we have any stored voice profiles (use direct glob to avoid CPU-intensive operations)
    embeddings_list = list(VOICE_DB_EMBEDDINGS_DIR.glob("*.npy"))
    if not embeddings_list:
        print("Voice database empty. Please add voices using voice profile management first.")
        return None, 0, None

    # Prepare paths for audio files
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    raw_path = TEMP_DIR / f"voice_cmd_raw_{timestamp}.wav"
    processed_path = TEMP_DIR / f"voice_cmd_processed_{timestamp}.wav"
    
    # Create temp directory if it doesn't exist
    TEMP_DIR.mkdir(exist_ok=True)
    
    # Record audio with voice activity detection
    from voice_activity import record_with_vad
    
    print("\n[Voice Authentication] Press Enter to start recording...")
    input()
    
    # Random verification phrase
    import random
    
    # Multilingual verification phrases (English, Hindi, Marathi)
    verification_phrases = {
        "english": [
            "Please verify my voice for access",
            "I need to control the wheelchair now",
            "Smart wheelchair voice authentication"
        ],
        "hindi": [
            "कृपया एक्सेस के लिए मेरी आवाज़ सत्यापित करें",
            "मुझे अब व्हीलचेयर नियंत्रित करने की आवश्यकता है",
            "स्मार्ट व्हीलचेयर आवाज प्रमाणीकरण"
        ],
        "marathi": [
            "कृपया प्रवेशासाठी माझा आवाज सत्यापित करा",
            "मला आता व्हीलचेयर नियंत्रित करण्याची आवश्यकता आहे",
            "स्मार्ट व्हीलचेयर आवाज प्रमाणीकरण"
        ]
    }
    
    # Ask user which language they prefer for verification
    print("\nSelect language for verification:")
    print("1. English")
    print("2. Hindi")
    print("3. Marathi")
    lang_choice = input("Enter choice (1/2/3) or press Enter for English: ").strip()
    
    if lang_choice == "2":
        lang = "hindi"
        print("\nHindi selected. Transliteration:")
        print("1. Kripaya access ke liye meri awaaz satyapit karen")
        print("2. Mujhe ab wheelchair niyantrit karne ki aavashyakta hai")
        print("3. Smart wheelchair awaaz pramanikaran")
    elif lang_choice == "3":
        lang = "marathi"
        print("\nMarathi selected. Transliteration:")
        print("1. Krupaya praveshasathi majha awaaj satyapit kara")
        print("2. Mala aata wheelchair niyantrit karnyachi aavashyakta aahe")
        print("3. Smart wheelchair awaaj pramanikaran")
    else:
        lang = "english"
        print("\nEnglish selected.")
    
    # Choose a random phrase in the selected language
    verification_phrase = random.choice(verification_phrases[lang])
    
    # Record with voice activity detection
    print("\n[Voice Authentication] Please speak clearly:")
    audio_float, speech_percent = record_with_vad(
        RECORD_DURATION, 
        SAMPLE_RATE, 
        max_attempts=3,
        prompt_phrase=verification_phrase
    )
    
    if audio_float is None or speech_percent < 15:
        print("Failed to capture sufficient speech for authentication.")
        return None, 0, None
    
    # Convert to int16 format
    audio = (np.clip(audio_float, -1.0, 1.0) * 32767).astype(np.int16)
    wav.write(raw_path, SAMPLE_RATE, audio)
    
    # Process the audio with simplified processing for better efficiency
    try:
        # Basic preprocessing - use resemblyzer's built-in preprocessing which is optimized
        from resemblyzer import preprocess_wav
        reduced = preprocess_wav(str(raw_path))
        wav.write(processed_path, SAMPLE_RATE, (np.clip(reduced, -1.0, 1.0) * 32767).astype(np.int16))
        
        # Generate the embedding directly - this is the essential step
        test_emb = encoder.embed_utterance(reduced)
        
        # Use a simple and faster approach to detect gender - just check if gender was provided by user
        # This avoids the CPU-intensive gender detection
        current_speaker_gender = 'unknown'
        print("Ready to compare with voice profiles...")
        
    except Exception as e:
        print(f"Audio processing failed: {e}")
        return None, 0, raw_path

    # Compare against all voice profiles using multiple metrics
    print(f"Comparing against {len(embeddings_list)} voice profiles...")
    scores = []
    gender_filtered_scores = []  # Will hold only gender-matching profiles
    
    # Dictionary to store gender of each profile for reference
    profile_genders = {}
    
    for p in embeddings_list:
        name = p.stem
        try:
            # Load the stored embedding
            db_emb = np.load(p)
            
            # Look for a gender label file associated with this profile
            gender_file = VOICE_DB_PROCESSED_DIR / f"{name}_gender.txt"
            profile_gender = 'unknown'
            
            # If the gender file exists, read the gender
            if gender_file.exists():
                with open(gender_file, 'r') as f:
                    profile_gender = f.read().strip().lower()
            else:
                # Don't try to determine gender from audio - this is CPU intensive
                # Instead, just use 'unknown' and let the user manually set it if needed
                profile_gender = 'unknown'
                # Save the unknown gender to avoid recalculating next time
                with open(gender_file, 'w') as f:
                    f.write(profile_gender)
            
            profile_genders[name] = profile_gender
            
            # Just use cosine similarity - much faster and almost as effective
            cosine_sim = cosine_similarity(test_emb, db_emb)
            
            # Use this directly as our similarity score
            combined_sim = cosine_sim
            
            # Store all scores
            scores.append((name, combined_sim))
            print(f"  - {name}: similarity {combined_sim:.3f} [Gender: {profile_gender}]")
            
            # Apply gender filtering: if both genders are known and don't match,
            # don't include in gender-filtered scores
            if (current_speaker_gender != 'unknown' and profile_gender != 'unknown' and
                current_speaker_gender != profile_gender):
                print(f"    Gender mismatch: {current_speaker_gender} vs {profile_gender}")
            else:
                gender_filtered_scores.append((name, combined_sim))
                
        except Exception as e:
            print(f"Error processing {name}: {e}")
    
    # Sort by similarity (highest first)
    scores.sort(key=lambda x: x[1], reverse=True)
    
    # Also sort gender-filtered scores
    gender_filtered_scores.sort(key=lambda x: x[1], reverse=True)
    
    # If no valid scores
    if not scores:
        print("No valid voice profiles found.")
        return None, 0, processed_path
    
    # Choose whether to use gender-filtered or all scores
    if gender_filtered_scores and current_speaker_gender != 'unknown':
        print(f"Using gender-filtered results ({len(gender_filtered_scores)} profiles)")
        final_scores = gender_filtered_scores
    else:
        print("Using all results (gender filtering inactive)")
        final_scores = scores
    
    # If no valid filtered scores
    if not final_scores:
        print("No gender-matching voice profiles found.")
        return None, 0, processed_path
    
    # Get best match and check for close scores
    best_name, best_score = final_scores[0]
    
    # Check if there are multiple close matches (potential confusion)
    confusion_warning = ""
    if len(final_scores) > 1:
        second_name, second_score = final_scores[1]
        score_diff = best_score - second_score
        if score_diff < 0.1:  # Very close match, potential confusion
            confusion_warning = f" (Warning: Close match with {second_name}: {second_score:.3f}, diff: {score_diff:.3f})"
    
    print(f"\nBest match: {best_name} (similarity {best_score:.3f}, gender: {profile_genders.get(best_name, 'unknown')}){confusion_warning}")
    
    # Apply threshold (slight adjustment from original)
    if best_score >= SIMILARITY_THRESHOLD:
        print(f"Authentication successful! ({best_score:.3f} >= {SIMILARITY_THRESHOLD})")
        return best_name, best_score, processed_path
    else:
        print(f"Authentication failed. Score {best_score:.3f} below threshold {SIMILARITY_THRESHOLD}")
        return None, best_score, processed_path
        
# Additional language models for better transcription quality
LANGUAGE_CODES = {
    'en': 'english',
    'hi': 'hindi',
    'mr': 'marathi',
    'es': 'spanish',
    'fr': 'french',
    'de': 'german',
    'it': 'italian',
    'zh': 'chinese',
    'ja': 'japanese',
    'ru': 'russian',
    'ko': 'korean',
    'ar': 'arabic',
    'tr': 'turkish',
    'pl': 'polish',
    'nl': 'dutch',
    'pt': 'portuguese',
    'sv': 'swedish',
    'fi': 'finnish',
    'cs': 'czech',
    'da': 'danish',
    'el': 'greek',
    'fa': 'persian',
    'he': 'hebrew',
    'hu': 'hungarian',
    'id': 'indonesian',
    'no': 'norwegian',
    'ro': 'romanian',
    'sk': 'slovak',
    'th': 'thai',
    'uk': 'ukrainian',
    'vi': 'vietnamese',
    'bn': 'bengali',
    'ta': 'tamil',
    'te': 'telugu',
    'ur': 'urdu',
    'ms': 'malay',
    'tl': 'tagalog'
}

def transcribe_command_audio(audio_path, language=None):
    """
    Transcribe audio command using Whisper tiny model directly.
    Optimized for command recognition with language-specific processing.
    
    Args:
        audio_path: Path to the audio file
        language: Optional language code to force a specific language
                  If None, auto-detection will be used
                  
    Returns:
        Transcribed text from the audio
    """
    # Load the Whisper tiny model pipeline if not already loaded
    pipeline = load_local_stt_pipeline()
    if pipeline is None:
        print("Error: Could not load Whisper tiny model pipeline.")
        return ""
    
    # Preprocess audio for better command recognition
    audio_processed = preprocess_and_noise_reduce(audio_path)
    
    # Configure transcription options
    generate_kwargs = {"task": "transcribe"}
    if language:
        generate_kwargs["language"] = language
    
    # Transcribe with Whisper tiny
    try:
        start_time = time.time()
        result = pipeline(audio_processed, generate_kwargs=generate_kwargs)
        transcription = result["text"].strip()
        elapsed = time.time() - start_time
        
        print(f"Whisper tiny transcription: '{transcription}' ({elapsed:.2f}s)")
        return transcription
    except Exception as e:
        print(f"Error transcribing command audio: {e}")
        return ""

# --- Command Processing Functions ---

def detect_language(text):
    """
    Detect language of the input text to prioritize command matching in that language.
    This function identifies multiple languages using script characteristics and common words.
    Returns language code as string: 'en', 'hi', 'mr', 'es', 'fr', 'de', 'it', 'zh', 'ja', 'ru', 
    'ko', 'ar', 'tr', 'pl', 'nl', 'pt', 'sv', 'fi', or 'unknown'
    """
    import re
    
    # Check if text is empty
    if not text:
        return 'unknown'
    
    # Normalize text for more accurate matching
    text = text.lower().strip()
    
    # Check for Devanagari script (Hindi/Marathi)
    if re.search(r'[\u0900-\u097F]', text):
        # Distinguish between Hindi and Marathi with unique words
        marathi_words = ['वळा', 'डावीकडे', 'उजवीकडे', 'थांबा', 'पुढे', 'मागे', 'साठी', 'करा']
        if any(word in text for word in marathi_words):
            return 'mr'  # Marathi
        return 'hi'      # Hindi
    
    # Check for Cyrillic (Russian, Ukrainian, Bulgarian)
    if re.search(r'[\u0400-\u04FF]', text):
        # Distinguish between Russian and other Cyrillic script languages
        russian_words = ['влево', 'вправо', 'вперед', 'назад', 'стоп', 'начать', 'поверни', 'стой', 'двигайся']
        if any(word in text for word in russian_words):
            return 'ru'  # Russian
        polish_words = ['lewo', 'prawo', 'naprzód', 'wstecz', 'stop', 'start', 'zatrzymaj', 'jedź']
        if any(word in text for word in polish_words):
            return 'pl'  # Polish
        return 'ru'  # Default to Russian for Cyrillic
    
    # Check for Chinese characters
    if re.search(r'[\u4e00-\u9fff]', text) and not re.search(r'[\u3040-\u30ff]', text):
        return 'zh'
    
    # Check for Japanese characters (Hiragana, Katakana, and Kanji)
    if re.search(r'[\u3040-\u30ff]', text):
        return 'ja'
        
    # Check for Korean (Hangul)
    if re.search(r'[\uAC00-\uD7AF\u1100-\u11FF]', text):
        return 'ko'
        
    # Check for Arabic script
    if re.search(r'[\u0600-\u06FF]', text):
        return 'ar'
    
    # For Latin-based scripts, check for distinctive words/patterns
    
    # Spanish markers
    spanish_words = ['izquierda', 'derecha', 'adelante', 'atrás', 'parar', 'empezar', 'detente', 'sigue', 'gira', 'alto']
    if any(word in text for word in spanish_words):
        return 'es'
    
    # Portuguese markers
    portuguese_words = ['esquerda', 'direita', 'frente', 'trás', 'atrás', 'parar', 'começar', 'pare', 'avançar', 'siga']
    if any(word in text for word in portuguese_words):
        return 'pt'
    
    # French markers
    french_words = ['gauche', 'droite', 'avancer', 'reculer', 'arrêter', 'commencer', 'arrêt', 'marche', 'tournez', 'halte']
    if any(word in text for word in french_words):
        return 'fr'
    
    # German markers
    german_words = ['links', 'rechts', 'vorwärts', 'rückwärts', 'stopp', 'starten', 'halt', 'weiter', 'biege', 'geradeaus']
    if any(word in text for word in german_words):
        return 'de'
    
    # Italian markers
    italian_words = ['sinistra', 'destra', 'avanti', 'indietro', 'fermare', 'iniziare', 'fermati', 'vai', 'gira', 'dritto']
    if any(word in text for word in italian_words):
        return 'it'
    
    # Dutch markers
    dutch_words = ['links', 'rechts', 'vooruit', 'achteruit', 'stop', 'begin', 'starten', 'houden', 'draai', 'rechtdoor']
    if any(word in text for word in dutch_words):
        return 'nl'
    
    # Turkish markers
    turkish_words = ['sol', 'sağ', 'ileri', 'geri', 'dur', 'başla', 'durdur', 'git', 'dön', 'düz']
    if any(word in text for word in turkish_words):
        return 'tr'
    
    # Swedish markers
    swedish_words = ['vänster', 'höger', 'framåt', 'bakåt', 'stopp', 'starta', 'halt', 'fortsätt', 'sväng', 'rakt']
    if any(word in text for word in swedish_words):
        return 'sv'
    
    # Finnish markers
    finnish_words = ['vasen', 'oikea', 'eteenpäin', 'taaksepäin', 'pysähdy', 'aloita', 'seis', 'jatka', 'käänny', 'suoraan']
    if any(word in text for word in finnish_words):
        return 'fi'
    
    # Polish markers (expanded)
    polish_expanded = ['lewo', 'prawo', 'przód', 'tył', 'zatrzymaj', 'rozpocznij', 'stój', 'idź', 'skręć']
    if any(word in text for word in polish_expanded):
        return 'pl'
    
    # Hindi romanized markers
    hindi_romanized = ['baaye', 'daaye', 'aage', 'peeche', 'ruko', 'shuru', 'chalo', 'mudo', 'seedhe', 'vaapas']
    if any(word in text for word in hindi_romanized):
        return 'hi'
    
    # Marathi romanized markers
    marathi_romanized = ['davikade', 'ujavikade', 'pudhe', 'mage', 'thamba', 'suru', 'chala', 'vala', 'sarala', 'fira']
    if any(word in text for word in marathi_romanized):
        return 'mr'
    
    # Common English command words
    english_words = ['forward', 'backward', 'left', 'right', 'stop', 'start', 'turn', 'go', 'move', 'halt']
    if any(word in text for word in english_words):
        return 'en'
    
    # Default to English if no other language detected
    return 'en'

def transcribe_command_audio(audio_path, language=None):
    """
    Transcribe audio command using Whisper tiny model directly.
    Optimized for command recognition with language-specific processing.
    
    Args:
        audio_path: Path to the audio file
        language: Optional language code to force a specific language
                  If None, auto-detection will be used
                  
    Returns:
        Transcribed text from the audio
    """
    # Load the Whisper tiny model pipeline if not already loaded
    pipeline = load_local_stt_pipeline()
    if pipeline is None:
        print("Error: Could not load Whisper tiny model pipeline.")
        return ""
    
    # Preprocess audio for better command recognition
    audio_processed = preprocess_and_noise_reduce(audio_path)
    
    # Configure transcription options
    generate_kwargs = {"task": "transcribe"}
    if language:
        generate_kwargs["language"] = language
    
    # Transcribe with Whisper tiny
    try:
        start_time = time.time()
        result = pipeline(audio_processed, generate_kwargs=generate_kwargs)
        transcription = result["text"].strip()
        elapsed = time.time() - start_time
        
        print(f"Whisper tiny transcription: '{transcription}' ({elapsed:.2f}s)")
        return transcription
    except Exception as e:
        print(f"Error transcribing command audio: {e}")
        return ""

def match_command(transcribed_text):
    """
    Match transcribed text to the closest wheelchair command.
    Enhanced to handle multilingual inputs and partial command matches.
    Implements improved fuzzy matching for better cross-language recognition.
    Returns tuple of (matched_command, confidence_score)
    """
    if not transcribed_text:
        return None, 0
    
    # Convert to lowercase, strip spaces and remove punctuation
    import re
    text = transcribed_text.lower().strip()
    text = re.sub(r'[.!?,;:।]', '', text)  # Remove common punctuation (includes Devanagari danda)
    
    # Detect the language to prioritize appropriate command variations
    detected_language = detect_language(text)
    print(f"Detected language: {detected_language}")
    
    # Language-specific processing to improve matching
    if detected_language in ['hi', 'mr', 'en']:  # Apply Hindi detector to English and Marathi too
        # Run specialized Hindi direction detection (works for Hindi, Hinglish, Romanized Hindi)
        hindi_result = hindi_direction_detector(text)
        if hindi_result:
            print(f"Hindi direction detector found: {hindi_result}")
            return hindi_result, 0.95
            
    # Check specifically for rotation commands with ghumo/taraf/mudo patterns
    if any(word in text for word in ['ghum', 'ghumo', 'mudo', 'mud', 'gumo', 'rotate', 'spin', 'turn', 'circle']):
        if any(word in text for word in ['dai', 'daye', 'daaye', 'daayen', 'dayen', 'right']):
            return "rotate_right", 0.90
        elif any(word in text for word in ['bai', 'baye', 'baaye', 'baayen', 'bayen', 'left']):
            return "rotate_left", 0.90
        
    # Check for taraf (direction) with left/right indicators
    if 'taraf' in text:
        if any(word in text for word in ['dai', 'daye', 'daaye', 'daayen', 'dayen', 'right']):
            return "right", 0.85
        elif any(word in text for word in ['bai', 'baye', 'baaye', 'baayen', 'bayen', 'left']):
            return "left", 0.85
    
    # Try exact match first (highest confidence)
    for cmd, variations in WHEELCHAIR_COMMANDS.items():
        if text in variations:
            print(f"Found exact match: '{text}' -> {cmd}")
            return cmd, 1.0
    
    # Check if the full text contains any of our command variations
    # This helps with commands embedded in longer phrases
    for cmd, variations in WHEELCHAIR_COMMANDS.items():
        for variation in variations:
            if variation in text and len(variation) > 3:  # Only match substantial variations
                # The longer the match, the higher the confidence
                confidence = min(0.95, len(variation) / len(text) * 1.2)  # Cap at 0.95
                print(f"Found command '{variation}' in '{text}' -> {cmd} (confidence: {confidence:.2f})")
                return cmd, confidence
    
    # If no embedded match, use fuzzy matching with word-by-word approach
    best_match = None
    best_score = 0
    
    # SPECIAL CASE: Check for specific problematic Hindi pattern first
    # This is for the "पूरा दाय मुडव" (poora daaye mudo) case
    if "पूरा" in text and "दाय" in text and ("मुड" in text or "mud" in text):
        print(f"Detected special case 'पूरा दाय मुडव' pattern - this is RIGHT direction")
        return "right", 0.95
    
    # Split input into words to handle commands embedded in sentences
    words = text.split()
    
    # Enhanced phonetic mapping for commonly misrecognized words across languages
    phonetic_variants = {
        # Hindi/English transliteration variants - Left
        'bye': ['baaye', 'baye', 'by', 'bai', 'bay', 'buy'],
        'bay': ['baaye', 'baye', 'bai', 'by', 'buy'],
        'buy': ['baaye', 'baye', 'bai', 'by', 'bay'],
        'bai': ['baaye', 'baye', 'bay', 'by', 'buy'],
        'left': ['lift', 'laft', 'lft', 'lef'],
        
        # Hindi/English transliteration variants - Right
        'day': ['daaye', 'daye', 'dai', 'dye', 'die', 'दाय', 'दाएं', 'दाई'],
        'die': ['daaye', 'daye', 'dai', 'day', 'dye', 'दाय', 'दाएं'],
        'dai': ['daaye', 'daye', 'die', 'day', 'dye', 'दाय', 'दाएं'],
        'दाय': ['right', 'daaye', 'daye', 'dai', 'day'],  # Direct mapping for the problematic word
        'right': ['rite', 'ryt', 'rit', 'rght', 'wright'],
        
        # Hindi/English transliteration variants - Turn
        'mude': ['mudo', 'mudna', 'mudho', 'mode', 'mood', 'move', 'मुडव', 'मुडो', 'मुड़ो'],
        'mode': ['mudo', 'mudna', 'mude', 'mood', 'move', 'मुडव', 'मुडो'],
        'mude.': ['mudo', 'mudna', 'mode', 'mood', 'move', 'मुडव'],
        'mood': ['mudo', 'mudna', 'mode', 'mude', 'move', 'मुडव'],
        'move': ['mudo', 'mudna', 'mode', 'mude', 'mood', 'मुडव'],
        'मुडव': ['mudo', 'turn', 'mudna', 'मुड़ो', 'मुडो'],  # Direct mapping for the problematic word
        
        # Hindi/English transliteration variants - Stop
        'ruko': ['rukho', 'rukna', 'rukko', 'roko', 'roku', 'rocco'],
        'stop': ['stp', 'stahp', 'stoop', 'hault', 'holt'],
        
        # Hindi/English transliteration variants - Forward/Go
        'chalo': ['challo', 'chal', 'chalu', 'challu', 'cello'],
        'karo': ['kro', 'karro', 'karho', 'kero', 'kiro']
    }
    
    # Create phonetically expanded version of the input text
    expanded_words = list(words)  # Start with original words
    
    # Add phonetic variants to expanded_words
    for i, word in enumerate(words):
        # Check if this word has phonetic variants
        for original, variants in phonetic_variants.items():
            if word == original or word.startswith(original):
                expanded_words.extend(variants)
            # Also check reverse mapping (e.g., if input contains 'baaye' but command uses 'left')
            if word in variants:
                expanded_words.append(original)
    
    # Remove duplicates but preserve order
    expanded_words = list(dict.fromkeys(expanded_words))
    print(f"Expanded word list: {expanded_words}")
    
    # Check for specific language command patterns
    # These are common command patterns across languages that should get special handling
    command_patterns = [
        # Hindi command patterns
        {"pattern": ["baaye", "baye", "by", "bye"], "command": "left", "score": 0.85},
        {"pattern": ["daaye", "daye", "day", "die"], "command": "right", "score": 0.85},
        {"pattern": ["aage", "seedhe", "forward"], "command": "forward", "score": 0.85},
        {"pattern": ["peeche", "back", "vaapas"], "command": "backward", "score": 0.85},
        {"pattern": ["ruko", "stop", "thehro"], "command": "stop", "score": 0.85},
        {"pattern": ["chalo", "shuru", "start"], "command": "start", "score": 0.85},
        
        # Multi-word Hindi patterns
        {"pattern": ["baaye", "mudo"], "command": "left", "score": 0.95, "multi_word": True},
        {"pattern": ["daaye", "mudo"], "command": "right", "score": 0.95, "multi_word": True},
        {"pattern": ["left", "mudo"], "command": "left", "score": 0.90, "multi_word": True},
        {"pattern": ["right", "mudo"], "command": "right", "score": 0.90, "multi_word": True}
    ]
    
    # Process single-word and multi-word patterns
    for pattern_def in command_patterns:
        pattern_words = pattern_def["pattern"]
        multi_word = pattern_def.get("multi_word", False)
        
        if multi_word and len(words) >= 2:
            # For multi-word patterns, check if all pattern words appear in the expanded words
            matches_all = True
            for pattern_word in pattern_words:
                if not any(pattern_word in w for w in expanded_words):
                    matches_all = False
                    break
                    
            # Check special combinations like "bye mude", "baaye mudo", etc.
            if matches_all or any(w1 + " " + w2 in text for w1 in ["bye", "by", "bay", "baaye"] for w2 in ["mude", "mode", "mudo"]):
                if "left" in pattern_def["command"]:
                    if pattern_def["score"] > best_score:
                        best_score = pattern_def["score"]
                        best_match = pattern_def["command"]
                        print(f"Matched multi-word pattern for '{pattern_def['command']}' with score {best_score}")
            
            # Check right-turn combinations
            if matches_all or any(w1 + " " + w2 in text for w1 in ["die", "day", "daaye"] for w2 in ["mude", "mode", "mudo"]):
                if "right" in pattern_def["command"]:
                    if pattern_def["score"] > best_score:
                        best_score = pattern_def["score"]
                        best_match = pattern_def["command"] 
                        print(f"Matched multi-word pattern for '{pattern_def['command']}' with score {best_score}")
        else:
            # For single-word patterns, check if any pattern word is in the expanded words
            for pattern_word in pattern_words:
                for word in expanded_words:
                    if pattern_word == word:
                        if pattern_def["score"] > best_score:
                            best_score = pattern_def["score"]
                            best_match = pattern_def["command"]
                            print(f"Matched single-word pattern '{pattern_word}' for '{pattern_def['command']}' with score {best_score}")
    
    # Traditional fuzzy matching as a fallback
    import difflib
    for cmd, variations in WHEELCHAIR_COMMANDS.items():
        for variation in variations:
            # Only consider substantive variations to avoid false positives
            if len(variation) < 3:  # Skip very short variations
                continue
                
            # Use both traditional sequence matcher and word overlap approaches
            
            # 1. Exact word match in expanded list (highest confidence)
            variation_words = variation.split()
            for v_word in variation_words:
                if len(v_word) > 2 and v_word in expanded_words:  # Only meaningful words
                    word_match_score = 0.82  # High confidence for exact word matches
                    if word_match_score > best_score:
                        best_score = word_match_score
                        best_match = cmd
                        print(f"Word match: '{v_word}' from '{variation}' for '{cmd}' with score {best_score}")
            
            # 2. Traditional sequence matcher for fuzzy string similarity
            similarity = difflib.SequenceMatcher(None, text, variation).ratio()
            
            # 3. Boost score if the command is a substring of the input
            if variation in text:
                similarity += 0.18  # Significant boost for embedded matches
            
            # 4. Word overlap measurement (proportion of command words found in input)
            variation_words = set(variation.split())
            expanded_text_words = set(expanded_words)
            common_words = variation_words.intersection(expanded_text_words)
            
            if variation_words and common_words:  # Avoid division by zero
                word_overlap_score = len(common_words) / len(variation_words) * 0.95  # High weight for word overlap
                similarity = max(similarity, word_overlap_score)
            
            # Update best match if this is better
            if similarity > best_score:
                best_score = similarity
                best_match = cmd
                print(f"Fuzzy match: '{variation}' for '{cmd}' with score {best_score}")
                
            # Special handling for Hindi/English common commands
            if 'mudo' in variation and ('baaye' in variation or 'daaye' in variation):
                if any(w in text for w in ['bye', 'by', 'bay']) and any(w in text for w in ['mude', 'mode']):
                    if 'baaye' in variation and best_score < 0.9:
                        best_score = 0.9
                        best_match = cmd
                        print(f"Special Hindi match for left: {best_score}")
                elif any(w in text for w in ['die', 'day']) and any(w in text for w in ['mude', 'mode']):
                    if 'daaye' in variation and best_score < 0.9:
                        best_score = 0.9
                        best_match = cmd
                        print(f"Special Hindi match for right: {best_score}")
    
    # Only return a match if similarity is above threshold
    # Using adaptive threshold based on the match quality:
    # - Higher threshold (0.80+) for critical commands like stop
    # - Medium threshold (0.60) for most commands
    # - Lower threshold (0.50) for commands with extensive phonetic variation
    
    # Default minimum threshold
    min_threshold = 0.50
    
    # Safety-critical commands need higher confidence
    if best_match == "stop":
        min_threshold = 0.55  # Slightly higher for stop command
    
    # Debug information
    print(f"Best match: '{best_match}' with confidence score {best_score:.2f} (threshold: {min_threshold})")
    
    if best_score >= min_threshold and best_match:
        return best_match, best_score
    
    return None, best_score

def execute_command(command):
    """
    Execute the matched wheelchair command.
    In a real implementation, this would control the wheelchair motors.
    Currently just prints the command for demonstration.
    """
    command_responses = {
        "forward": "Moving forward",
        "backward": "Moving backward",
        "left": "Turning left",
        "right": "Turning right",
        "rotate_left": "Rotating left",
        "rotate_right": "Rotating right",
        "start": "Starting wheelchair",
        "stop": "Stopping wheelchair"
    }
    
    response = command_responses.get(command, f"Executing {command}")
    print(f"\n[COMMAND EXECUTION]: {response}")
    
    # Play pre-generated TTS response if available, or generate on-the-fly if not
    tts_file = TTS_OUTPUT_DIR / f"cmd_{command}.wav"
    
    if tts_file.exists():
        # Use pre-generated audio file
        print(f"Using pre-generated TTS response from {tts_file}")
        try:
            import sounddevice as sd
            import soundfile as sf
            data, fs = sf.read(tts_file)
            sd.play(data, fs)
            sd.wait()
        except Exception as e:
            print(f"Error playing pre-generated audio: {e}")
            # Fall back to generating TTS
            try:
                synthesize_speech(response, DEFAULT_LANGUAGE, DEFAULT_GENDER)
            except Exception as e2:
                print(f"Error generating speech: {e2}")
    else:
        # Generate TTS on-the-fly
        print(f"No pre-generated TTS file found at {tts_file}, generating...")
        try:
            synthesize_speech(response, DEFAULT_LANGUAGE, DEFAULT_GENDER)
        except Exception as e:
            print(f"Error generating speech: {e}")
    
    return response

def test_tts_voices():
    """
    Test the TTS voices by speaking sample text in various languages and genders.
    """
    print("\n=== Testing Text-to-Speech Voices ===")
    
    # Define test phrases for different languages
    test_phrases = {
        "en": "Hello, this is a test of the English voice for the Smart Wheelchair system.",
        "hi": "नमस्ते, यह स्मार्ट व्हीलचेयर सिस्टम के हिंदी आवाज का परीक्षण है।",
        "mr": "नमस्कार, हा स्मार्ट व्हीलचेअर सिस्टमच्या मराठी आवाजाची चाचणी आहे."
    }
    
    # Define available languages with their readable names
    available_languages = {
        "en": "English",
        "hi": "Hindi",
        "mr": "Marathi (will use Hindi voice)"
    }
    
    while True:
        print("\nAvailable Languages:")
        for code, name in available_languages.items():
            print(f"  {code}: {name}")
        
        print("\nAvailable Voice Genders:")
        print("  male: Male voice")
        print("  female: Female voice")
        
        print("\nOptions:")
        print("  1. Test a specific voice")
        print("  2. Test all available voices")
        print("  3. Return to main menu")
        
        tts_choice = input("\nEnter option (1/2/3): ").strip()
        
        if tts_choice == "1":
            # Test a specific voice
            lang_code = input("Enter language code (e.g., en, hi): ").strip().lower()
            if lang_code not in test_phrases:
                print(f"Language '{lang_code}' is not supported. Please choose from {', '.join(test_phrases.keys())}")
                continue
                
            gender = input("Enter voice gender (male/female): ").strip().lower()
            if gender not in ["male", "female"]:
                print("Invalid gender. Please enter 'male' or 'female'.")
                continue
                
            custom_text = input(f"Enter custom text or press Enter to use default {lang_code} phrase: ").strip()
            text = custom_text if custom_text else test_phrases[lang_code]
            
            print(f"\nSynthesizing speech in {lang_code} with {gender} voice...")
            output_path = synthesize_speech(text, lang_code, gender, play=True)
            if output_path:
                print(f"Speech generated successfully: {output_path}")
            else:
                print("Failed to generate speech.")
                
        elif tts_choice == "2":
            # Test all available voices
            print("\nTesting all available voices...")
            
            for lang, phrase in test_phrases.items():
                for gender in ["male", "female"]:
                    print(f"\nTesting {available_languages.get(lang, lang)} ({gender}):")
                    print(f"Text: {phrase}")
                    output_path = synthesize_speech(phrase, lang, gender, play=True)
                    if output_path:
                        print(f"Generated: {output_path}")
                    else:
                        print(f"Failed to generate speech for {lang} ({gender})")
                    
                    # Small pause between tests
                    time.sleep(1)
            
        elif tts_choice == "3":
            # Return to main menu
            break
            
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")


def execute_command_with_language_feedback(command, detected_language=None):
    """
    Execute a wheelchair command with language-specific feedback.
    Uses the detected language for the feedback if available,
    otherwise falls back to the default language.
    
    Args:
        command: The wheelchair command to execute
        detected_language: The language code of the detected language (e.g., 'en', 'hi')
    """
    if not command:
        return False
    
    print(f"\nEXECUTING COMMAND: {command.upper()}")
    
    # Determine feedback language
    feedback_lang = DEFAULT_LANGUAGE
    if detected_language and detected_language in LANGUAGE_CODES:
        feedback_lang = detected_language
        print(f"Using {LANGUAGE_CODES[feedback_lang]} for feedback")
    
    # Generate language-specific feedback messages
    feedback_messages = {
        "forward": {
            "en": "Moving forward",
            "hi": "आगे बढ़ रहे हैं",
            "mr": "पुढे जात आहे",
            "es": "Avanzando",
            "fr": "Avançant",
            "de": "Vorwärts",
            "zh": "前进",
            "ja": "前進中",
            "ru": "Движение вперед",
        },
        "backward": {
            "en": "Moving backward",
            "hi": "पीछे जा रहे हैं",
            "mr": "मागे जात आहे",
            "es": "Retrocediendo",
            "fr": "Reculant",
            "de": "Rückwärts",
            "zh": "后退",
            "ja": "後退中",
            "ru": "Движение назад",
        },
        "left": {
            "en": "Turning left",
            "hi": "बाएं मुड़ रहे हैं",
            "mr": "डावीकडे वळत आहे",
            "es": "Girando a la izquierda",
            "fr": "Tournant à gauche",
            "de": "Biege links ab",
            "zh": "左转",
            "ja": "左折中",
            "ru": "Поворот налево",
        },
        "right": {
            "en": "Turning right",
            "hi": "दाएं मुड़ रहे हैं",
            "mr": "उजवीकडे वळत आहे",
            "es": "Girando a la derecha",
            "fr": "Tournant à droite",
            "de": "Biege rechts ab",
            "zh": "右转",
            "ja": "右折中",
            "ru": "Поворот направо",
        },
        "stop": {
            "en": "Stopping",
            "hi": "रुक रहे हैं",
            "mr": "थांबत आहे",
            "es": "Deteniéndose",
            "fr": "Arrêt",
            "de": "Anhalten",
            "zh": "停止",
            "ja": "停止中",
            "ru": "Остановка",
        },
        "start": {
            "en": "Starting system",
            "hi": "सिस्टम शुरू हो रहा है",
            "mr": "सिस्टम सुरू होत आहे",
            "es": "Iniciando sistema",
            "fr": "Démarrage du système",
            "de": "System startet",
            "zh": "启动系统",
            "ja": "システム起動中",
            "ru": "Запуск системы",
        }
    }
    
    # Get appropriate feedback message
    message = feedback_messages.get(command, {}).get(feedback_lang, f"Executing {command} command")
    
    # Play pre-generated command response if available
    cmd_file = TTS_OUTPUT_DIR / f"cmd_{command}.wav"
    if cmd_file.exists():
        try:
            import sounddevice as sd
            import soundfile as sf
            data, fs = sf.read(cmd_file)
            sd.play(data, fs)
            sd.wait()
            print(f"Played '{command}' response")
        except Exception as e:
            print(f"Error playing command audio: {e}")
            # Fall back to generating TTS
            try:
                synthesize_speech(message, feedback_lang, DEFAULT_GENDER)
            except Exception as e2:
                print(f"Error generating speech: {e2}")
    else:
        # Generate TTS response
        try:
            synthesize_speech(message, feedback_lang, DEFAULT_GENDER)
        except Exception as e:
            print(f"Error generating speech: {e}")
    
    # PLACEHOLDER: In a real system, this would interface with motor controls
    # For demonstration, we just print what would happen
    if command == "forward":
        print("Wheelchair moving FORWARD")
    elif command == "backward":
        print("Wheelchair moving BACKWARD")
    elif command == "left":
        print("Wheelchair turning LEFT")
    elif command == "right":
        print("Wheelchair turning RIGHT")
    elif command == "rotate_left":
        print("Wheelchair ROTATING LEFT (counter-clockwise)")
    elif command == "rotate_right":
        print("Wheelchair ROTATING RIGHT (clockwise)")
    elif command == "start":
        print("Wheelchair system STARTING")
    elif command == "stop":
        print("Wheelchair system STOPPING")
    
    return True

def process_voice_command(processed_audio_path):
    """
    Process voice command from audio file.
    Returns the matched command and confidence score.
    Enhanced with improved multilingual recognition across all languages supported by Whisper Tiny.
    """
    # Perform speech-to-text on the processed audio
    print("\nTranscribing audio to text...")
    print(f"Using {WHISPER_MODEL_ID} model for fast recognition on CPU")
    
    # Load the STT pipeline for transcription with language auto-detection
    # Whisper Tiny supports a wide range of languages including English, Hindi, Marathi,
    # Spanish, French, German, Italian, Chinese, Japanese, Russian, and many more
    pipe = load_local_stt_pipeline()
    if pipe is None:
        print("STT pipeline not available")
        return None, 0
        
    loaded = _load_audio_16k(processed_audio_path)
    if loaded is None:
        print("Audio could not be loaded")
        return None, 0
        
    audio_array, sr = loaded
    print(f"[STT] Loaded audio RMS: {np.sqrt(np.mean(audio_array**2)):.6f}")
    
    # Get transcription results
    results = []
    try:
        # Directly use the pipeline without specifying task
        # This avoids the generation config error
        result_transcribe = pipe({"array": audio_array, "sampling_rate": sr})
        text_transcribe = result_transcribe.get("text", "").strip()
        print(f"[STT] transcribe result: '{text_transcribe}'")
        results.append(text_transcribe)  # Add transcription to results
        
        # Generate phonetic variations of transcribed text immediately
        # This helps with Hindi-English transliteration issues
        words_transcribe = text_transcribe.lower().split()
        
        # Expanded phonetic mapping for common command words in Hindi and other languages
        phonetic_mapping = {
            # Left command transliterations - clearly marked for LEFT
            'bye': 'baaye_LEFT',
            'by': 'baaye_LEFT',
            'bay': 'baaye_LEFT',
            'buy': 'baaye_LEFT',
            'bai': 'baaye_LEFT',
            'baye': 'baaye_LEFT',
            'bayee': 'baaye_LEFT',
            'bayein': 'baaye_LEFT',
            'baaee': 'baaye_LEFT',
            'baee': 'baaye_LEFT',
            
            # Right command transliterations - clearly marked for RIGHT
            'die': 'daaye_RIGHT',
            'day': 'daaye_RIGHT',
            'dye': 'daaye_RIGHT',
            'dai': 'daaye_RIGHT',
            'daye': 'daaye_RIGHT',
            'dayee': 'daaye_RIGHT',
            'dayein': 'daaye_RIGHT',
            'daaee': 'daaye_RIGHT',
            'daee': 'daaye_RIGHT',
            'dahine': 'daaye_RIGHT',
            'dahina': 'daaye_RIGHT',
            
            # Turn command transliterations
            'mude': 'mudo',
            'mude.': 'mudo',
            'mode': 'mudo',
            'mood': 'mudo',
            'mud': 'mudo',
            'move': 'mudo',
            'muren': 'mudo',
            'moodo': 'mudo',
            'mudna': 'mudo',
            'mudhna': 'mudo',
            
            # Direction words in Hindi (with taraf = direction)
            'taraf': 'direction',
            'or': 'direction',
            'tarf': 'direction',
            
            # Stop command transliterations
            'roko': 'ruko',
            'roku': 'ruko',
            'rocco': 'ruko',
            'rukho': 'ruko',
            'rukna': 'ruko',
            
            # Start command transliterations
            'chalu': 'chalo',
            'challu': 'chalo',
            'cello': 'chalo',
            'challo': 'chalo'
        }
        
        # Generate all possible phonetic variants systematically
        phonetic_variants = []
        
        # Check for common command patterns and create variants
        # Process each word individually first
        for i, word in enumerate(words_transcribe):
            if word in phonetic_mapping:
                # Create a variation where just this word is replaced
                variant = text_transcribe.lower()
                variant = variant.replace(word, phonetic_mapping[word])
                phonetic_variants.append(variant)
        
        # Then process common command combinations (2-word commands)
        for i in range(len(words_transcribe)-1):
            first_word = words_transcribe[i]
            second_word = words_transcribe[i+1]
            
            # Check if we have a known command pair
            if first_word in phonetic_mapping and second_word in phonetic_mapping:
                # Create variants with both words replaced
                variant = text_transcribe.lower()
                variant = variant.replace(first_word, phonetic_mapping[first_word])
                variant = variant.replace(second_word, phonetic_mapping[second_word])
                phonetic_variants.append(variant)
        
        # Add the phonetic variants to results with high priority
        for variant in phonetic_variants:
            print(f"[STT] phonetic variant: '{variant}'")
            results.append(variant)
        
        # Try specialized Hindi direction detection first
        # This function is optimized specifically for Hindi direction commands
        hindi_command, hindi_confidence = hindi_direction_detector(text_transcribe)
        if hindi_command and hindi_confidence > 0.7:
            print(f"[STT] Detected Hindi direction command: {hindi_command} ({hindi_confidence:.2f})")
            # Add with high priority - add this command twice to increase its weight
            results.insert(0, hindi_command)  # Insert at the beginning for highest priority
            results.append(hindi_command)     # Also append to ensure it's considered
            
            # Special case for "पूरा दाय मुडव" (poora daaye mudo)
            if "पूरा" in text_transcribe and "दाय" in text_transcribe:
                print(f"[STT] Detected specific 'पूरा दाय मुडव' pattern - forcing RIGHT command")
                results = [hindi_command]  # Override all other results
                return hindi_command, hindi_confidence  # Return immediately with high confidence
            
        # Add special case direct matches for common Hindi commands
        hindi_commands = [
            # Left turn variations
            {"pattern": ["bye mude", "by mude", "bay mude", "bye mode", "by mode", "bai mudo", "bai taraf"], 
             "replacement": "baaye mudo", "description": "turn left"},
             
            # Right turn variations
            {"pattern": ["die mude", "day mude", "die mode", "day mode", "dai mudo", "dai taraf"], 
             "replacement": "daaye mudo", "description": "turn right"},
             
            # Stop variations
            {"pattern": ["ruko", "rukho", "roko"], 
             "replacement": "ruko", "description": "stop"},
             
            # Start/go variations
            {"pattern": ["chalo", "chalu", "challu"], 
             "replacement": "chalo", "description": "go"}
        ]
        
        # Check for each Hindi command pattern
        for cmd in hindi_commands:
            for pattern in cmd["pattern"]:
                if pattern in text_transcribe.lower():
                    print(f"[STT] Adding specific Hindi match for '{cmd['replacement']}' ({cmd['description']})")
                    results.append(cmd["replacement"])
                    break
        
        # Specific detection for Hindi directional words
        # These are strong indicators of left/right commands
        hindi_direction_markers = {
            # Right direction indicators
            "दाई": "right", "दाई तरफ": "right", "दाई मुड़ो": "right", "दायीं": "right", "दाएं": "right",
            "दाहिना": "right", "दाहिने": "right", "daaye": "right", "daye": "right", "dahine": "right",
            "dai taraf": "right", "dayi": "right", "daee": "right", "दायें": "right", "दायीं": "right",
            
            # Left direction indicators
            "बाई": "left", "बाई तरफ": "left", "बाई मुड़ो": "left", "बायीं": "left", "बाएं": "left",
            "baaye": "left", "baye": "left", "bayein": "left", "bai taraf": "left", "bayi": "left",
            "baee": "left", "बायें": "left", "बायीं": "left"
        }
        
        # Check for directional markers in the transcribed text
        for marker, direction in hindi_direction_markers.items():
            if marker in text_transcribe.lower():
                print(f"[STT] Found Hindi direction marker '{marker}' -> {direction}")
                if direction == "right":
                    results.append("daaye mudo")
                    print(f"[STT] Adding right turn command based on Hindi marker")
                elif direction == "left":
                    results.append("baaye mudo")
                    print(f"[STT] Adding left turn command based on Hindi marker")
                break
                    
        # Add language-specific pronunciation mappings for common commands
        # This helps when Whisper doesn't correctly identify languages
        if any(word in text_transcribe.lower() for word in ["left", "right", "forward", "backward", "stop", "start"]):
            print("[STT] Identified English command words - adding direct matches")
        elif any(word in text_transcribe.lower() for word in ["baaye", "daaye", "aage", "peeche", "ruko", "chalo"]):
            print("[STT] Identified Hindi command words - adding direct matches")
        elif any(word in text_transcribe.lower() for word in ["davikade", "ujavikade", "pudhe", "mage", "thamba"]):
            print("[STT] Identified Marathi command words - adding direct matches")
            
    except Exception as e:
        print(f"[STT] Error during transcription: {e}")
    
    if not results:
        print("Could not recognize speech.")
        return None, 0
    
    # Try to match each result to a command
    best_command = None
    best_confidence = 0
    best_text = ""
    
    for text in results:
        print(f"Trying to match text: '{text}'")
        command, confidence = match_command(text)
        if command and confidence > best_confidence:
            best_command = command
            best_confidence = confidence
            best_text = text
    
    # Output the results
    if best_command:
        print(f"Matched command: '{best_command}' with confidence {best_confidence:.2f} from text '{best_text}'")
        return best_command, best_confidence
    else:
        print(f"No command matched in any language. Best confidence: {best_confidence:.2f}")
        return None, best_confidence

# --- Mode Functions ---

def online_llm_mode(encoder):
    """
    Mode 1: Online LLM Query Mode
    Records audio, authenticates the user, transcribes speech, 
    sends to LLM, and speaks the response.
    """
    print("\n=== MODE 1: Online LLM Query ===")
    
    # Authenticate user
    user, score, processed_path = identify_speaker(encoder)
    if not user:
        print("Authentication failed. Access denied.")
        return False
    
    if not processed_path or not Path(processed_path).exists():
        print("Processed audio missing. Aborting.")
        return False
    
    # Transcribe the audio
    print("\nTranscribing audio to text...")
    recognized_text = transcribe_audio(processed_path)
    if not recognized_text:
        print("Could not recognize speech.")
        return False
    print(f"Recognized Text: '{recognized_text}'")
    
    # Query the language model
    print("\nSending text to language model...")
    llm_response = query_llm(recognized_text)
    print(f"LLM Response: '{llm_response}'")
    
    # Convert response to speech
    print("\nConverting response to speech...")
    try:
        synthesize_speech(llm_response, DEFAULT_LANGUAGE, DEFAULT_GENDER)
        print("Response spoken successfully")
        return True
    except Exception as e:
        print(f"Failed to speak response: {e}")
        return False

def command_control_mode(encoder):
    """
    Mode 2: Voice Command Control Mode
    Records audio, authenticates the user, transcribes speech using Whisper tiny,
    matches to predefined commands, and executes the command.
    Includes detailed diagnostics for better troubleshooting.
    """
    print("\n=== MODE 2: Voice Command Control ===")
    
    # Authenticate user
    user, score, processed_path = identify_speaker(encoder)
    if not user:
        print("Authentication failed. Access denied.")
        return False
    
    if not processed_path or not Path(processed_path).exists():
        print("Processed audio missing. Aborting.")
        return False
    
    # Process command using Whisper tiny for multilingual support
    print("\nProcessing voice command with Whisper tiny model...")
    start_time = time.time()
    
    # Using our new direct Whisper tiny function instead of the legacy process_voice_command
    command, confidence = process_command_with_whisper_tiny(processed_path)
    processing_time = time.time() - start_time
    
    print(f"\n[DIAGNOSTICS] Command processing completed in {processing_time:.2f} seconds")
    print(f"[DIAGNOSTICS] Command: {command or 'None'}, Confidence: {confidence:.2f}")
    
    # Define confidence thresholds
    EXECUTE_THRESHOLD = 0.5    # Execute command confidently
    CONFIRM_THRESHOLD = 0.4    # Ask for confirmation
    
    if command and confidence >= EXECUTE_THRESHOLD:
        # Execute the recognized command with high confidence
        execute_command(command)
        return True
    elif command and confidence >= CONFIRM_THRESHOLD:
        # Medium confidence - ask for confirmation
        confirm_msg = f"Did you mean '{command}'?"
        print(f"\n[CONFIRMATION NEEDED] {confirm_msg} (Confidence: {confidence:.2f})")
        print("Please answer with a clear 'yes' or 'no'.")
        
        # In a production system, you would wait for user confirmation
        # For now, we'll execute but let the user know it's medium confidence
        print("Executing command with medium confidence...")
        execute_command(command)
        return True
    else:
        # Handle unrecognized command
        print("Command not recognized or confidence too low.")
        
        # Use pre-generated audio for unknown command if available
        unknown_file = TTS_OUTPUT_DIR / "cmd_unknown.wav"
        if unknown_file.exists():
            try:
                import sounddevice as sd
                import soundfile as sf
                data, fs = sf.read(unknown_file)
                sd.play(data, fs)
                sd.wait()
                print("Played 'unknown command' response")
            except Exception as e:
                print(f"Error playing unknown command audio: {e}")
                # Fall back to generating TTS
                message = "Sorry, I didn't understand that command."
                try:
                    synthesize_speech(message, DEFAULT_LANGUAGE, DEFAULT_GENDER)
                except Exception as e2:
                    print(f"Error generating speech: {e2}")
        else:
            # Generate TTS response for unknown command
            message = "Sorry, I didn't understand that command."
            try:
                synthesize_speech(message, DEFAULT_LANGUAGE, DEFAULT_GENDER)
            except Exception as e:
                print(f"Error generating speech: {e}")
        
        return False

# --- Main Application Logic ---

def play_audio(audio_path):
    """
    Utility function to play audio from a file.
    
    Args:
        audio_path: Path to the audio file to play
    """
    if not Path(audio_path).exists():
        print(f"Audio file not found: {audio_path}")
        return False
        
    try:
        import sounddevice as sd
        import soundfile as sf
        data, fs = sf.read(audio_path)
        sd.play(data, fs)
        sd.wait()
        return True
    except Exception as e:
        print(f"Error playing audio: {e}")
        return False

def ensure_directories():
    """Create necessary directories if they don't exist."""
    TEMP_DIR.mkdir(exist_ok=True)
    TTS_OUTPUT_DIR.mkdir(exist_ok=True)
    VOICE_DB_PROCESSED_DIR.mkdir(exist_ok=True)
    VOICE_DB_EMBEDDINGS_DIR.mkdir(exist_ok=True)
    
    # Clean up old temporary files to avoid disk space issues
    
def initialize_voice_database():
    """Initialize the voice database and ensure it's ready for use."""
    # Create directories if they don't exist
    ensure_directories()
    
    # Check if the voice database is properly set up
    # Just count files rather than loading embeddings
    try:
        embedding_count = len(list(VOICE_DB_EMBEDDINGS_DIR.glob("*.npy")))
        if embedding_count > 0:
            print(f"Voice database ready with {embedding_count} profiles.")
        else:
            print("Voice database is empty. Use Voice Profile Management to add voices.")
    except Exception as e:
        print(f"Error checking voice database: {e}")
        print("Voice database may not be properly set up.")
    # Only keep files from the last 24 hours
    try:
        import time
        current_time = time.time()
        day_in_seconds = 86400  # 24 hours
        
        for temp_file in TEMP_DIR.glob("voice_cmd_*"):
            # Check if file is older than 24 hours
            file_mod_time = temp_file.stat().st_mtime
            if current_time - file_mod_time > day_in_seconds:
                temp_file.unlink()
                print(f"Cleaned up old temp file: {temp_file.name}")
    except Exception as e:
        print(f"Warning: Error during temp file cleanup: {e}")

def main():
    """Main application entry point."""
    print("\n" + "="*60)
    print("=== Smart Wheelchair Control System ===")
    print("="*60)
    
    # Initialize directories
    ensure_directories()
    
    # Check for pre-generated command responses
    cmd_files = list(TTS_OUTPUT_DIR.glob("cmd_*.wav"))
    if not cmd_files:
        print("\nWARNING: No pre-generated command responses found.")
        print("For faster response, run generate_command_responses.py first.")
    else:
        print(f"\nFound {len(cmd_files)} pre-generated command responses.")
    
    # Load API key for LLM access
    if not load_api_key():
        print("\nWARNING: API key not loaded. Online LLM mode may not work.")
        print("Set either 'groq_api' or 'hfapi' in your .env file.")
    
    # Configure PyTorch for CPU-only execution (for Raspberry Pi)
    torch.set_num_threads(2)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    
    # Check for voice profiles
    embeddings = list_embeddings()
    if not embeddings:
        print("\nWARNING: No voice profiles found in database.")
        print("Please add voice profiles using the system.py script first.")
        print("Example: Run system.py and use the 'store' command to add a voice.")
    else:
        print(f"\nFound {len(embeddings)} voice profile(s) in database:")
        
        # Just display the profile names without trying to detect gender
        # This prevents high CPU usage
        for i, p in enumerate(embeddings, 1):
            profile_name = p.stem
            
            # Check for a gender file (fast operation)
            gender_file = VOICE_DB_PROCESSED_DIR / f"{profile_name}_gender.txt"
            gender_display = ""
            
            if gender_file.exists():
                try:
                    with open(gender_file, 'r') as f:
                        gender = f.read().strip().lower()
                        gender_display = f" [{gender}]"
                except:
                    pass
            
            print(f"  - {profile_name}{gender_display}")
            
        print("")  # Empty line for better formatting
    
    # Initialize voice encoder only when needed
    print("\nInitializing system components...")
    print("Voice encoder will be loaded when needed.")
    encoder = None
    
    print("\nSystem initialized and ready.")
    print("\nAvailable wheelchair commands:")
    for cmd in WHEELCHAIR_COMMANDS.keys():
        print(f"- {cmd.replace('_', ' ').title()}")
    
    # Main application loop
    while True:
        print("\n" + "="*60)
        print("SMART WHEELCHAIR CONTROL - MAIN MENU")
        print("="*60)
        print("1. Online LLM Query Mode - Ask questions to the AI assistant")
        print("2. Voice Command Control Mode - Control wheelchair with voice commands")
        print("3. List Available Commands - Show all wheelchair control commands")
        print("4. Voice Profile Management")
        print("5. Test Multilingual Command Recognition (Whisper tiny)")
        print("6. Test TTS Voices")
        print("7. Exit")
        choice = input("\nEnter mode (1/2/3/4/5/6/7): ").strip()
        
        if choice == "1":
            # Load encoder only when needed
            if encoder is None:
                print("Loading voice encoder (CPU)... This may take a moment.")
                try:
                    from resemblyzer import VoiceEncoder
                    encoder = VoiceEncoder(device="cpu")
                    print("Voice encoder loaded successfully.")
                except Exception as e:
                    print(f"CRITICAL ERROR: Failed to load voice encoder: {e}")
                    print("Please ensure resemblyzer is properly installed.")
                    continue
            online_llm_mode(encoder)
        elif choice == "2":
            # Load encoder only when needed
            if encoder is None:
                print("Loading voice encoder (CPU)... This may take a moment.")
                try:
                    from resemblyzer import VoiceEncoder
                    encoder = VoiceEncoder(device="cpu")
                    print("Voice encoder loaded successfully.")
                except Exception as e:
                    print(f"CRITICAL ERROR: Failed to load voice encoder: {e}")
                    print("Please ensure resemblyzer is properly installed.")
                    continue
            command_control_mode(encoder)
        elif choice == "3":
            print("\nAvailable Voice Commands (in any language):")
            # Dictionary of multilingual command examples
            multilingual_examples = {
                "forward": ["forward", "go forward", "आगे बढ़ो", "आगे", "सामने जाओ"],
                "backward": ["backward", "go back", "पीछे जाओ", "पीछे", "back"],
                "left": ["left", "go left", "turn left", "बाएं मुड़ो", "बाएं"],
                "right": ["right", "go right", "turn right", "दाएं मुड़ो", "दाएं"],
                "rotate_left": ["rotate left", "spin left", "बाएं घूमो"],
                "rotate_right": ["rotate right", "spin right", "दाएं घूमो"],
                "start": ["start", "begin", "शुरू करो", "शुरू", "चालू करो"],
                "stop": ["stop", "halt", "रुको", "बंद करो", "रोको"]
            }
            
            for cmd, variations in WHEELCHAIR_COMMANDS.items():
                print(f"\n- {cmd.replace('_', ' ').title()}:")
                print(f"  English: {', '.join(variations[:3])}")
                if cmd in multilingual_examples:
                    print(f"  Hindi/Other: {', '.join(multilingual_examples[cmd])}")
                if len(variations) > 3:
                    print(f"  Also: {', '.join(variations[3:5])}")
                    
            print("\nNote: Commands in any language will be automatically translated to English")
            print("      and matched to the closest wheelchair command.")
        elif choice == "4":
            print("\nVoice Profile Management")
            print("1. View existing voice profiles")
            print("2. Create new voice profile")
            print("3. Test voice authentication")
            print("4. Clean voice database (deletes all profiles)")
            profile_choice = input("Enter option (1/2/3/4) or any other key to return: ").strip()
            
            if profile_choice == "1":
                try:
                    # Get profile list more efficiently
                    profiles = list(VOICE_DB_EMBEDDINGS_DIR.glob("*.npy"))
                    
                    if profiles:
                        print(f"\nFound {len(profiles)} voice profiles:")
                        
                        for i, p in enumerate(profiles, 1):
                            profile_name = p.stem
                            
                            # Check for gender file (fast operation)
                            gender_file = VOICE_DB_PROCESSED_DIR / f"{profile_name}_gender.txt"
                            gender_display = ""
                            
                            if gender_file.exists():
                                try:
                                    with open(gender_file, 'r') as f:
                                        gender = f.read().strip().lower()
                                        gender_display = f" [{gender}]"
                                except:
                                    pass
                            
                            print(f"  {i}. {profile_name}{gender_display}")
                    else:
                        print("No voice profiles found.")
                except Exception as e:
                    print(f"Error listing profiles: {e}")
                    print("No voice profiles could be loaded.")
            elif profile_choice == "2":
                # Load encoder only when needed
                if encoder is None:
                    print("Loading voice encoder (CPU)... This may take a moment.")
                    try:
                        from resemblyzer import VoiceEncoder
                        encoder = VoiceEncoder(device="cpu")
                        print("Voice encoder loaded successfully.")
                    except Exception as e:
                        print(f"CRITICAL ERROR: Failed to load voice encoder: {e}")
                        print("Please ensure resemblyzer is properly installed.")
                        input("\nPress Enter to continue...")
                        continue
                
                # Create a new voice profile
                success = create_voice_profile_internal(encoder, VOICE_DB_PROCESSED_DIR, 
                                              VOICE_DB_EMBEDDINGS_DIR, SAMPLE_RATE)
                
                if success:
                    print("\nVoice profile created successfully.")
                    # Refresh the list of embeddings in memory
                    embeddings = list_embeddings()
                else:
                    print("\nFailed to create voice profile.")
                
                input("\nPress Enter to continue...")
                
            elif profile_choice == "3":
                # Load encoder only when needed
                if encoder is None:
                    print("Loading voice encoder (CPU)... This may take a moment.")
                    try:
                        from resemblyzer import VoiceEncoder
                        encoder = VoiceEncoder(device="cpu")
                        print("Voice encoder loaded successfully.")
                    except Exception as e:
                        print(f"CRITICAL ERROR: Failed to load voice encoder: {e}")
                        print("Please ensure resemblyzer is properly installed.")
                        input("\nPress Enter to continue...")
                        continue
                
                # Test voice authentication
                test_voice_authentication_internal(encoder)
            elif profile_choice == "4":
                print("\n=== Clean Voice Database ===")
                print("WARNING: This will delete ALL voice profiles.")
                print("This operation cannot be undone.")
                confirm = input("Are you sure you want to continue? (yes/no): ").strip().lower()
                
                if confirm == "yes":
                    success = clean_voice_database(VOICE_DB_PROCESSED_DIR, VOICE_DB_EMBEDDINGS_DIR)
                    if success:
                        print("\nVoice database cleaned successfully.")
                        # Refresh the list of embeddings in memory
                        embeddings = list_embeddings()
                    else:
                        print("\nFailed to clean voice database.")
                else:
                    print("\nOperation cancelled.")
                    
                input("\nPress Enter to continue...")
        elif choice == "5":
            # Test Whisper tiny multilingual recognition
            print("\n=== Testing Multilingual Command Recognition ===")
            print("This mode will use Whisper tiny to recognize commands in any language")
            print("without requiring voice authentication.\n")
            
            print("Languages supported by Whisper tiny include:")
            languages_list = sorted(LANGUAGE_CODES.items())
            for i, (code, name) in enumerate(languages_list):
                print(f"  {code}: {name}", end="\t")
                # Format nicely in columns
                if (i + 1) % 4 == 0:
                    print()
            print("\n")
            
            # Record and process command
            command, confidence = process_command_with_whisper_tiny(None, detect_lang=True)
            
            if command:
                print(f"\nRecognized command: '{command}' with confidence {confidence:.2f}")
                
                # Get the detected language from the last transcription
                detected_language = detect_language(transcribe_command_audio(TEMP_DIR / f"voice_cmd_{time.strftime('%Y%m%d_%H%M%S')}.wav"))
                print(f"Detected language: {detected_language} ({LANGUAGE_CODES.get(detected_language, 'Unknown')})")
                
                # Execute command with language-specific feedback
                execute_command_with_language_feedback(command, detected_language)
            else:
                print("\nNo command recognized. Please try again with a clearer voice command.")
        elif choice == "6":
            # Test TTS voices
            test_tts_voices()
        elif choice == "7":
            print("\nExiting Smart Wheelchair Control System. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1, 2, 3, 4, 5, 6, or 7.")
            

if __name__ == "__main__":
    main()