#!/usr/bin/env python3
"""
Multi-Voice TTS Manager for Smart Wheelchair

This module provides a unified interface for text-to-speech synthesis
using multiple TTS engines and voice models. It supports:

1. Piper TTS (local, offline) - for English and Hindi
2. Optional voice cloning using XTTS models
3. Consistent audio playback across the application

The module handles model selection based on language and gender,
and provides fallbacks when a specific voice is not available.

Author: GitHub Copilot
Date: October 2, 2025
"""

import os
import subprocess
import time
from pathlib import Path
import sounddevice as sd
import soundfile as sf
import numpy as np
from typing import Dict, Tuple, Optional, List, Union

# Import shared components from system.py (reusing what's available)
from system import (
    MODELS_DIR,
    OPTIMIZED_DIR,
    TTS_OUTPUT_DIR,
)

# Define paths
TEMP_DIR = Path("./temp")

# Ensure these directories exist
def ensure_directories():
    """Create necessary directories if they don't exist."""
    TEMP_DIR.mkdir(exist_ok=True)
    TTS_OUTPUT_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    OPTIMIZED_DIR.mkdir(exist_ok=True)

# Voice model configuration
VOICE_MODELS = {
    "en": {
        "male": {
            "name": "en_US-ryan-medium",
            "model_path": str(MODELS_DIR / "en_US-ryan-medium.onnx"),
            "config_path": str(MODELS_DIR / "en_US-ryan-medium.onnx.json"),
        },
        "female": {
            "name": "en_US-lessac-medium",
            "model_path": str(MODELS_DIR / "en_US-lessac-medium.onnx"),
            "config_path": str(MODELS_DIR / "en_US-lessac-medium.onnx.json"),
        }
    },
    "hi": {
        "male": {
            "name": "hi_IN-rohan-medium",
            "model_path": str(MODELS_DIR / "hi_IN-rohan-medium.onnx"),
            "config_path": str(MODELS_DIR / "hi_IN-rohan-medium.onnx.json"),
        },
        "female": {
            "name": "hi_IN-priyamvada-medium",
            "model_path": str(MODELS_DIR / "hi_IN-priyamvada-medium.onnx"),
            "config_path": str(MODELS_DIR / "hi_IN-priyamvada-medium.onnx.json"),
        }
    }
}

# Define fallback order for languages and genders
LANGUAGE_FALLBACKS = {
    "mr": "hi",  # Marathi -> Hindi
    "pa": "hi",  # Punjabi -> Hindi
    "bn": "hi",  # Bengali -> Hindi
    "ta": "hi",  # Tamil -> Hindi
    "te": "hi",  # Telugu -> Hindi
    "ml": "hi",  # Malayalam -> Hindi
    "gu": "hi",  # Gujarati -> Hindi
    "kn": "hi",  # Kannada -> Hindi
    "ur": "hi",  # Urdu -> Hindi
}

GENDER_FALLBACKS = {
    "male": "female",
    "female": "male",
}

def check_piper_available() -> bool:
    """Check if Piper TTS is available on the system."""
    try:
        result = subprocess.run(['which', 'piper'], 
                              stdout=subprocess.PIPE, 
                              stderr=subprocess.PIPE)
        return result.returncode == 0
    except Exception:
        return False

def play_audio(audio_path: str) -> bool:
    """
    Play audio from a file using sounddevice.
    
    Args:
        audio_path: Path to the audio file to play
        
    Returns:
        bool: True if playback was successful, False otherwise
    """
    if not Path(audio_path).exists():
        print(f"Audio file not found: {audio_path}")
        return False
        
    try:
        # Initialize sounddevice before loading data
        sd.check_output_settings()
        
        # Small pause to ensure device is ready
        time.sleep(0.1)
        
        # Load and play audio
        data, fs = sf.read(audio_path)
        
        # Add a small bit of silence at the beginning to prevent cutting
        # Only if the audio is not too short
        if len(data) > 1000:  # Check if audio has enough samples
            silence_duration = int(0.15 * fs)  # 150ms of silence
            if len(data.shape) == 2:  # Stereo
                silence = np.zeros((silence_duration, 2), dtype=data.dtype)
            else:  # Mono
                silence = np.zeros(silence_duration, dtype=data.dtype)
            data = np.concatenate((silence, data))
        
        # Play with a slight delay to ensure device is fully ready
        sd.play(data, fs)
        sd.wait()
        return True
    except Exception as e:
        print(f"Error playing audio: {e}")
        return False

def get_voice_model(language: str, gender: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    Get the appropriate voice model for the given language and gender.
    Implements fallback mechanisms if exact match isn't available.
    
    Args:
        language: Language code (e.g., 'en', 'hi', 'mr')
        gender: Gender ('male' or 'female')
        
    Returns:
        Tuple of (model_path, config_path, voice_name) or (None, None, None) if no suitable model is found
    """
    # Try the requested language
    if language in VOICE_MODELS and gender in VOICE_MODELS[language]:
        voice_info = VOICE_MODELS[language][gender]
        model_path = voice_info["model_path"]
        config_path = voice_info["config_path"]
        
        # Check if model files exist, first in models dir, then in optimized dir
        if Path(model_path).exists() and Path(config_path).exists():
            return model_path, config_path, voice_info["name"]
        
        # Try the optimized directory as fallback
        opt_model = str(OPTIMIZED_DIR / Path(model_path).name)
        opt_config = str(OPTIMIZED_DIR / Path(config_path).name)
        
        if Path(opt_model).exists() and Path(opt_config).exists():
            return opt_model, opt_config, voice_info["name"]
    
    # Try fallback language
    fallback_lang = LANGUAGE_FALLBACKS.get(language)
    if fallback_lang in VOICE_MODELS and gender in VOICE_MODELS[fallback_lang]:
        voice_info = VOICE_MODELS[fallback_lang][gender]
        model_path = voice_info["model_path"]
        config_path = voice_info["config_path"]
        
        if Path(model_path).exists() and Path(config_path).exists():
            print(f"Using fallback language: {fallback_lang} instead of {language}")
            return model_path, config_path, voice_info["name"]
        
        # Try the optimized directory as fallback
        opt_model = str(OPTIMIZED_DIR / Path(model_path).name)
        opt_config = str(OPTIMIZED_DIR / Path(config_path).name)
        
        if Path(opt_model).exists() and Path(opt_config).exists():
            print(f"Using fallback language: {fallback_lang} instead of {language}")
            return opt_model, opt_config, voice_info["name"]
    
    # Try fallback gender with original language
    fallback_gender = GENDER_FALLBACKS.get(gender)
    if language in VOICE_MODELS and fallback_gender in VOICE_MODELS[language]:
        voice_info = VOICE_MODELS[language][fallback_gender]
        model_path = voice_info["model_path"]
        config_path = voice_info["config_path"]
        
        if Path(model_path).exists() and Path(config_path).exists():
            print(f"Using fallback gender: {fallback_gender} instead of {gender}")
            return model_path, config_path, voice_info["name"]
        
        # Try the optimized directory as fallback
        opt_model = str(OPTIMIZED_DIR / Path(model_path).name)
        opt_config = str(OPTIMIZED_DIR / Path(config_path).name)
        
        if Path(opt_model).exists() and Path(opt_config).exists():
            print(f"Using fallback gender: {fallback_gender} instead of {gender}")
            return opt_model, opt_config, voice_info["name"]
    
    # Try English as a last resort
    if language != "en" and "en" in VOICE_MODELS and gender in VOICE_MODELS["en"]:
        voice_info = VOICE_MODELS["en"][gender]
        model_path = voice_info["model_path"]
        config_path = voice_info["config_path"]
        
        if Path(model_path).exists() and Path(config_path).exists():
            print(f"Using English as a fallback for {language}")
            return model_path, config_path, voice_info["name"]
        
        # Try the optimized directory as fallback
        opt_model = str(OPTIMIZED_DIR / Path(model_path).name)
        opt_config = str(OPTIMIZED_DIR / Path(config_path).name)
        
        if Path(opt_model).exists() and Path(opt_config).exists():
            print(f"Using English as a fallback for {language}")
            return opt_model, opt_config, voice_info["name"]
    
    # Nothing worked, return None
    return None, None, None

def synthesize_speech(text: str, language: str = "en", gender: str = "male", 
                     output_file: Optional[str] = None, play: bool = True) -> Optional[str]:
    """
    Synthesize speech using Piper TTS.
    
    Args:
        text: Text to convert to speech
        language: Language code (e.g., 'en', 'hi')
        gender: Voice gender ('male' or 'female')
        output_file: Optional custom output file path
        play: Whether to play audio after synthesis
        
    Returns:
        Path to generated audio file or None if synthesis failed
    """
    ensure_directories()
    
    if not check_piper_available():
        print("Piper TTS not available. Please install it with: pip install piper-tts")
        return None
    
    # Clean text for shell command
    safe_text = text.replace('"', "'").replace('\n', ' ').strip()
    if not safe_text:
        print("Empty text provided for speech synthesis")
        return None
    
    # Get appropriate voice model
    model_path, config_path, voice_name = get_voice_model(language, gender)
    if not model_path or not Path(model_path).exists():
        print(f"No suitable voice model found for language={language}, gender={gender}")
        return None
    
    # Set output path
    if output_file:
        out_path = output_file
    else:
        timestamp = int(time.time())
        out_path = str(TTS_OUTPUT_DIR / f"tts_{language}_{gender}_{timestamp}.wav")
    
    # Create command with padding - adding silent phonemes at the start to prevent cutting
    # The "--pad-begin 0.15" option adds 150ms silence to the beginning to prevent cutoff
    cmd = f'echo "{safe_text}" | piper --model "{model_path}" --output_file "{out_path}" --pad-begin 0.15'
    
    # Execute command
    try:
        print(f"[TTS] Synthesizing speech: {safe_text[:30]}... using {voice_name}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"[TTS] Synthesis failed: {result.stderr}")
            return None
        
        print(f"[TTS] Speech generated: {out_path}")
        
        # Play the audio if requested
        if play and Path(out_path).exists():
            play_audio(out_path)
        
        return out_path
    
    except Exception as e:
        print(f"[TTS] Error during speech synthesis: {e}")
        return None

# Main function for testing
def main():
    """Test the TTS module with various languages and voices."""
    ensure_directories()
    
    # Test different languages and genders
    test_cases = [
        {"text": "Hello, this is a test of the English male voice.", "language": "en", "gender": "male"},
        {"text": "Hello, this is a test of the English female voice.", "language": "en", "gender": "female"},
        {"text": "नमस्ते, यह हिंदी पुरुष आवाज का एक परीक्षण है।", "language": "hi", "gender": "male"},
        {"text": "नमस्ते, यह हिंदी महिला आवाज का एक परीक्षण है।", "language": "hi", "gender": "female"},
        # Test fallbacks
        {"text": "नमस्कार, हा मराठी पुरुष आवाजाची चाचणी आहे.", "language": "mr", "gender": "male"},
    ]
    
    for case in test_cases:
        print(f"\nTesting: {case['language']}-{case['gender']}: {case['text']}")
        output_path = synthesize_speech(
            text=case['text'],
            language=case['language'],
            gender=case['gender'],
            play=True
        )
        if output_path:
            print(f"Success: {output_path}")
        else:
            print("Failed to synthesize speech")

if __name__ == "__main__":
    main()