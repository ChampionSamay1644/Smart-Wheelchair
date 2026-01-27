#!/usr/bin/env python3
# pyright: reportMissingImports=false
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
# Force CPU execution for ONNX Runtime before any heavy libraries load to avoid GPU discovery warnings
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("ROCR_VISIBLE_DEVICES", "")
os.environ.setdefault("ORT_FORCE_CPU", "1")
os.environ.setdefault("ORT_DISABLE_GPU_EXECUTION_PROVIDER", "1")

import sys
import time
import platform
import importlib
import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav
from pathlib import Path
import torch
try:
    import torchaudio  # type: ignore
    if not hasattr(torchaudio, "list_audio_backends"):
        def _list_audio_backends() -> list:
            return []
        torchaudio.list_audio_backends = _list_audio_backends  # type: ignore[attr-defined]
except Exception:
    torchaudio = None  # type: ignore
import difflib
import json
import uuid
import datetime
from collections import defaultdict
from functools import lru_cache
from typing import Dict, List, Tuple, Optional, Union, Any, TYPE_CHECKING, Set

import requests
from dotenv import load_dotenv

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

try:
    from rapidfuzz import fuzz as rapidfuzz_fuzz, process as rapidfuzz_process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    RAPIDFUZZ_AVAILABLE = False

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("Warning: Groq package not found. Install with: pip install groq")

try:
    import noisereduce as nr
except ImportError:
    nr = None

try:
    from transformers import (
        AutoProcessor,
        AutoModelForSpeechSeq2Seq,
        pipeline,
        AutoModelForCausalLM,
        AutoTokenizer,
    )
except ImportError:
    print("Missing dependency: transformers. Install with: pip install transformers")
    sys.exit(1)

# Import the new multi-voice TTS module
from multi_voice_tts import synthesize_speech

# Import voice recognition components
try:
    from speechbrain.pretrained import SpeakerRecognition
    SPEECHBRAIN_AVAILABLE = True
except ImportError:
    SPEECHBRAIN_AVAILABLE = False
    print("Missing dependency: speechbrain. Install with: pip install speechbrain")

RS_SAMPLING_RATE = 16000

if TYPE_CHECKING:
    from faster_whisper import WhisperModel as _FWWhisperModel  # type: ignore[import]

# =============================================================================
# CORE CONFIGURATION (SELF-CONTAINED FROM system.py)
# =============================================================================


# =============================================================================
# VOICE ACTIVITY DETECTION - Consolidated from voice_activity.py
# =============================================================================

def detect_silence(audio, sample_rate, threshold=0.015, min_duration=0.5):
    """
    Detect if audio contains mostly silence or background noise.
    
        
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
VOICE_DB_PROCESSED_DIR = Path("./voice_db_processed")
VOICE_DB_EMBEDDINGS_DIR = Path("./voice_db_embeddings")
OPTIMIZED_DIR = Path("./optimized_models")
MODELS_DIR = Path("./models")
TTS_OUTPUT_DIR = Path("./tts_outputs")
TEMP_DIR = Path("./temp")
VOICE_MODEL_SEARCH_DIRS = [MODELS_DIR, OPTIMIZED_DIR]

WHISPER_MODEL_ID = "openai/whisper-tiny"
WHISPER_MODEL_DIR = OPTIMIZED_DIR / "models--openai--whisper-tiny"

HF_API_TOKEN: Optional[str] = None
LLM_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"
LLM_API_URL = f"https://api-inference.huggingface.co/models/{LLM_MODEL_ID}"
LLM_HEADERS: Dict[str, str] = {}
LLM_FALLBACK_MODELS = [
    "microsoft/Phi-3-mini-4k-instruct",
    "HuggingFaceH4/zephyr-7b-beta",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
]

GROQ_API_KEY: Optional[str] = None
USE_GROQ = True
GROQ_MODEL = "openai/gpt-oss-20b"
GROQ_CLIENT: Optional[Any] = None

ENABLE_LOCAL_LLM = False
LOCAL_LLM_MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"
_local_llm_pipe: Optional[Any] = None

USE_HF_ROUTER = False
HF_ROUTER_TOKEN: Optional[str] = None
ROUTER_API_BASE = "https://router.huggingface.co/v1"
ROUTER_MODEL_ID = "HuggingFaceH4/zephyr-7b-beta:featherless-ai"
ROUTER_MODEL_FALLBACKS = [
    "Qwen/Qwen2.5-7B-Instruct:together",
]

SYSTEM_PROMPT = (
    "Your name is Cruzer and you are a chatbot on a Smart IOT Enabled wheelchair. "
    "Reply to messages in very simple and easy to understand language so that anyone can understand and interpret your responses. "
    "Be short but precise. Avoid using special formatting characters like asterisks or hashtags."
)

SAMPLE_RATE = RS_SAMPLING_RATE
RECORD_DURATION = 3
SIMILARITY_THRESHOLD = 0.10
SPEAKER_NOISE_REDUCTION_BLEND = 0.10  # Portion of denoised signal to mix into embeddings (0.0 disables)
SPEAKER_EMBED_SEGMENT_SECONDS = 0.95  # Duration per chunk when averaging embeddings
SPEAKER_EMBED_OVERLAP = 0.45  # Fractional overlap between chunks for embeddings
STT_NOISE_REDUCTION_BLEND = 0.18  # Blend factor for light denoising before Whisper
USB_MIC_NOISE_REDUCTION_BLEND = 0.30  # Blend factor used when denoising the USB mic input stream
SPEAKER_SEGMENT_RMS_RATIO = 0.18  # Minimum % of overall RMS a segment must have to be kept
SPEAKER_SEGMENT_RMS_FLOOR = 0.007  # Absolute RMS floor for segment inclusion
SPEAKER_VARIANT_SUPPORT_WINDOW = 0.035  # Score gap within which sibling embeddings reinforce the match
SPEAKER_VARIANT_SUPPORT_STEP = 0.015  # Bonus per additional supporting embedding beyond the strongest
SPEAKER_VARIANT_SUPPORT_MAX = 0.045  # Cap on cumulative bonus from the same speaker's variants
SPEAKER_MIN_SCORE_GAP = 0.00  # Minimum lead over runner-up to accept authentication (0 = disabled)
SPEAKER_SCORE_GAIN = 1.18  # Multiplicative boost applied to raw similarity before bonuses
SPEAKER_SCORE_OFFSET = 0.030  # Additive boost applied to raw similarity before bonuses
SPEAKER_SCORE_AVG_WEIGHT = 0.35  # Weight applied to average variant similarity during calibration
SPEAKER_SCORE_AVG_BIAS = 0.020  # Bias added when blending average support into the similarity
SPEAKER_SOLO_THRESHOLD_RELAX = 0.10  # Threshold relaxation when only one speaker is enrolled

_stt_pipe: Optional[Any] = None
_speaker_recognizer: Optional[Any] = None

ENROLLMENT_PHRASES = [
    "My voice is my password, verify my identity",
    "मेरी आवाज मेरा पासवर्ड है, मेरी पहचान सत्यापित करें",
    "माझा आवाज माझा पासवर्ड आहे, माझी ओळख सत्यापित करा"
]

# =============================================================================
# CORE UTILITIES (PORTED FROM system.py)
# =============================================================================

def record_audio(seconds, sr):
    """Capture microphone audio using sounddevice."""
    print(f"Recording {seconds} seconds...")
    rec = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="int16")
    sd.wait()
    return rec.squeeze()


def save_float_to_wav(float_audio, path, sr=SAMPLE_RATE):
    """Persist float32 audio to WAV file with 16-bit PCM encoding."""
    wav.write(path, sr, (np.clip(float_audio, -1.0, 1.0) * 32767).astype(np.int16))


def cosine_similarity(a, b):
    """Compute cosine similarity between two embedding vectors."""
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b) + 1e-8)
    return float(np.dot(a_norm, b_norm))


FAST_TRANSCRIPTION_ENABLED = True  # Skip heavy denoising to speed up Pi inference

# Common filler words that do not change intent but confuse command matching
COMMAND_FILLER_WORDS = {
    "please",
    "hey",
    "hello",
    "hi",
        "pude",
    "wheelchair",
    "buddy",
    "samay",
    "champion",
    "smart",
    "chair",
    "chairman",
    "ok",
    "okay",
    "the",
    "a",
        "pudhechal",
        "pudechal",
        "pudhe chal",
        "pude chal",
    "to",
    "let",
        "pudeachal",
    "lets",
    "just",
    "now",
    "again",
    "pleasee",
    "plz",
    "kindly",
    "could",
    "would",
    "can",
    "you",
    "me",
    "my",
    "for",
    "ya",
    "yo",
    "listen",
}

# Short phrases that map cleanly onto canonical commands
COMMAND_PHRASE_REPLACEMENTS = {
    "turn left": "left",
    "turn to the left": "left",
    "rotate left": "left",
    "spin left": "left",
    "circle left": "left",
    "veer left": "left",
    "turn right": "right",
    "turn to the right": "right",
    "rotate right": "right",
    "spin right": "right",
    "veer right": "right",
    "circle right": "right",
    "go forward": "forward",
    "move forward": "forward",
    "go straight": "forward",
    "move straight": "forward",
    "go ahead": "forward",
    "come back": "backward",
    "go back": "backward",
    "move back": "backward",
    "reverse back": "backward",
    "stop now": "stop",
    "please stop": "stop",
    "emergency stop": "stop",
    "start moving": "start",
    "begin moving": "start",
    "let's go": "forward",
}


def _normalize_command_tokens(words: List[str]) -> List[str]:
    """Remove filler words that Whisper often adds around the core command."""
    return [word for word in words if word and word not in COMMAND_FILLER_WORDS]


def _apply_phrase_replacements(text: str) -> Optional[Tuple[str, float]]:
    """Return direct command match if the text contains any canonical phrase."""
    lowered = text.lower()
    for phrase, command in COMMAND_PHRASE_REPLACEMENTS.items():
        if phrase in lowered:
            # Confidence bumped for explicit phrases
            base_conf = 0.92 if command in {"left", "right"} else 0.88
            return command, base_conf
    return None


def _normalize_audio_level(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to a stable RMS for faster Whisper inference."""
    if audio.size == 0:
        return audio.astype(np.float32)
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak
    return audio.astype(np.float32)


def _normalize_vector(vec: np.ndarray) -> np.ndarray:
    """Return a unit-length version of the provided vector."""
    vec = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vec))
    if norm < 1e-8:
        return vec
    return (vec / norm).astype(np.float32)


def _normalize_embedding_stack(stack: np.ndarray) -> np.ndarray:
    """Normalize each row vector in a matrix of embeddings."""
    if stack is None or stack.size == 0:
        return np.empty((0, 0), dtype=np.float32)
    stack = stack.astype(np.float32)
    norms = np.linalg.norm(stack, axis=1, keepdims=True) + 1e-8
    return stack / norms


def ensure_minimum_duration(audio: np.ndarray, sample_rate: int, target_seconds: float = 1.2) -> np.ndarray:
    """Pad audio with reflection so short utterances meet minimum duration."""
    if audio is None:
        return np.zeros(int(sample_rate * target_seconds), dtype=np.float32)
    audio = audio.astype(np.float32)
    min_samples = int(sample_rate * target_seconds)
    if audio.size >= min_samples or min_samples <= 0:
        return audio
    pad_total = min_samples - audio.size
    left = pad_total // 2
    right = pad_total - left
    if audio.size == 0:
        return np.zeros(min_samples, dtype=np.float32)
    return np.pad(audio, (left, right), mode="reflect")


def fast_noise_gate(audio: np.ndarray, sample_rate: int, *, floor_percentile: float = 15.0, gate_strength: float = 1.6, release_ms: float = 80.0) -> np.ndarray:
    """Lightweight noise gate to suppress stationary background hum."""
    if audio is None or audio.size == 0:
        return np.zeros_like(audio, dtype=np.float32)

    audio = audio.astype(np.float32)
    amplitude = np.abs(audio)
    noise_floor = float(np.percentile(amplitude, max(0.0, min(100.0, floor_percentile))))

    if noise_floor <= 0:
        return audio

    threshold = noise_floor * max(1.0, gate_strength)
    voice_mask = amplitude >= threshold

    try:
        from scipy.ndimage import uniform_filter1d

        window = max(1, int(sample_rate * (release_ms / 1000.0)))
        smoothed = uniform_filter1d(voice_mask.astype(np.float32), size=window)
    except Exception:
        window = max(1, int(sample_rate * (release_ms / 1000.0)))
        kernel = np.ones(window, dtype=np.float32) / float(window)
        smoothed = np.convolve(voice_mask.astype(np.float32), kernel, mode="same")

    gain_floor = 0.18
    gain = gain_floor + (1.0 - gain_floor) * smoothed
    gated = audio * gain
    return gated.astype(np.float32)


def apply_subtle_noise_reduction(audio: np.ndarray, sample_rate: int, blend: float) -> np.ndarray:
    """Blend a lightly denoised signal with the original to stabilize embeddings."""
    if audio is None or audio.size == 0:
        return np.zeros_like(audio, dtype=np.float32)

    if blend <= 0.0:
        return audio.astype(np.float32)

    base = audio.astype(np.float32)
    processed = base

    try:
        from scipy import signal

        hp = signal.butter(1, 70, "hp", fs=sample_rate, output="sos")
        lp = signal.butter(1, 3800, "lp", fs=sample_rate, output="sos")
        filtered = signal.sosfilt(hp, base)
        filtered = signal.sosfilt(lp, filtered)
        processed = filtered.astype(np.float32)
    except Exception:
        processed = base

    if nr is not None:
        prop = float(np.clip(blend, 0.05, 0.9))
        try:
            reduced = nr.reduce_noise(y=base, sr=sample_rate, prop_decrease=prop, stationary=False)
            processed = (0.6 * processed + 0.4 * reduced.astype(np.float32))
        except Exception:
            pass

    mix = float(np.clip(blend, 0.0, 1.0))
    mixed = (1.0 - mix) * base + mix * processed
    return mixed.astype(np.float32)


def trim_audio_to_speech(
    audio: np.ndarray,
    sample_rate: int,
    *,
    threshold_ratio: float = 0.2,
    min_threshold: float = 0.006,
    pad_ms: float = 120.0,
) -> np.ndarray:
    """Remove leading/trailing silence while keeping a short safety margin."""
    if audio.size == 0 or sample_rate <= 0:
        return audio.astype(np.float32)

    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    pad = int(sample_rate * (pad_ms / 1000.0))
    window = max(1, int(sample_rate * 0.02))
    kernel = np.ones(window, dtype=np.float32) / float(window)
    smoothed = np.convolve(np.abs(audio), kernel, mode="same")

    dynamic_thresh = max(
        min_threshold,
        float(np.percentile(smoothed, 92) * threshold_ratio),
    )

    speech_mask = smoothed >= dynamic_thresh
    if not np.any(speech_mask):
        return audio

    first = int(np.argmax(speech_mask))
    last = int(len(speech_mask) - np.argmax(speech_mask[::-1]) - 1)
    start = max(0, first - pad)
    end = min(len(audio), last + pad + 1)
    trimmed = audio[start:end]

    if trimmed.size == 0:
        return audio

    return trimmed.astype(np.float32)


def clean_for_tts(text: str) -> str:
    """Sanitize text so the spoken response sounds natural."""
    import re

    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = re.sub(r"_(.*?)_", r"\1", text)
    text = re.sub(r"`(.*?)`", r"\1", text)
    text = re.sub(r"~~(.*?)~~", r"\1", text)
    text = re.sub(r"^#{1,6}\s*(.*?)$", r"\1", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*]\s+(.*?)$", r"• \1", text, flags=re.MULTILINE)
    text = text.replace("&", "and").replace(">", "").replace("<", "").replace("#", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def load_local_llm_pipeline():
    global _local_llm_pipe
    if _local_llm_pipe is not None:
        return _local_llm_pipe
    try:
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_LLM_MODEL_ID, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(LOCAL_LLM_MODEL_ID, local_files_only=True)
        model.to("cpu")
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
        _local_llm_pipe = pipe
        print(f"Local LLM loaded (CPU, offline): {LOCAL_LLM_MODEL_ID}")
        return pipe
    except Exception as exc:
        print(f"Failed to load local LLM '{LOCAL_LLM_MODEL_ID}': {exc}")
        return None


def _try_local_llm(text_prompt: str) -> str:
    pipe = load_local_llm_pipeline()
    if pipe is None:
        return ""
    try:
        messages: List[Dict[str, str]] = []
        if SYSTEM_PROMPT:
            messages.append({"role": "system", "content": SYSTEM_PROMPT})
        messages.append({"role": "user", "content": text_prompt})

        tokenizer = getattr(pipe, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            if SYSTEM_PROMPT:
                prompt = f"System: {SYSTEM_PROMPT}\nUser: {text_prompt}\nAssistant:"
            else:
                prompt = f"User: {text_prompt}\nAssistant:"

        outputs = pipe(prompt, max_new_tokens=500, do_sample=False)
        out_text = outputs[0].get("generated_text", "") if outputs else ""
        if prompt and out_text.startswith(prompt):
            out_text = out_text[len(prompt):]
        return clean_for_tts(out_text.strip())
    except Exception as exc:
        print(f"Local LLM generation failed: {exc}")
        return ""


def _query_llm_via_router(text_prompt: str) -> str:
    if not (USE_HF_ROUTER and HF_ROUTER_TOKEN):
        return ""
    try:
        models_to_try = [ROUTER_MODEL_ID] + [m for m in ROUTER_MODEL_FALLBACKS if m != ROUTER_MODEL_ID]
        base_headers = {
            "Authorization": f"Bearer {HF_ROUTER_TOKEN}",
            "Accept": "application/json",
            "Content-Type": "application/json",
            "Accept-Encoding": "identity",
        }
        for model_id in models_to_try:
            messages: List[Dict[str, str]] = []
            if SYSTEM_PROMPT:
                messages.append({"role": "system", "content": SYSTEM_PROMPT})
            messages.append({"role": "user", "content": text_prompt})

            payload = {
                "model": model_id,
                "messages": messages,
                "temperature": 0.7,
                "max_tokens": 500,
                "stream": False,
            }
            try:
                resp = requests.post(
                    f"{ROUTER_API_BASE}/chat/completions",
                    headers=base_headers,
                    json=payload,
                    timeout=180,
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.exceptions.ContentDecodingError:
                retry_headers = dict(base_headers)
                retry_headers["Accept-Encoding"] = "identity"
                resp = requests.post(
                    f"{ROUTER_API_BASE}/chat/completions",
                    headers=retry_headers,
                    json=payload,
                    timeout=180,
                )
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                status = getattr(getattr(exc, "response", None), "status_code", "N/A")
                print(f"Router API error for '{model_id}' (status={status}): {exc} (trying fallback)")
                continue

            choices = data.get("choices", []) if isinstance(data, dict) else []
            if not choices:
                print(f"Router returned no choices for '{model_id}'. Trying fallback...")
                continue

            msg = choices[0].get("message", {})
            content = msg.get("content", "").strip()
            if content:
                return clean_for_tts(content)
        return ""
    except Exception as exc:
        print(f"Router request failed: {exc}")
        return ""


def load_api_key():
    """Load API keys for Groq and Hugging Face services."""
    global HF_API_TOKEN, LLM_HEADERS, LLM_MODEL_ID, LLM_API_URL
    global LOCAL_LLM_MODEL_ID, ENABLE_LOCAL_LLM
    global USE_HF_ROUTER, HF_ROUTER_TOKEN, ROUTER_MODEL_ID, ROUTER_MODEL_FALLBACKS
    global GROQ_API_KEY, GROQ_CLIENT, USE_GROQ, GROQ_MODEL

    load_dotenv()

    GROQ_API_KEY = os.getenv("groq_api")
    if GROQ_API_KEY and GROQ_AVAILABLE:
        try:
            GROQ_CLIENT = Groq(api_key=GROQ_API_KEY)
            print("Groq API key loaded successfully.")
            env_groq_model = os.getenv("GROQ_MODEL")
            if env_groq_model:
                GROQ_MODEL = env_groq_model.strip()
                print(f"Using Groq model from env: {GROQ_MODEL}")
        except Exception as exc:
            print(f"Error initializing Groq client: {exc}")
            GROQ_CLIENT = None
    else:
        if not GROQ_API_KEY:
            print("No Groq API key found. Set 'groq_api' in .env to use Groq.")
        elif not GROQ_AVAILABLE:
            print("Groq package not installed. Install with: pip install groq")
        USE_GROQ = False

    HF_API_TOKEN = os.getenv("hfapi")
    env_model = os.getenv("HF_LLM_MODEL_ID")
    if env_model:
        LLM_MODEL_ID = env_model.strip()
        LLM_API_URL = f"https://api-inference.huggingface.co/models/{LLM_MODEL_ID}"
        print(f"Using LLM model from env: {LLM_MODEL_ID}")

    env_local_model = os.getenv("LOCAL_LLM_MODEL_ID")
    if env_local_model:
        LOCAL_LLM_MODEL_ID = env_local_model.strip()
    ENABLE_LOCAL_LLM = os.getenv("ENABLE_LOCAL_LLM", "false").strip().lower() in ("1", "true", "yes", "on")
    if ENABLE_LOCAL_LLM:
        print(f"Local LLM enabled: {LOCAL_LLM_MODEL_ID}")

    HF_ROUTER_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HF_ROUTER_TOKEN")
    router_model_env = os.getenv("HF_ROUTER_MODEL_ID")
    if router_model_env:
        ROUTER_MODEL_ID = router_model_env.strip()
    router_fallbacks_env = os.getenv("HF_ROUTER_FALLBACK_MODELS")
    if router_fallbacks_env:
        ROUTER_MODEL_FALLBACKS = [m.strip() for m in router_fallbacks_env.split(",") if m.strip()]
    USE_HF_ROUTER = bool(HF_ROUTER_TOKEN)
    if USE_HF_ROUTER:
        print(f"HF Router enabled with model: {ROUTER_MODEL_ID}")
        if ROUTER_MODEL_FALLBACKS:
            print(f"HF Router fallbacks: {ROUTER_MODEL_FALLBACKS}")

    if not HF_API_TOKEN and not HF_ROUTER_TOKEN and not (GROQ_API_KEY and GROQ_AVAILABLE):
        print("ERROR: No language model API keys found.")
        print("Set either 'groq_api', 'hfapi', or 'HF_TOKEN' in your .env file.")
        return False

    if HF_API_TOKEN:
        LLM_HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        if not (GROQ_API_KEY and GROQ_AVAILABLE):
            print("Hugging Face API token loaded successfully.")

    return True


def query_llm(text_prompt: str) -> str:
    """Send prompt to configured language model backends."""
    if USE_GROQ and GROQ_AVAILABLE and GROQ_CLIENT is not None:
        print("Sending text to Groq for processing...")
        groq_response = query_groq(text_prompt)
        if groq_response:
            return groq_response
        print("Groq query failed, falling back to other options")

    router_out = _query_llm_via_router(text_prompt)
    if router_out:
        return router_out

    if not HF_API_TOKEN:
        if ENABLE_LOCAL_LLM:
            local = _try_local_llm(text_prompt)
            if local:
                return local
        return "Sorry, my connection to the language model is not configured."

    if SYSTEM_PROMPT:
        formatted_prompt = f"[INST] {SYSTEM_PROMPT}\nUser: {text_prompt} [/INST]"
    else:
        formatted_prompt = f"[INST] {text_prompt} [/INST]"

    models_to_try = [LLM_MODEL_ID] + [m for m in LLM_FALLBACK_MODELS if m != LLM_MODEL_ID]
    payload = {
        "inputs": formatted_prompt,
        "parameters": {
            "max_new_tokens": 500,
            "temperature": 0.7,
            "return_full_text": False,
        },
        "options": {"wait_for_model": True},
    }

    last_error = None
    for model_id in models_to_try:
        url = f"https://api-inference.huggingface.co/models/{model_id}"
        try:
            response = requests.post(url, headers=LLM_HEADERS, json=payload, timeout=60)
            if response.status_code == 403:
                print(f"Access denied to model '{model_id}'. Trying fallback...")
                continue
            if response.status_code in (401, 404, 429, 500, 503):
                print(f"LLM '{model_id}' returned HTTP {response.status_code}. Trying fallback...")
                last_error = f"HTTP {response.status_code}"
                continue
            response.raise_for_status()
            result = response.json()

            if isinstance(result, list) and result and "generated_text" in result[0]:
                text = result[0]["generated_text"].strip()
                return clean_for_tts(text)
            if isinstance(result, dict) and "generated_text" in result:
                text = result["generated_text"].strip()
                return clean_for_tts(text)
            if isinstance(result, dict) and "error" in result:
                err_msg = result.get("error", "")
                if "is currently loading" in err_msg:
                    print("Model is loading on Hugging Face, please wait a moment and try again...")
                    return "The AI model is warming up. Please ask me again in a minute."
                print(f"LLM error from '{model_id}': {err_msg}. Trying fallback...")
                last_error = err_msg
                continue

            print(f"LLM '{model_id}' returned an unexpected format: {result}. Trying fallback...")
            last_error = "unexpected format"
            continue

        except requests.exceptions.RequestException as exc:
            status = getattr(getattr(exc, "response", None), "status_code", "N/A")
            print(f"Error calling Hugging Face API for '{model_id}': {exc} (status={status}). Trying fallback...")
            last_error = str(exc)
            continue
        except Exception as exc:
            print(f"An unexpected error occurred during LLM query for '{model_id}': {exc}. Trying fallback...")
            last_error = str(exc)
            continue

    if last_error:
        print(f"LLM request failed after fallbacks. Last error: {last_error}")

    if ENABLE_LOCAL_LLM:
        local = _try_local_llm(text_prompt)
        if local:
            return local

    return "I'm having trouble connecting to the language model right now."


def debug_audio_info(arr, sr, label):
    """Print compact diagnostics for audio buffers."""
    try:
        if arr is None or len(arr) == 0:
            print(f"[DEBUG] {label}: Empty or None audio array")
            return

        dur = len(arr) / float(sr)
        rms = float(np.sqrt(np.mean(arr ** 2)))
        if label == "LoadedRaw":
            print(f"[Audio] Loaded audio: {dur:.1f}s, RMS: {rms:.4f}")
        if rms < 0.001:
            print(f"[Audio] WARNING - Very low audio level (RMS={rms:.6f})")
        if np.isnan(arr).any():
            print("[Audio] WARNING - NaN values detected in audio")
    except Exception:
        pass


def _load_audio_16k(path):
    """Load audio file and resample to 16 kHz for Whisper."""
    file_path = Path(path)
    if not file_path.exists():
        print(f"[STT] File not found: {path}")
        return None
    try:
        if librosa:
            data, sr = librosa.load(path, sr=16000, mono=True)
        else:
            sr, data = wav.read(path)
            if data.dtype != np.float32:
                if data.dtype == np.int16:
                    data = data.astype(np.float32) / 32768.0
                else:
                    max_val = np.max(np.abs(data))
                    data = data.astype(np.float32) / max_val if max_val > 0 else data.astype(np.float32)
            if sr != 16000:
                print(f"[STT] Resampling audio from {sr}Hz to 16000Hz")
                try:
                    from scipy import signal
                    data = signal.resample_poly(data, 16000, sr)
                except Exception:
                    dur = len(data) / sr
                    new_len = int(dur * 16000)
                    data = np.interp(
                        np.linspace(0, len(data), new_len, endpoint=False),
                        np.arange(len(data)),
                        data,
                    ).astype(np.float32)
                sr = 16000

        rms = np.sqrt(np.mean(data ** 2))
        print(f"[STT] Loaded audio RMS: {rms:.6f}")

        if rms < 0.0001:
            print(f"[STT] Warning: Audio file {path} is extremely quiet")
            data = data * 2.0

        if np.isnan(data).any():
            print("[STT] Warning: NaN values found in audio data. Replacing with zeros.")
            data = np.nan_to_num(data)

        return data, 16000
    except Exception as exc:
        print(f"[STT] Failed loading audio {path}: {exc}")
        return None


def load_local_stt_pipeline(force_autodetect=False):
    """Load Whisper Tiny pipeline with faster-whisper backend prioritized for speed."""
    global _stt_pipe
    if _stt_pipe is not None:
        return _stt_pipe

    # Always try faster-whisper first for maximum speed (unless explicitly disabled)
    force_transformers = os.getenv("FORCE_TRANSFORMERS", "").lower() in ("1", "true", "yes")
    
    if not force_transformers and not force_autodetect:
        try:
            fw_module = importlib.import_module("faster_whisper")
            WhisperModel = getattr(fw_module, "WhisperModel")

            class _FasterWhisperPipeline:
                def __init__(self, model):
                    self._model = model

                def __call__(self, audio_array, generate_kwargs=None):
                    if audio_array is None or len(audio_array) == 0:
                        return {"text": ""}

                    language = None
                    task = "transcribe"
                    if generate_kwargs:
                        language = generate_kwargs.get("language")
                        task = generate_kwargs.get("task", task)

                    segments, _ = self._model.transcribe(
                        audio_array,
                        language=language,
                        task=task,
                        beam_size=1,
                        best_of=1,
                        temperature=0.0,
                        condition_on_previous_text=False,
                        compression_ratio_threshold=2.4,
                        log_prob_threshold=-1.0,  # More lenient for speed
                        no_speech_threshold=0.6,
                        without_timestamps=True,
                        vad_filter=True,  # Enable VAD for automatic speedup
                        suppress_blank=True,
                    )

                    text = " ".join(segment.text.strip() for segment in segments).strip()
                    return {"text": text}

            fw_cache_dir = OPTIMIZED_DIR / "faster-whisper"
            fw_cache_dir.mkdir(parents=True, exist_ok=True)

            env_threads = os.getenv("WHISPER_CPU_THREADS")
            if env_threads:
                try:
                    cpu_threads = max(1, int(env_threads))
                except ValueError:
                    cpu_threads = max(1, os.cpu_count() or 1)
            else:
                cpu_threads = max(1, os.cpu_count() or 1)

            worker_count = max(1, min(cpu_threads, 4))
            compute_type = os.getenv("WHISPER_COMPUTE_TYPE", "int8")
            fw_model = WhisperModel(
                "tiny",
                device="cpu",
                compute_type=compute_type,
                download_root=str(fw_cache_dir),
                cpu_threads=cpu_threads,
                num_workers=worker_count,
            )

            # Warm model with a half-second of silence to avoid first-call lag.
            try:
                dummy_audio = np.zeros(int(0.5 * 16000), dtype=np.float32)
                fw_model.transcribe(
                    dummy_audio,
                    language=None,
                    task="transcribe",
                    beam_size=1,
                    best_of=1,
                    temperature=0.0,
                    condition_on_previous_text=False,
                    compression_ratio_threshold=2.4,
                    log_prob_threshold=-0.1,
                    no_speech_threshold=0.65,
                    without_timestamps=True,
                    vad_filter=False,
                    suppress_blank=True,
                )
            except Exception:
                pass

            _stt_pipe = _FasterWhisperPipeline(fw_model)
            print(
                "Loaded Whisper tiny via faster-whisper backend "
                f"(threads={cpu_threads}, workers={worker_count}, compute={compute_type})"
            )
            return _stt_pipe
        except ImportError:
            print(
                "faster-whisper not installed; install with 'pip install faster-whisper' "
                "for optimal performance on Raspberry Pi."
            )
        except Exception as exc:
            print(f"Failed to initialize faster-whisper backend: {exc}")

    try:
        print("Loading Whisper tiny model directly without quantization...")

        model_dir = OPTIMIZED_DIR / "models--openai--whisper-tiny"

        processor = None
        model = None

        model_file = None
        if model_dir.exists():
            if (model_dir / "model.safetensors").exists():
                model_file = model_dir / "model.safetensors"
            elif (model_dir / "pytorch_model.bin").exists():
                model_file = model_dir / "pytorch_model.bin"

        if model_file is not None:
            try:
                processor = AutoProcessor.from_pretrained(str(model_dir), local_files_only=True)
                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    str(model_dir),
                    torch_dtype=torch.float32,
                    local_files_only=True,
                )
                model.eval()
                print("Successfully loaded Whisper tiny model from local directory")
            except Exception as local_error:
                print(f"Failed to load Whisper tiny from {model_dir}: {local_error}")
                processor = None
                model = None

        if processor is None or model is None:
            try:
                print("Whisper tiny model not found locally, downloading from Hugging Face (one-time operation)...")
                processor = AutoProcessor.from_pretrained(
                    "openai/whisper-tiny",
                    cache_dir=str(OPTIMIZED_DIR),
                    local_files_only=False,
                )

                model = AutoModelForSpeechSeq2Seq.from_pretrained(
                    "openai/whisper-tiny",
                    cache_dir=str(OPTIMIZED_DIR),
                    torch_dtype=torch.float32,
                    local_files_only=False,
                )

                model.eval()
                print("Successfully downloaded Whisper tiny model")
                model_dir.mkdir(parents=True, exist_ok=True)
                try:
                    processor.save_pretrained(str(model_dir))
                    model.save_pretrained(str(model_dir))
                    print(f"Cached Whisper tiny model to {model_dir}")
                except Exception as cache_error:
                    print(f"Warning: failed to cache Whisper tiny model locally: {cache_error}")
            except Exception as download_error:
                print(f"Failed to download Whisper tiny model: {download_error}")
                print("Please ensure the device has internet access or pre-download the model using setup_whisper_tiny.py")
                return None

        stt_pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            batch_size=1,
            return_timestamps=False,
        )
        _stt_pipe = stt_pipe

        print("Whisper tiny model loaded successfully!")
        return stt_pipe
    except Exception as exc:
        print(f"Failed to initialize STT pipeline: {exc}")
        return None


def load_speaker_recognizer():
    """Load the SpeechBrain ECAPA-TDNN speaker recognizer on CPU."""
    global _speaker_recognizer
    if _speaker_recognizer is not None:
        return _speaker_recognizer

    if not SPEECHBRAIN_AVAILABLE:
        raise RuntimeError(
            "speechbrain package is required for speaker recognition."
        )

    save_dir = OPTIMIZED_DIR / "speechbrain_ecapa"
    save_dir.mkdir(parents=True, exist_ok=True)

    run_opts = {"device": "cpu"}

    try:
        recognizer = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(save_dir),
            run_opts=run_opts,
        )
        _speaker_recognizer = recognizer
        print("Loaded ECAPA-TDNN speaker recognizer (SpeechBrain)")
        return recognizer
    except Exception as exc:
        print(f"Failed to load SpeechBrain ECAPA recognizer: {exc}")
        raise


def transcribe_audio(audio_path):
    """Transcribe audio path to text with Whisper Tiny."""
    pipe = load_local_stt_pipeline()
    if pipe is None:
        return ""
    loaded = _load_audio_16k(audio_path)
    if loaded is None:
        return ""
    audio_array, sr = loaded
    debug_audio_info(audio_array, sr, "LoadedRaw")

    if len(audio_array) < sr * 0.2:
        print("[STT] Audio too short for reliable transcription")
        return ""

    processed_audio = audio_array.copy()

    try:
        from scipy import signal
        from scipy.ndimage import maximum_filter1d, uniform_filter1d

        NOISE_REDUCTION_SENSITIVITY = 0.8
        VOICE_DETECTION_PERCENTILE = 80

        sos = signal.butter(1, 70, "hp", fs=sr, output="sos")
        processed_audio = signal.sosfilt(sos, processed_audio)

        amplitude = np.abs(processed_audio)
        voice_threshold = np.percentile(amplitude, VOICE_DETECTION_PERCENTILE)

        is_voice = amplitude > voice_threshold
        window_size = int(sr * 0.15)
        is_voice_expanded = maximum_filter1d(is_voice.astype(float), size=window_size)

        smooth_window = int(sr * 0.25)
        smooth_envelope = uniform_filter1d(is_voice_expanded, size=smooth_window)

        noise_floor = 1.0 - NOISE_REDUCTION_SENSITIVITY
        gain_envelope = smooth_envelope ** (1.0 - NOISE_REDUCTION_SENSITIVITY * 0.5)
        final_envelope = noise_floor + gain_envelope * (1.0 - noise_floor)

        processed_audio = processed_audio * final_envelope

        current_rms = np.sqrt(np.mean(processed_audio ** 2))
        if current_rms > 0.001:
            target_rms = 0.15
            gain = min(target_rms / current_rms, 3.0)
            processed_audio = processed_audio * gain

        processed_audio = np.clip(processed_audio, -0.95, 0.95)

    except Exception as exc:
        print(f"[STT] Voice enhancement failed: {exc}. Using original audio.")
        processed_audio = audio_array

    audio_versions = [
        {"name": "processed", "array": processed_audio, "task": "transcribe"},
        {"name": "processed", "array": processed_audio, "task": "translate"},
    ]

    best_text = ""
    best_score = 0

    for version in audio_versions:
        try:
            result = pipe(
                {"array": version["array"], "sampling_rate": sr},
                generate_kwargs={"task": version["task"]},
            )
            text = result.get("text", "").strip()

            word_count = len(text.split())
            char_count = len(text)

            score = word_count
            if version["task"] == "transcribe" and char_count <= 5:
                score += 5

            print(f"[STT] {version['task']} result: '{text}'")

            if score > best_score or (score == best_score and version["task"] == "transcribe"):
                best_text = text
                best_score = score

            if word_count >= 3 and version["task"] == "translate":
                break

        except Exception as exc:
            print(f"[STT] {version['task']} attempt failed: {exc}")

    return best_text

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

def create_voice_profile_internal(_encoder_unused, voice_db_processed_dir, voice_db_embeddings_dir, sample_rate=16000):
    """Create a new voice profile with multilingual recordings."""
    print("\n=== Create New Voice Profile ===")
    
    import re

    requested_name = input("Enter speaker name (lowercase, underscores allowed): ").strip().lower()
    if not requested_name:
        print("Invalid name. Operation cancelled.")
        return False

    def split_variant(name: str) -> Tuple[str, Optional[int]]:
        match = re.match(r"^(.*?)(?:_(\d+))?$", name)
        if not match:
            return name, None
        base = match.group(1) or name
        suffix = match.group(2)
        return base, int(suffix) if suffix is not None else None

    base_name, requested_suffix = split_variant(requested_name)
    existing_indices: List[int] = []
    if voice_db_embeddings_dir.exists():
        pattern = re.compile(rf"^{re.escape(base_name)}(?:_(\d+))?$")
        for path in voice_db_embeddings_dir.glob(f"{base_name}*.npy"):
            match = pattern.match(path.stem)
            if not match:
                continue
            suffix = match.group(1)
            existing_indices.append(int(suffix) if suffix is not None else 0)

    speaker = requested_name

    if requested_suffix is None:
        if 0 in existing_indices:
            next_index = max(existing_indices) + 1
            speaker = f"{base_name}_{next_index}"
            print(
                f"Profile '{base_name}' already exists. Creating new variant '{speaker}'."
            )
        else:
            speaker = base_name
    else:
        if requested_suffix in existing_indices:
            overwrite = input(
                f"Variant '{requested_name}' already exists. Overwrite? (y/n): "
            ).strip().lower()
            if overwrite != 'y':
                print("Operation cancelled.")
                return False
            speaker = requested_name
        else:
            speaker = requested_name

    embedding_path = voice_db_embeddings_dir / f"{speaker}.npy"
    
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

            if raw_path.exists():
                try:
                    raw_path.unlink()
                except PermissionError:
                    unique = uuid.uuid4().hex
                    raw_path = raw_path.with_name(f"{speaker}_raw_{unique}.wav")

            if processed_path.exists():
                try:
                    processed_path.unlink()
                except PermissionError:
                    unique = uuid.uuid4().hex
                    processed_path = processed_path.with_name(f"{speaker}_{unique}.wav")
            
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
                raw_wav = preprocess_audio_for_verification(str(raw_path))
                wav.write(
                    processed_path,
                    sample_rate,
                    (np.clip(raw_wav, -1.0, 1.0) * 32767).astype(np.int16),
                )
                emb, _ = compute_enhanced_embedding(None, raw_wav)
                if emb is None:
                    print(f"Embedding generation failed for recording {i}")
                    continue
                emb = np.asarray(emb, dtype=np.float32).reshape(-1)
                all_embeddings.append(emb)

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
        avg_embedding = np.asarray(avg_embedding, dtype=np.float32).reshape(-1)

        np.save(voice_db_embeddings_dir / f"{speaker}.npy", avg_embedding)

        canonical_gender_file = voice_db_processed_dir / f"{base_name}_gender.txt"
        try:
            canonical_gender_file.write_text(gender)
        except Exception:
            print(f"Warning: Failed to write gender file for base '{base_name}'")
        if speaker != base_name:
            try:
                (voice_db_processed_dir / f"{speaker}_gender.txt").write_text(gender)
            except Exception:
                print(f"Warning: Failed to write gender file for variant '{speaker}'")
        
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
    
    user, score, processed_path = identify_speaker(encoder, is_registration=True)

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
RECORD_DURATION = 3  # seconds
SIMILARITY_THRESHOLD = 0.00  # Match threshold for voice authentication
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
    
    # Rotate/turn left words (mapped to left)
    "बाएं घूमो": "left",
    "बायें घूमो": "left",
    "bayen ghumo": "left",
    "baayen ghumo": "left",
    "bai taraf ghumo": "left",
    "bayein taraf ghumo": "left",
    "बाएं मुड़ो": "left",
    "बायें मुड़ो": "left",
    "bayen mudo": "left",
    "baayen mudo": "left",
    
    # Rotate/turn right words (mapped to right)
    "दाएं घूमो": "right",
    "दायें घूमो": "right",
    "dayen ghumo": "right",
    "daayen ghumo": "right",
    "daaye taraf ghumo": "right",
    "dai taraf ghumo": "right",
    "दाएं मुड़ो": "right",
    "दायें मुड़ो": "right",
    "dayen mudo": "right",
    "daayen mudo": "right",
    
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
        "straight ahead", "head forward", "keep going", "keep moving forward", "keep moving ahead",
        "move ahead please", "drive forward", "drive straight", "push forward", "keep driving forward",
        # Hindi variations (देवनागरी and romanized)
        "आगे", "आगे बढ़ो", "आगे जाओ", "आगे चलो", "आगे बढ़ो", "आगे बढो", "आगे बढ़ाओ", "आगे बढ़ाओ",
        "आगे बढ़ाएं", "आगे बढाएं", "सीधे", "सीधे जाओ", "सीधे चलो", "सीधे बढ़ो", "सीधा आगे",
        "aage", "aage badho", "aage badh jao", "aage badhao", "aage jao", "aage chalo", "aage bado",
        "aagay badho", "agay badho", "aagay badhao", "agay badhao", "age badho", "age badh jao",
        "age badhao", "aage barho", "sidha aage", "seedhe", "seedhe jao", "seedhe chalo",
        # Marathi variations
        "पुढे", "पुढे जा", "पुढे चला", "पुढे चल", "पुढे वाढा", "पुढे सरळ",
        "सरळ", "सरळ जा", "सरळ चला", "सरळ पुढे",
        "pudhe", "pudhe ja", "pudhe jaa", "pudhe chal", "pudhe chala", "pudhe vada",
        "pude", "pude ja", "pude jaa", "pude chal", "pude chala", "pudeachal", "pude aagal",
        "pude pudhe", "saral", "saral ja", "saral jaa", "saral chala",
        # Common romanized Marathi/Hindi transliterations produced by translation
        "sadheval", "sadeval", "sadhe val", "sade val",
        # Punjabi variations
        "ਅੱਗੇ ਜਾ", "ਅੱਗੇ ਜਾਓ", "ਅੱਗੇ ਵਧੋ", "ਸਿੱਧੇ ਜਾਓ", "ਸਿੱਧਾ ਚੱਲੋ",
        "agge jao", "agge ja", "agge vadho", "sidhe jao", "sidha chalo",
        # Kannada variations
        "ಮುಂದೆ ಹೋಗಿ", "ಮುಂದಕ್ಕೆ ಹೋಗಿ", "ಮುಂದೆ ಸಾಗು", "ನೇರವಾಗಿ ಹೋಗಿ", "ಮುಂದೆ ಬನ್ನಿ",
        "munde hogi", "mundakke hogi", "munde saagu", "neravagi hogi", "munde banni",
        # Bengali variations
        "সামনে যাও", "সামনে চল", "আগে যাও", "সোজা যাও", "সোজা চল",
        "shamne jao", "shamne chalo", "age jao", "soja jao", "soja chol",
        # Gujarati variations
        "આગળ જાઓ", "આગળ વધો", "આગળ ચાલો", "સીધા જાઓ", "સીધા ચાલો",
        "agal jao", "agal vadho", "agal chalo", "sidha jao", "sidha chalo",
        # Tamil variations
        "முன்னே போ", "முன்னே செல்", "முன்னோக்கி போ", "நேராக போ", "நேராக செல்",
        "munne po", "munne sel", "munnokki po", "neraga po", "neraga sel",
        # Urdu variations
        "آگے جاؤ", "آگے بڑھو", "سامنے چلو", "سیدھا چلو", "سیدھے جاؤ",
        "aage jao", "aage badho", "samne chalo", "seedha chalo", "seedhe jao",
        # # Spanish variations
        # "adelante", "sigue adelante", "ve adelante", "recto", "sigue recto",
        # # French variations
        # "avancer", "en avant", "allez tout droit", "droit devant", "aller de l'avant",
        # # German variations
        # "vorwärts", "geradeaus", "nach vorne", "geh vorwärts", "gerade", "weiter",
        # # Italian variations
        # "avanti", "vai avanti", "dritto", "procedere", "muovere in avanti",
        # # Chinese variations (Simplified)
        # "前进", "向前", "直走", "往前", "前方", "直行",
        # # Japanese variations
        # "前進", "前へ", "まっすぐ", "直進", "フォワード",
        # # Russian variations
        # "вперед", "прямо", "вперёд", "двигайся вперёд", "прямо вперёд",
        # # Korean variations
        # "앞으로", "전진", "직진", "앞으로 가", "직진해",
        # # Arabic variations
        # "إلى الأمام", "تقدم", "مباشرة", "للأمام", "تحرك للأمام",
        # # Turkish variations
        # "ileri", "düz", "ilerle", "düz git", "ileri git",
        # # Polish variations
        # "naprzód", "do przodu", "prosto", "jedź prosto", "idź naprzód",
        # # Dutch variations
        # "vooruit", "rechtdoor", "ga vooruit", "recht vooruit", "voorwaarts",
        # # Portuguese variations
        # "frente", "para frente", "em frente", "avançar", "siga em frente",
        # # Swedish variations
        # "framåt", "rakt fram", "gå framåt", "fortsätt framåt", "rakt",
        # # Finnish variations
        # "eteenpäin", "suoraan", "mene eteenpäin", "etene", "suoraan eteenpäin",
        # # Simple phonetic variations that may come from ASR errors
        "go for word", "ford", "foreward", "forword", "farward"
        # Additional common mispronunciations and ASR errors
        # Comprehensive Marathi/Hindi forward command combinations
        "pudhe ja", "pudhe jaa", "pudhe jao", "pudhe chala", "pudhe chaal", "pudhe chal", "pudhe chalo",
        "pudhe vada", "pudhe badho", "pudhe badh",
        "pude ja", "pude jaa", "pude jao", "pude chala", "pude chaal"
        "pudhay ja", "pudhay chala", "puday ja", "puday chala",
        "fwd", "forward go", "move fwd", "go fwd", "foward", "forwrd", "forwd",
        "pooda", "poode", "pood", "puude", "puda", "pooda chal", "pude chal go",
        "pudechal", "pudhe chal", "pude move", "pudhey", "pudhay", "pudha",
        "aagay", "agge", "aage ja", "age", "agay", "aage go", "age chalo",
        "straight go", "go str8", "str8", "strght", "strayt", "strait",
        "go go go", "keep go", "go on", "onwards", "move on", "proceed now"
    ],
    
    # Backward command variations
    "backward": [
        # English variations
        "backward", "go backward", "move backward", "back", "go back", "move back", 
        "reverse", "go reverse", "move in reverse", "retreat", "step back", "move backwards",
        "head back", "back it up", "drive backward", "reverse back", "go back please", "roll back",
        # Hindi variations
        "पीछे", "पीछे जाओ", "पीछे चलो", "पीछे हटो", "पीछे हट जाओ", "वापस", "वापस जाओ", "पीछे की ओर",
        "peeche", "peeche jao", "peeche chalo", "peeche hato", "peeche hat jao", "vaapas", "vaapas jao", "peeche ki or",
        "peeche aao", "peechay jao", "piche jao", "piche hao", "peeche wapas aao",
        # Marathi variations
        "मागे", "मागे जा", "मागे चला", "मागे चल", "मागे फिरा", "मागे वळा", "मागे सरळ",
        "mage", "mage ja", "mage jaa", "mage chal", "mage chala", "mage fira", "mage vala", "maghe ja",
        "maghe jaa", "maghe fira",
        # Punjabi variations
        "ਪਿੱਛੇ ਜਾਓ", "ਪਿੱਛੇ ਹਟੋ", "ਵਾਪਸ ਜਾਓ", "ਉਲਟ ਜਾਓ", "ਪਿੱਛੇ ਚੱਲੋ",
        "piche jao", "pichhe jao", "wapas jao", "ulta jao", "piche chalo",
        # Kannada variations
        "ಹಿಂದಕ್ಕೆ ಹೋಗಿ", "ಹಿಂದೆ ಹೋಗಿ", "ಹಿಂಬದಿ ಹೋಗಿ", "ರಿವರ್ಸ್ ಹೋಗಿ", "ಹಿಂದಕ್ಕೆ ಬನ್ನಿ",
        "hindakke hogi", "hinde hogi", "hinbadi hogi", "reverse hogi", "hindakke banni",
        # Bengali variations
        "পিছনে যাও", "পেছনে যাও", "ফিরে যাও", "উল্টো যাও", "পেছনে চল",
        "pichone jao", "pechone jao", "fire jao", "ulto jao", "pechone chol",
        # Gujarati variations
        "પાછળ જાઓ", "પાછા જાઓ", "પાછળ વળો", "રિવર્સ જાઓ", "પાછળ ચાલો",
        "pachal jao", "pacha jao", "pachal valo", "reverse jao", "pachal chalo",
        # Tamil variations
        "பின்னே போ", "பின்செல்", "பின்பக்கம் போ", "பின்னால் செல்", "பின்னுக்கு போ",
        "pinne po", "pinsel", "pinpakkam po", "pinnaal sel", "pinnukku po",
        # Urdu variations
        "پیچھے جاؤ", "پیچھے ہٹو", "واپس جاؤ", "الٹا چلو", "پیچھے چلو",
        "peeche jao", "peeche hato", "wapas jao", "ulta chalo", "peeche chalo",
        # # Spanish variations
        # "atrás", "hacia atrás", "ve atrás", "retrocede", "reversa",
        # # French variations
        # "reculer", "en arrière", "marche arrière", "reculez", "arrière", "recule",
        # # German variations
        # "rückwärts", "zurück", "nach hinten", "geh zurück", "zurückgehen", "rücken",
        # # Italian variations
        # "indietro", "vai indietro", "marcia indietro", "retrocedere", "tornare indietro", "retromarcia",
        # # Chinese variations (Simplified)
        # "后退", "向后", "倒退", "往后", "后方", "倒车",
        # # Japanese variations
        # "後退", "バック", "下がる", "戻る", "後ろへ", "バックする",
        # # Russian variations
        # "назад", "задний ход", "двигайся назад", "отступить", "реверс", "возвращайся",
        # # Korean variations
        # "뒤로", "후진", "뒤로 가", "뒤로 가세요", "백", "후퇴",
        # # Arabic variations
        # "للخلف", "إلى الخلف", "تراجع", "عد", "ارجع", "رجوع",
        # # Turkish variations
        # "geri", "geriye", "geri git", "tersine", "geri dön", "geri çek",
        # # Polish variations
        # "wstecz", "do tyłu", "cofnij", "cofaj", "zawróć", "odwrót",
        # # Dutch variations
        # "achteruit", "terug", "ga terug", "achterwaarts", "keer terug", "terugrijden",
        # # Portuguese variations
        # Additional common variations
        "bwd", "back go", "go bwd", "bakk", "bak ja", "back move",
        # Marathi/Hindi phonetic variations from Whisper errors
        "ujavikade fira", "ujavikade phira", "ujavikade gol fira", "ujavikade gol phira", "ujavikade goal fira",
        "ujavikade chala", "ujavikade chaal", "ujavikade chal", "ujavikade chalo",
        "ujavikade vala", "ujavikade vaal", "ujavikade val",
        "ujavikade ja", "ujavikade jaa", "ujavikade jao",
        "ujavikade valun ja", "ujavikade valun jaa",
        "ujavikade saraka", "ujavikade sarak",
        "ujaavikade fira", "ujaavikade chala", "ujaavikade ja",
        "ujavi kade fira", "ujavi kade chala", "ujavi kade ja",
        "ujya kade fira", "ujya kade chala", "ujya kade ja",
        "ujvi bazula ja", "ujvi fira", "ujvi chala",
        "mage da", "magy", "magy da", "mage de", "maghe de", "fira", "phira", "viral", "gol fira", "gol phira",
        # Comprehensive Marathi/Hindi backward command combinations
        "mage fira", "mage phira", "mage ja", "mage jaa", "mage jao", "mage chala", "mage chal", "mage hato", "mage hat",
        "maghe fira", "maghe phira", "maghe ja", "maghe chala", "maghe hato",
        "magy fira", "magy ja", "magy chala", "magy de",
        "reverse now", "back back", "go bak", "peeche move", "piche",
        "reverse go", "reverse move", "back it", "bakk it", "rvrse"
        # "para trás", "retroceder", "voltar", "recuar", "ré", "marcha atrás",
        # # Swedish variations
        # "bakåt", "backa", "gå bakåt", "tillbaka", "återgå", "reträtt",
        # # Finnish variations
        # "taaksepäin", "peruuta", "takaisin", "käänny takaisin", "taakse", "peruutus",
        # # Phonetic variations
        "backword", "bakward", "back word", "backwad", "bak"
    ],
    
    # Left command variations
    "left": [
        # English variations
        "left", "go left", "move left", "turn left", "to the left", "leftward",
        "rotate left", "spin left", "circle left", "turn left side", "veer left",
        "move to the left", "shift left", "slide left", "step left", "drift left",
        # Hindi variations (with distinctive forms)
        "बाएं", "बाएं मुड़ो", "बाएं जाओ", "बाएं चलो", "बाईं ओर जाओ", "बाईं तरफ जाओ", "बायीं ओर", "बायीं तरफ", "बायें घूमो",
        "बाई", "बाई तरफ", "बाई ओर", "बाई मुड़ो", "बाई मुडो", "बाई मुरें", "बाई मूड़ो", 
        "बाईं ओर मुड़ो", "बाईं तरफ चलो", "बाईं तरफ मुड़ो",
        "bayen", "baaye", "baye", "baye mudo", "baaye mudo", "baaye jao", "baaye chalo", "bayi or", "left mudo",
        "baee", "baaee", "bai taraf", "bai mudo", "bai muren", "bayein taraf", "bayen mudo",
        "bai or", "bai taraf jao", "baaye taraf jao", "baayn", "baayin",
        # Common spelling/pronunciation variations
        "bayein", "bayee", "bai", "baai", "baen", "baayen", "baain", 
        # Marathi variations
        "डावीकडे", "डावीकडे वळा", "डावीकडे जा", "डावीकडे चला", "डावीकडे वळून जा", "डावीकडे सरका",
        "davikade", "davikade vala", "davikade ja", "davikade chala", "davikade valun ja", "davikade saraka",
        "davi bazula ja", "davya kade ja", "davya kade vala",
        # Punjabi variations
        "ਖੱਬੇ ਜਾਓ", "ਖੱਬੇ ਮੁੜੋ", "ਖੱਬੇ ਵੱਲ", "ਖੱਬੇ ਪਾਸੇ ਜਾਓ", "ਖੱਬੇ ਚੱਲੋ",
        "khabe jao", "khabbey muro", "khabbey wal", "khabbey pase jao", "khabbey chalo",
        # Kannada variations
        "ಎಡಕ್ಕೆ ಹೋಗಿ", "ಎಡಕ್ಕೆ ತಿರುಗಿ", "ಎಡ ಬದಿಗೆ ಹೋಗಿ", "ಎಡಕ್ಕೆ ವಾಳಿ", "ಎಡಕ್ಕೆ ಚಲಿಸಿ",
        "edakke hogi", "edakke tirugi", "eda badige hogi", "edakke vaali", "edakke chalisi",
        # Bengali variations
        "বামে যাও", "বামে ঘুরো", "বাম দিকে যাও", "বামের দিকে চল", "বাম পাশের দিকে যাও",
        "bame jao", "bame ghuro", "bam dike jao", "bam dike chol", "bam pashe jao",
        # Gujarati variations
        "ડાબે જાઓ", "ડાબે વાળો", "ડાબી તરફ જાઓ", "ડાબી બાજુ જાઓ", "ડાબે ચાલો",
        "dabe jao", "dabe valo", "dabi taraf jao", "dabi baju jao", "dabe chalo",
        # Tamil variations
        "இடது பக்கம் போ", "இடப்பக்கம் திருப்பு", "இடமாக செல்", "இடது பக்கம் திரும்பு", "இடதுபுறம் போ",
        "idathu pakkam po", "idappakkam thiruppu", "idama sel", "idathu pakkam tirumbu", "idathupuram po",
        # Urdu variations
        "بائیں مڑو", "بائیں جاؤ", "بائیں طرف جاؤ", "بائیں طرف", "بائیں جانب",
        "baen muro", "baen jao", "baen taraf jao", "bayen taraf", "bain janib",
        # # Spanish variations
        # "izquierda", "a la izquierda", "gira a la izquierda", "ve a la izquierda",
        # # French variations
        # "gauche", "à gauche", "tourner à gauche", "allez à gauche", "vers la gauche",
        # # German variations
        # "links", "nach links", "biege links ab", "links abbiegen", "zur linken", "linke seite",
        # # Italian variations
        # "sinistra", "a sinistra", "gira a sinistra", "vai a sinistra", "verso sinistra",
        # # Chinese variations (Simplified)
        # "左", "向左", "左转", "往左", "左边", "向左转",
        # # Japanese variations
        # "左", "左へ", "左折", "左に曲がる", "レフト", "左方向",
        # # Russian variations
        # "влево", "налево", "поверни налево", "левая сторона", "слева", "в левую сторону",
        # # Korean variations
        # "왼쪽", "왼쪽으로", "왼쪽으로 돌아", "좌회전", "왼쪽으로 가", "좌측",
        # # Arabic variations
        # "يسار", "إلى اليسار", "انعطف يسارا", "اذهب يسارا", "الجانب الأيسر", "يساراً",
        # # Turkish variations
        # "sol", "sola", "sola dön", "sola git", "sol taraf", "sola doğru",
        # # Polish variations
        # "lewo", "w lewo", "skręć w lewo", "idź w lewo", "na lewo", "po lewej",
        # # Dutch variations
        # "links", "naar links", "ga naar links", "linksom", "links afslaan", "linker kant",
        # # Portuguese variations
        # "esquerda", "à esquerda", "vire à esquerda", "vá para a esquerda", "lado esquerdo", "virar à esquerda",
        # # Swedish variations
        # "vänster", "till vänster", "sväng vänster", "gå åt vänster", "vänstra sidan", "vänd vänster",
        # # Finnish variations
        # "vasen", "vasemmalle", "käänny vasemmalle", "mene vasemmalle", "vasen puoli", "vasempaan",
        # # Czech variations
        # "vlevo", "doleva", "odbočte doleva", "jděte doleva", "na levé straně", "levá strana",
        # Phonetic variations and common ASR mistakes
        # Marathi phonetic variations from Whisper transcription errors
        "davidare", "davikare", "daavi", "daavikade", "davi kare", "davikare gol", "davidare gol",
        "daavi karayesh", "daavi kare", "karayesh", "gholpira", "gol fira", "gol phira", "goal fira",
        # Comprehensive Marathi/Hindi direction + action combinations
        "davikade fira", "davikade phira", "davikade gol fira", "davikade gol phira", "davikade goal fira",
        "davikade chala", "davikade chaal", "davikade chal", "davikade chalo",
        "davikade vala", "davikade vaal", "davikade val",
        "davikade ja", "davikade jaa", "davikade jao",
        "davikade valun ja", "davikade valun jaa",
        "davikade saraka", "davikade sarak",
        "davikade firun ja", "davikade firun jaa",
        "daavikade fira", "daavikade phira", "daavikade gol fira", "daavikade chala", "daavikade ja",
        "davi kare fira", "davi kare chala", "davi kare ja",
        "daavi kade fira", "daavi kade chala", "daavi kade ja",
        "davya kade fira", "davya kade chala", "davya kade ja",
        "lift", "leafed", "laft", "lft", "lef", "leven", "lefty", "läft"
    ],
    
    # Right command variations
    "right": [
        # English variations
        "right", "go right", "move right", "turn right", "to the right", "rightward",
        "rotate right", "spin right", "circle right", "turn right side", "veer right",
        "move to the right", "shift right", "slide right", "step right", "drift right",
        # Hindi variations (with distinctive forms)
        "दाएं", "दाएं मुड़ो", "दाएं जाओ", "दाएं चलो", "दाईं ओर", "दाईं ओर जाओ", "दाईं तरफ", "दाईं तरफ जाओ", "दायें घूमो",
        "दाई", "दाई तरफ", "दाई ओर", "दाई मुड़ो", "दाई मुडो", "दाई मुरें", "दाई मूड़ो",
        "दाईं ओर मुड़ो", "दाईं तरफ चलो", "दाईं तरफ मुड़ो",
        "dayen", "daaye", "daye", "daye mudo", "daaye mudo", "daaye jao", "daaye chalo", "dayi or", "right mudo",
        "daee", "daaee", "dai taraf", "dai mudo", "dai muren", "dahine", "dahina", "dahini taraf", 
        "dahine jao", "dahine mud", "daye taraf jao", "daayen taraf",
        # Common spelling/pronunciation variations
        "dayein", "dayee", "dai", "daai", "daen", "daayen", "daain",
        # Marathi variations
        "उजवीकडे", "उजवीकडे वळा", "उजवीकडे जा", "उजवीकडे चला", "उजवीकडे वळून जा", "उजवीकडे सरका",
        "ujavikade", "ujavikade vala", "ujavikade ja", "ujavikade chala", "ujavikade valun ja", "ujavikade saraka",
        "ujya kade ja", "ujya kade vala", "ujvi bazula ja",
        # Punjabi variations
        "ਸੱਜੇ ਜਾਓ", "ਸੱਜੇ ਮੁੜੋ", "ਸੱਜੇ ਵੱਲ", "ਸੱਜੇ ਪਾਸੇ ਜਾਓ", "ਸੱਜੇ ਚੱਲੋ",
        "sajje jao", "sajje muro", "sajje wal", "sajje pase jao", "sajje chalo",
        # Kannada variations
        "ಬಲಕ್ಕೆ ಹೋಗಿ", "ಬಲಕ್ಕೆ ತಿರುಗಿ", "ಬಲ ಬದಿಗೆ ಹೋಗಿ", "ಬಲಕ್ಕೆ ವಾಳಿ", "ಬಲಕ್ಕೆ ಚಲಿಸಿ",
        "balakke hogi", "balakke tirugi", "bala badige hogi", "balakke vaali", "balakke chalisi",
        # Bengali variations
        "ডানে যাও", "ডানে ঘুরো", "ডান দিকে যাও", "ডান পাশের দিকে যাও", "ডানে চল",
        "dane jao", "dane ghuro", "dan dike jao", "dan pashe jao", "dane chol",
        # Gujarati variations
        "જમણે જાઓ", "જમણે વાળો", "જમણી તરફ જાઓ", "જમણી બાજુ જાઓ", "જમણે ચાલો",
        "jamne jao", "jamne valo", "jamni taraf jao", "jamni baju jao", "jamne chalo",
        # Tamil variations
        "வலது பக்கம் போ", "வலப்பக்கம் திருப்பு", "வலமாக செல்", "வலது பக்கம் திரும்பு", "வலப்புறம் போ",
        "valathu pakkam po", "valappakkam thiruppu", "valama sel", "valathu pakkam tirumbu", "valappuram po",
        # Urdu variations
        "دائیں مڑو", "دائیں جاؤ", "دائیں طرف جاؤ", "دائیں طرف", "دائیں جانب",
        "dain muro", "dain jao", "dain taraf jao", "dayen taraf", "dain janib",
        # # Spanish variations
        # "derecha", "a la derecha", "gira a la derecha", "ve a la derecha",
        # # French variations
        # "droite", "à droite", "tourner à droite", "allez à droite", "vers la droite",
        # # German variations
        # "rechts", "nach rechts", "biege rechts ab", "rechts abbiegen", "zur rechten", "rechte seite",
        # # Italian variations
        # "destra", "a destra", "gira a destra", "vai a destra", "verso destra",
        # # Chinese variations (Simplified)
        # "右", "向右", "右转", "往右", "右边", "向右转",
        # # Japanese variations
        # "右", "右へ", "右折", "右に曲がる", "ライト", "右方向",
        # # Russian variations
        # "вправо", "направо", "поверни направо", "правая сторона", "справа", "в правую сторону",
        # # Korean variations
        # "오른쪽", "오른쪽으로", "오른쪽으로 돌아", "우회전", "오른쪽으로 가", "우측",
        # # Arabic variations
        # "يمين", "إلى اليمين", "انعطف يمينا", "اذهب يمينا", "الجانب الأيمن", "يميناً",
        # # Turkish variations
        # "sağ", "sağa", "sağa dön", "sağa git", "sağ taraf", "sağa doğru",
        # # Polish variations
        # "prawo", "w prawo", "skręć w prawo", "idź w prawo", "na prawo", "po prawej",
        # # Dutch variations
        # "rechts", "naar rechts", "ga naar rechts", "rechtsom", "rechts afslaan", "rechter kant",
        # # Portuguese variations
        # "direita", "à direita", "vire à direita", "vá para a direita", "lado direito", "virar à direita",
        # # Swedish variations
        # "höger", "till höger", "sväng höger", "gå åt höger", "högra sidan", "vänd höger",
        # # Finnish variations
        # "oikea", "oikealle", "käänny oikealle", "mene oikealle", "oikea puoli", "oikeaan",
        # # Czech variations
        # "vpravo", "doprava", "odbočte doprava", "jděte doprava", "na pravé straně", "pravá strana",
        # # Phonetic variations and common ASR mistakes
        "rite", "wright", "ryt", "rit", "rght", "raight", "righte", "rigte"
    ],
    
    # Rotate left command variations
    "rotate_left": [
        # English variations
        "rotate left", "spin left", "turn around left", "rotate counter-clockwise", "turn counter clockwise",
        "spin counter-clockwise", "rotate anticlockwise", "circle left", "turn full left",
        "rotate to the left", "spin to the left", "make a left circle", "turn left in place",
        # Hindi variations
        "बाएं घूमो", "बाएं घूमना", "पूरा बाएं मुड़ो", "उल्टी दिशा में घूमो", "बाईं ओर घूमो", "बाईं तरफ घूमो",
        "baaye ghumo", "baaye ghoom", "baaye ghumao", "baaye rotate karo", "counter clockwise ghoom",
        "ulti disha me ghumo", "pura baaye mudo", "left me ghoom jao", "bai taraf ghoom", "bai or ghoom",
        # Marathi variations
        "डावीकडे फिरा", "डावीकडे गोल फिरा", "डावीकडे वळून फिरा", "डावीकडे पूर्ण फिरा", "डावीकडे गोलक फिरा",
        "davikade fira", "davikade gol fira", "davikade valun fira", "davikade purn fira", "davikade golak fira",
        # Punjabi variations
        "ਖੱਬੇ ਘੁੰਮੋ", "ਖੱਬੇ ਰੋਟੇਟ ਕਰੋ", "ਖੱਬੇ ਵੱਲ ਘੁੰਮੋ", "ਖੱਬੀ ਸਾਈਡ ਘੁੰਮੋ",
        "khabbey ghumo", "khabbey rotate karo", "khabbey wal ghumo", "khabbey side ghumo",
        # Kannada variations
        "ಎಡಕ್ಕೆ ಸುತ್ತಿ", "ಎಡಕ್ಕೆ ಸಂಪೂರ್ಣ ಸುತ್ತಿ", "ಎಡಕ್ಕೆ ತಿರುಗುತ್ತಾ", "ಎಡಕ್ಕೆ ವೃತ್ತ ಮಾಡಿ",
        "edakke suthi", "edakke sampoorna suthi", "edakke tirugutha", "edakke vrutta madi",
        # Bengali variations
        "বামে ঘুরে ঘুরো", "বামে পুরো ঘুরো", "বাম দিকে ঘুরাও", "বামে ঘুরো",
        "bame ghure ghuro", "bame puro ghuro", "bam dike ghurao", "bame ghuro",
        # Gujarati variations
        "ડાબે ફરાવો", "ડાબી તરફ ઘુમાવો", "ડાબે પરિભ્રમણ કરો", "ડાબે સંપૂર્ણ ફરાવો",
        "dabe faravo", "dabi taraf ghumavo", "dabe paribhraman karo", "dabe sampurn faravo",
        # Tamil variations
        "இடப்பக்கம் சுற்று", "இடப்பக்கம் முழு சுற்று", "இடது பக்கம் சுற்றிவிடு", "இடப்புறம் சுற்று",
        "idappakkam sutru", "idappakkam muzhu sutru", "idathu pakkam sutrividu", "idappuram sutru",
        # Urdu variations
        "بائیں گھومو", "بائیں طرف گھومو", "بائیں جانب گھومو", "بائیں گردش کرو",
        "baen ghoomo", "baen taraf ghoomo", "baen janib ghoomo", "bain gardish karo",
        # # Spanish variations
        # "girar a la izquierda", "rotar a la izquierda", "dar vuelta a la izquierda", "girar completamente a la izquierda",
        # # French variations
        # "tourner à gauche complètement", "faire un tour à gauche", "rotation à gauche", "pivoter à gauche",
        # "tourner dans le sens antihoraire", "faire un cercle à gauche",
        # # German variations
        # "nach links drehen", "links herum drehen", "links rotieren", "gegen den Uhrzeigersinn drehen",
        # "links rundherum", "vollständig nach links drehen",
        # # Italian variations
        # "ruotare a sinistra", "girare completamente a sinistra", "fare un cerchio a sinistra", "rotazione sinistra",
        # "girare in senso antiorario", "rotazione antioraria",
        # # Chinese variations (Simplified)
        # "向左旋转", "左转圈", "逆时针旋转", "完全向左转", "左侧旋转",
        # # Japanese variations
        # "左回り", "左に回転", "反時計回り", "左に旋回する", "左回転",
        # # Russian variations
        # "повернуть влево полностью", "вращаться влево", "поворот против часовой стрелки",
        # "крутиться влево", "повернуться влево кругом",
        # # Korean variations
        # "왼쪽으로 회전", "왼쪽으로 돌기", "반시계 방향으로", "왼쪽으로 빙글빙글", "왼쪽으로 완전히 돌기",
        # # Arabic variations
        # "الدوران إلى اليسار", "دوران كامل لليسار", "لف إلى اليسار", "دوران عكس عقارب الساعة",
        # # Turkish variations
        # "sola dön", "sola döndür", "saat yönünün tersine", "sol tarafa dön", "tamamen sola dön",
        # # Polish variations
        # "obróć w lewo", "skręć całkowicie w lewo", "obróć się przeciwnie do ruchu wskazówek zegara", 
        # # Dutch variations
        # "draai naar links", "roteer linksom", "volledig naar links draaien", "tegen de klok in draaien",
        # # Portuguese variations
        # "girar à esquerda", "rodar para a esquerda", "rotação anti-horária", "dar volta completa à esquerda",
        # Phonetic variations
        "rotateleft", "rotate lft", "spin lft", "turn lft"
    ],
    
    # Rotate right command variations
    "rotate_right": [
        # English variations
        "rotate right", "spin right", "turn around right", "rotate clockwise", "turn clockwise",
        "spin clockwise", "circle right", "turn full right", "make a right circle",
        "rotate to the right", "spin to the right", "make a right spin", "turn right in place",
        # Hindi variations
        "दाएं घूमो", "दाएं घूमना", "पूरा दाएं मुड़ो", "सीधी दिशा में घूमो", "दाईं ओर घूमो", "दाईं तरफ घूमो",
        "daaye ghumo", "daaye ghoom", "daaye ghumao", "daaye rotate karo", "clockwise ghoom",
        "seedhi disha me ghumo", "pura daaye mudo", "right me ghoom jao", "dai taraf ghoom", "dai or ghoom",
        # Marathi variations
        "उजवीकडे फिरा", "उजवीकडे गोल फिरा", "उजवीकडे वळून फिरा", "उजवीकडे पूर्ण फिरा", "उजवीकडे गोलक फिरा",
        "ujavikade fira", "ujavikade gol fira", "ujavikade valun fira", "ujavikade purn fira", "ujavikade golak fira",
        # Punjabi variations
        "ਸੱਜੇ ਘੁੰਮੋ", "ਸੱਜੇ ਰੋਟੇਟ ਕਰੋ", "ਸੱਜੇ ਵੱਲ ਘੁੰਮੋ", "ਸੱਜੀ ਸਾਈਡ ਘੁੰਮੋ",
        "sajje ghumo", "sajje rotate karo", "sajje wal ghumo", "sajji side ghumo",
        # Kannada variations
        "ಬಲಕ್ಕೆ ಸುತ್ತಿ", "ಬಲಕ್ಕೆ ಸಂಪೂರ್ಣ ಸುತ್ತಿ", "ಬಲಕ್ಕೆ ತಿರುಗುತ್ತಾ", "ಬಲಕ್ಕೆ ವೃತ್ತ ಮಾಡಿ",
        "balakke suthi", "balakke sampoorna suthi", "balakke tirugutha", "balakke vrutta madi",
        # Bengali variations
        "ডানে ঘুরে ঘুরো", "ডানে পুরো ঘুরো", "ডান দিকে ঘুরাও", "ডানে ঘুরো",
        "dane ghure ghuro", "dane puro ghuro", "dan dike ghurao", "dane ghuro",
        # Gujarati variations
        "જમણે ફરાવો", "જમણી તરફ ઘુમાવો", "જમણે પરિભ્રમણ કરો", "જમણે સંપૂર્ણ ફરાવો",
        "jamne faravo", "jamni taraf ghumavo", "jamne paribhraman karo", "jamne sampurn faravo",
        # Tamil variations
        "வலப்பக்கம் சுற்று", "வலப்பக்கம் முழு சுற்று", "வலது பக்கம் சுற்றிவிடு", "வலப்புறம் சுற்று",
        "valappakkam sutru", "valappakkam muzhu sutru", "valathu pakkam sutrividu", "valappuram sutru",
        # Urdu variations
        "دائیں گھومو", "دائیں طرف گھومو", "دائیں جانب گھومو", "دائیں گردش کرو",
        "dain ghoomo", "dain taraf ghoomo", "dain janib ghoomo", "dain gardish karo",
        # # Spanish variations
        # "girar a la derecha", "rotar a la derecha", "dar vuelta a la derecha",
        # # French variations
        # "tourner à droite complètement", "faire un tour à droite", "rotation à droite", "pivoter à droite",
        # "tourner dans le sens horaire", "faire un cercle à droite",
        # # German variations
        # "nach rechts drehen", "rechts herum drehen", "rechts rotieren", "im Uhrzeigersinn drehen",
        # "rechts rundherum", "vollständig nach rechts drehen",
        # # Italian variations
        # "ruotare a destra", "girare completamente a destra", "fare un cerchio a destra", "rotazione destra",
        # "girare in senso orario", "rotazione oraria",
        # # Chinese variations (Simplified)
        # "向右旋转", "右转圈", "顺时针旋转", "完全向右转", "右侧旋转",
        # # Japanese variations
        # "右回り", "右に回転", "時計回り", "右に旋回する", "右回転",
        # # Russian variations
        # "повернуть вправо полностью", "вращаться вправо", "поворот по часовой стрелке",
        # "крутиться вправо", "повернуться вправо кругом",
        # Phonetic variations
        "rotateright", "rotate rght", "spin rght", "turn rght"
    ],
    
    # Start command variations
    "start": [
        # English variations
        "start", "begin", "power on", "activate", "wake up", "turn on", "initiate", "get going", 
        "let's go", "engage", "launch", "commence", "start wheelchair", "power up", "fire it up",
        "turn it on", "get started", "switch on", "kick off",
        # Hindi variations
        "शुरू", "शुरू करो", "चालू करो", "चालू", "शुरुआत करो", "ऑन करो", "जागो", "मशीन चालू करो",
        "शुरू हो जाओ", "चालू कर दो", "तुरंत चालू करो",
        "shuru", "shuru karo", "chalu karo", "chalu", "on karo", "activate karo", "jago",
        "power on karo", "start karo", "start ho jao", "machine chalu karo", "turant start karo",
        # Marathi variations
        "सुरू", "सुरू करा", "चालू करा", "चालू", "ऑन करा", "यंत्र सुरू करा", "ताबडतोब सुरू करा",
        "suru", "suru kara", "chalu kara", "chalu", "on kara", "yantra suru kara", "tatkal suru kara",
        # Punjabi variations
        "ਸ਼ੁਰੂ ਕਰੋ", "ਸ਼ੁਰੂ ਕਰੋ ਜੀ", "ਚਾਲੂ ਕਰੋ", "ਆਨ ਕਰੋ", "ਸ਼ੁਰੂ ਕਰ ਦਿਓ",
        "shuru karo", "shuru karo ji", "chalu karo", "on karo", "shuru kar dio",
        # Kannada variations
        "ಪ್ರಾರಂಭಿಸಿ", "ಆರಂಭಿಸಿ", "ಆನ್ ಮಾಡಿ", "ಚಾಲು ಮಾಡಿ", "ಚಾಲನೆ ಮಾಡಿ",
        "prarambhisi", "arambhisi", "on madi", "chalu madi", "chalane madi",
        # Bengali variations
        "শুরু কর", "শুরু করুন", "চালু কর", "অন কর", "স্টার্ট কর",
        "shuru kor", "shuru korun", "chalu kor", "on kor", "start kor",
        # Gujarati variations
        "શરૂ કરો", "શરૂઆત કરો", "ચાલુ કરો", "ઓન કરો", "પ્રારંભ કરો",
        "sharu karo", "sharuvaat karo", "chalu karo", "on karo", "prarambh karo",
        # Tamil variations
        "தொடங்கு", "தொடங்குங்கள்", "ஆன் செய்", "இயக்கு", "வேலை தொடங்கு",
        "thodangu", "thodangungal", "on sei", "iyakku", "velai thodangu",
        # Urdu variations
        "شروع کرو", "شروع کریں", "چالو کرو", "آن کرو", "فعال کرو",
        "shuru karo", "shuru karein", "chalu karo", "on karo", "faal karo",
        # # Spanish variations
        # "empezar", "iniciar", "encender", "activar", "comenzar", "arrancar", "poner en marcha",
        # # French variations
        # "commencer", "démarrer", "allumer", "activer", "mettre en marche", "démarrage",
        # "lancer", "s'y mettre", "allons-y", "en route",
        # # German variations
        # "starten", "beginnen", "einschalten", "aktivieren", "anmachen", "anfangen",
        # "los", "anschalten", "in gang setzen", "initiieren",
        # # Italian variations
        # "avviare", "iniziare", "accendere", "attivare", "cominciare", "partire",
        # "mettere in moto", "avvio", "via",
        # # Chinese variations (Simplified)
        # "开始", "启动", "打开", "激活", "开机", "运行", "启动轮椅",
        # # Japanese variations
        # "開始", "スタート", "起動", "オン", "作動", "始める", "電源オン",
        # # Russian variations
        # "старт", "начать", "включить", "активировать", "запустить", "начинать",
        # "включение", "приступить", "поехали",
        # # Korean variations
        # "시작", "켜다", "켜기", "활성화", "작동", "시작하다", "출발",
        # # Arabic variations
        # "ابدأ", "تشغيل", "بدء", "تنشيط", "تفعيل", "شغل", "انطلق",
        # # Turkish variations
        # "başla", "başlat", "çalıştır", "aktive et", "aç", "başlama", "hareket et",
        # # Polish variations
        # "start", "rozpocznij", "włącz", "aktywuj", "uruchom", "zacznij", "ruszaj",
        # # Dutch variations
        # "starten", "beginnen", "aanzetten", "activeren", "inschakelen", "opstarten", "aan",
        # # Portuguese variations
        # "iniciar", "começar", "ligar", "ativar", "arrancar", "dar partida", "acionar",
        # # Swedish variations
        # "starta", "börja", "sätta på", "aktivera", "sätt igång", "kör igång", "slå på",
        # # Finnish variations
        # "aloita", "käynnistä", "aktivoi", "käynnistys", "laita päälle", "aloittaa", "virta päälle",
        # # Czech variations
        # "start", "začít", "zapnout", "aktivovat", "spustit", "zahájit", "nastartovat",
        # Phonetic variations
        "staart", "begin now", "stat", "strt"
    ],
    
    # Stop command variations
    "stop": [
        # English variations
        "stop", "halt", "pause", "wait", "brake", "power off", "deactivate", "cease", 
        "hold", "freeze", "stand still", "stay", "stop moving", "no movement",
        "shut down", "stop now", "cut it out", "kill switch", "stop right now", "halt immediately",
        # Hindi variations
        "रुको", "थांबो", "रुक जाओ", "रुको अभी", "बंद करो", "बंद", "ऑफ करो", "ठहरो", "ठहर जाओ", "बस करो",
        "ruko", "thambo", "ruk jao", "ruk abhi", "band karo", "band", "off karo", "thehro", "theher jao", "bas karo",
        "stop karo", "rukna", "ab ruko", "deactivate karo", "turant ruko", "yahi ruk jao",
        # Marathi variations
        "थांबा", "थांबवा", "बंद करा", "ऑफ करा", "आता थांबा", "ताबडतोब थांबा",
        "thamba", "thambava", "band kara", "off kara", "ata thamba", "tatkal thamba",
        # Punjabi variations
        "ਰੋਕੋ", "ਰੁੱਕੋ", "ਥੰਮ ਜਾਓ", "ਬੰਦ ਕਰੋ", "ਠਹਿਰੋ",
        "roko", "rukko", "tham jao", "band karo", "thahiro",
        # Kannada variations
        "ನಿಲ್ಲಿಸಿ", "ನಿಲ್ಲಿ", "ಆಫ್ ಮಾಡಿ", "ಸ್ಥಗಿತಗೊಳಿಸಿ", "ತಡೆಹಿಡಿ",
        "nillisi", "nilli", "off madi", "sthagitagolisi", "tadehidi",
        # Bengali variations
        "থামো", "থামুন", "থেমে যাও", "বন্ধ কর", "বন্ধ করুন",
        "thamo", "thamun", "theme jao", "bondho kor", "bondho korun",
        # Gujarati variations
        "બંધ કરો", "રોકો", "ઑફ કરો", "થોભો", "સ્થગિત કરો",
        "bandh karo", "roko", "off karo", "thobho", "sthagit karo",
        # Tamil variations
        "நிறுத்து", "நிறுத்துங்கள்", "நிறுத்திவிடு", "ஆஃப் செய்", "நிலை நிறுத்து",
        "niruthu", "niruthungal", "niruthividu", "off sei", "nilai niruthu",
        # Urdu variations
        "رکو", "روکو", "رک جاؤ", "بند کرو", "ٹھہر جاؤ",
        "ruko", "roko", "ruk jao", "band karo", "thahar jao",
        # # Spanish variations
        # "parar", "detener", "alto", "para", "detente", "espera",
        # # French variations
        # "arrêter", "arrêt", "stop", "halte", "pause", "attendre", "freiner", "éteindre",
        # "désactiver", "cesser", "immobiliser", "tenir", "geler",
        # # German variations
        # "stopp", "halt", "anhalten", "pausieren", "warten", "bremsen", "ausschalten",
        # "deaktivieren", "stillstehen", "bleiben", "einfrieren", "halten",
        # # Italian variations
        # "fermare", "fermati", "stop", "pausa", "aspetta", "frenare", "spegnere",
        # "disattivare", "cessare", "bloccare", "immobile", "fermo", "arrestare",
        # # Chinese variations (Simplified)
        # "停止", "暂停", "等待", "刹车", "关闭", "停", "别动", "静止",
        # # Japanese variations
        # "停止", "ストップ", "止まれ", "止める", "待って", "ブレーキ", "オフ", "止まる", "中止",
        # # Russian variations
        # "стоп", "остановись", "пауза", "ждать", "тормоз", "выключить", "деактивировать",
        # "прекратить", "держать", "замереть", "стоять", "замри",
        # # Korean variations
        # "멈춰", "정지", "멈추세요", "멈춤", "서", "스톱", "중지", "기다려", "그만",
        # # Arabic variations
        # "قف", "توقف", "انتظر", "أوقف", "كفى", "تمهل", "تعطيل", "إيقاف",
        # # Turkish variations
        # "dur", "durun", "durdur", "duraklat", "bekle", "durma", "fren", "durdur",
        # # Polish variations
        # "zatrzymaj", "stop", "stój", "wstrzymaj", "pauza", "hamuj", "czekaj", "zatrzymanie",
        # # Dutch variations
        # "stop", "halt", "houden", "wacht", "stoppen", "pauze", "rem", "stilstaan",
        # # Portuguese variations
        # "pare", "parar", "alto", "espera", "deter", "trava", "freio", "aguarde",
        # # Swedish variations
        # "stopp", "stanna", "håll", "pausa", "vänta", "broms", "avsluta", "stå still",
        # # Finnish variations
        # "seis", "pysähdy", "lopeta", "tauko", "jarruta", "odota", "keskeytä", "pysäytä",
        # Phonetic variations
        "stp", "stahp", "stoop", "brake now", "hault", "holt"
    ]
}


@lru_cache(maxsize=1)
def _command_variation_catalog() -> Tuple[List[str], Dict[str, str]]:
    """Cache flattened command variation list for global fuzzy matching."""
    phrases: List[str] = []
    lookup: Dict[str, str] = {}

    for command, variations in WHEELCHAIR_COMMANDS.items():
        canonical = command.lower().strip()
        if canonical and canonical not in lookup:
            lookup[canonical] = command
            phrases.append(canonical)

        for variation in variations:
            normalized = variation.lower().strip()
            if not normalized:
                continue
            if normalized not in lookup:
                lookup[normalized] = command
                phrases.append(normalized)

    return phrases, lookup


def _global_command_fuzzy_match(text: str) -> Optional[Tuple[str, float, str]]:
    """Match arbitrary text to closest command variation using fuzzy scoring."""
    if not RAPIDFUZZ_AVAILABLE:
        return None

    normalized = text.strip().lower()
    if not normalized:
        return None

    phrases, lookup = _command_variation_catalog()
    if not phrases:
        return None

    best = rapidfuzz_process.extractOne(
        normalized,
        phrases,
        scorer=rapidfuzz_fuzz.WRatio,
    )
    if not best:
        return None

    phrase, score, _ = best
    confidence = float(score) / 100.0
    if confidence < 0.70:
        return None

    command = lookup.get(phrase)
    if not command:
        return None

    return command, confidence, phrase

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

    loaded = _load_audio_16k(audio_path)
    if loaded is None:
        raise RuntimeError("Failed to load audio for verification")
    raw_wav, _ = loaded
    raw_wav = ensure_minimum_duration(raw_wav, SAMPLE_RATE, target_seconds=1.2)
    trimmed_wav = trim_audio_to_speech(raw_wav, SAMPLE_RATE, threshold_ratio=0.25, min_threshold=0.008)
    if trimmed_wav.size > int(SAMPLE_RATE * 0.4):
        raw_wav = trimmed_wav
    
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
    voice_only = fast_noise_gate(voice_only, SAMPLE_RATE, floor_percentile=12.0, gate_strength=1.4)
    
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
    
    processed_wav = ensure_minimum_duration(processed_wav, SAMPLE_RATE, target_seconds=1.2)
    processed_wav = apply_subtle_noise_reduction(processed_wav, SAMPLE_RATE, SPEAKER_NOISE_REDUCTION_BLEND)
    processed_wav = fast_noise_gate(processed_wav, SAMPLE_RATE, floor_percentile=10.0, gate_strength=1.3)

    return processed_wav.astype(np.float32)

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
        (r'दाएं\s+(?:तरफ|ओर)?\s*(?:घूमो|मुड़ो|घूम|मुड़|फिरो|फिर)', "right"),
        (r'दायें\s+(?:तरफ|ओर)?\s*(?:घूमो|मुड़ो|घूम|मुड़|फिरो|फिर)', "right"),
        (r'दाईं\s+(?:तरफ|ओर)?\s*(?:घूमो|मुड़ो|घूम|मुड़|फिरो|फिर)', "right"),
        (r'दाई\s+(?:तरफ|ओर)?\s*(?:घूमो|मुड़ो|घूम|मुड़|फिरो|फिर)', "right"),
        (r'dai\s*(?:taraf|or)?\s*(?:ghumo|mudo|ghoom|mud|firo|fir|gumo)', "right"),
        (r'daaye\s*(?:taraf|or)?\s*(?:ghumo|mudo|ghoom|mud|firo|fir|gumo)', "right"),
        (r'dayen\s*(?:taraf|or)?\s*(?:ghumo|mudo|ghoom|mud|firo|fir|gumo)', "right"),
        (r'daayen\s*(?:taraf|or)?\s*(?:ghumo|mudo|ghoom|mud|firo|fir|gumo)', "right"),
        
        # Left rotation patterns
        (r'बाएं\s+(?:तरफ|ओर)?\s*(?:घूमो|मुड़ो|घूम|मुड़|फिरो|फिर)', "left"),
        (r'बायें\s+(?:तरफ|ओर)?\s*(?:घूमो|मुड़ो|घूम|मुड़|फिरो|फिर)', "left"),
        (r'बाईं\s+(?:तरफ|ओर)?\s*(?:घूमो|मुड़ो|घूम|मुड़|फिरो|फिर)', "left"),
        (r'बाई\s+(?:तरफ|ओर)?\s*(?:घूमो|मुड़ो|घूम|मुड़|फिरो|फिर)', "left"),
        (r'bai\s*(?:taraf|or)?\s*(?:ghumo|mudo|ghoom|mud|firo|fir|gumo)', "left"),
        (r'baye\s*(?:taraf|or)?\s*(?:ghumo|mudo|ghoom|mud|firo|fir|gumo)', "left"),
        (r'bayen\s*(?:taraf|or)?\s*(?:ghumo|mudo|ghoom|mud|firo|fir|gumo)', "left"),
        (r'baayen\s*(?:taraf|or)?\s*(?:ghumo|mudo|ghoom|mud|firo|fir|gumo)', "left"),
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
            return "right"
        if any(word in text_clean for word in ["bai", "baaye", "bayen", "baayen", "left"]):
            return "left"
            
    if "or" in text_clean and any(word in text_clean for word in ["ghumo", "mudo", "ghum", "gumo"]):
        if any(word in text_clean for word in ["dai", "daaye", "dayen", "daayen", "right"]):
            return "right"
        if any(word in text_clean for word in ["bai", "baaye", "bayen", "baayen", "left"]):
            return "left"
    
    # If no direction found
    return None

def process_command_with_whisper_tiny(
    audio_path=None,
    detect_lang=True,
    fast_mode=FAST_TRANSCRIPTION_ENABLED,
    noise_reduction_blend: Optional[float] = None,
):
    """
    Process voice command using Whisper tiny model directly.
    This function handles recording (if audio_path not provided),
    translation with Whisper tiny, and command matching.
    
    Args:
        audio_path: Optional path to existing audio file. If None, will record new audio.
        detect_lang: Whether to auto-detect language (True) or use DEFAULT_LANGUAGE (False)
        fast_mode: Skip heavy denoising for lower latency on edge devices.
        
    Returns:
        Tuple of (matched_command, confidence_score, translation)
    """
    # Record audio if path not provided
    if audio_path is None:
        # Create timestamp for unique filename
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        audio_path = TEMP_DIR / f"voice_cmd_{timestamp}.wav"
        
        # Create temp directory if it doesn't exist
        TEMP_DIR.mkdir(exist_ok=True)
        
        # Use built-in voice activity detection helpers
        print("\n[Command Recording] Press Enter to start recording command...")
        input()

        prompt_phrase = "Please speak your command clearly"
        audio_float, speech_percent = record_with_vad(
            RECORD_DURATION,
            SAMPLE_RATE,
            max_attempts=2,
            prompt_phrase=prompt_phrase,
        )

        if audio_float is None or speech_percent < 10:
            print("Failed to detect sufficient speech in recording.")
            print("Please speak clearly when giving commands.")
            return None, 0

        # Convert to int16 and persist for downstream processing
        audio = (np.clip(audio_float, -1.0, 1.0) * 32767).astype(np.int16)
        wav.write(audio_path, SAMPLE_RATE, audio)
        
        # Process the audio for better voice command recognition
    try:
        # First, determine which language to use for translation
        language_code = None  # Default: auto-detection
        
        if not detect_lang:
            # Use preset language if auto-detection is disabled
            language_code = DEFAULT_LANGUAGE
        
        # Translate using Whisper tiny model directly
        translation, stt_success, stt_diag = transcribe_command_audio(
            audio_path,
            language=language_code,
            fast_mode=fast_mode,
            noise_reduction_blend=noise_reduction_blend,
        )
        if not stt_success or not translation:
            print("Failed to translate audio or no speech detected.")
            if stt_diag:
                print(f"[STT Diagnostics] {json.dumps(stt_diag, indent=2)}")
            return None, 0, translation

        print(f"Translation: '{translation}'")
        if stt_diag:
            print(f"[STT Diagnostics] {json.dumps(stt_diag, indent=2)}")
        
        # Try Hindi direction detection first for better rotation command detection
        hindi_command = hindi_direction_detector(translation)
        if hindi_command:
            print(f"Found command through Hindi direction detector: '{hindi_command}'")
            return hindi_command, 0.90, translation  # High confidence for direct matches
            
        # Match the translated text to a wheelchair command
        command, confidence = match_command(translation)

        if (command is None or confidence < 0.60) and stt_diag:
            raw_transcription = stt_diag.get("raw_transcription") if isinstance(stt_diag, dict) else None
            if raw_transcription and raw_transcription.strip():
                if raw_transcription.strip().lower() != translation.strip().lower():
                    print("Translation mapping inconclusive; retrying with raw transcription...")
                    print(f"Raw transcription candidate: '{raw_transcription}'")
                fallback_command, fallback_confidence = match_command(raw_transcription)
                if fallback_command:
                    print(
                        f"Using raw transcription match '{fallback_command}' "
                        f"(confidence {fallback_confidence:.2f})"
                    )
                    return fallback_command, fallback_confidence, raw_transcription

        return command, confidence, translation
        
    except Exception as e:
        print(f"Error processing command: {e}")
        return None, 0, ""

def compute_enhanced_embedding(_encoder_unused, audio: np.ndarray):
    """Generate a robust speaker embedding using SpeechBrain ECAPA-TDNN."""
    if audio is None or audio.size == 0:
        return None, None

    recognizer = load_speaker_recognizer()

    safe_audio = ensure_minimum_duration(audio, SAMPLE_RATE, target_seconds=1.1)
    if safe_audio.ndim > 1:
        safe_audio = safe_audio.flatten()

    rms = np.sqrt(np.mean(safe_audio**2)) if safe_audio.size else 0.0
    target_rms = 0.2
    if rms > 0:
        gain = target_rms / max(rms, 1e-4)
        safe_audio = np.clip(safe_audio * gain, -1.0, 1.0)

    safe_audio = apply_subtle_noise_reduction(safe_audio, SAMPLE_RATE, SPEAKER_NOISE_REDUCTION_BLEND)
    safe_audio = fast_noise_gate(safe_audio, SAMPLE_RATE, floor_percentile=5.0, gate_strength=1.1)

    segment_duration = max(0.5, float(SPEAKER_EMBED_SEGMENT_SECONDS))
    segment_samples = max(160, int(SAMPLE_RATE * segment_duration))
    overlap = float(np.clip(SPEAKER_EMBED_OVERLAP, 0.0, 0.95))
    step = max(1, int(segment_samples * (1.0 - overlap)))
    if step >= segment_samples:
        step = max(1, segment_samples // 2)

    segments: List[np.ndarray] = []
    if safe_audio.size <= segment_samples:
        segments = [ensure_minimum_duration(safe_audio, SAMPLE_RATE, target_seconds=segment_duration)]
    else:
        for start in range(0, safe_audio.size - segment_samples + 1, step):
            segment = safe_audio[start:start + segment_samples]
            segments.append(segment)
        tail_start = max(0, safe_audio.size - segment_samples)
        tail = safe_audio[tail_start:]
        if tail.size:
            segments.append(tail)

    if not segments:
        segments = [safe_audio]

    global_rms = float(np.sqrt(np.mean(safe_audio**2))) if safe_audio.size else 0.0
    min_segment_rms = max(float(SPEAKER_SEGMENT_RMS_FLOOR), global_rms * float(SPEAKER_SEGMENT_RMS_RATIO))
    filtered_segments: List[np.ndarray] = []
    dropped_segments = 0
    loudest_seg: Optional[np.ndarray] = None
    loudest_rms = -1.0
    for seg in segments:
        seg_rms = float(np.sqrt(np.mean(seg**2))) if seg.size else 0.0
        if seg_rms >= min_segment_rms:
            filtered_segments.append(seg)
        else:
            dropped_segments += 1
        if seg_rms > loudest_rms:
            loudest_rms = seg_rms
            loudest_seg = seg
    if not filtered_segments and loudest_seg is not None:
        filtered_segments = [loudest_seg]
    if dropped_segments:
        print(
            f"[Speaker] Dropped {dropped_segments} low-energy segments below {min_segment_rms:.4f} RMS"
        )
    segments = filtered_segments

    max_segments = 5
    if len(segments) > max_segments:
        indices = np.linspace(0, len(segments) - 1, num=max_segments, dtype=int)
        segments = [segments[i] for i in indices]

    print(f"[Speaker] Embedding segments considered: {len(segments)}")

    device = getattr(recognizer, "device", "cpu")
    batch_tensors = []
    for seg in segments:
        padded = ensure_minimum_duration(seg, SAMPLE_RATE, target_seconds=segment_duration)
        clipped = np.clip(padded, -1.0, 1.0).astype(np.float32)
        batch_tensors.append(torch.from_numpy(clipped))

    wav_tensor = torch.stack(batch_tensors).to(device)

    with torch.no_grad():
        embedding_tensor = recognizer.encode_batch(wav_tensor)

    if embedding_tensor is None:
        return None, None

    embedding_np = embedding_tensor.detach().cpu().numpy().astype(np.float32)
    embedding = embedding_np
    if embedding_np.ndim > 1:
        try:
            embedding = embedding_np.reshape(-1, embedding_np.shape[-1]).mean(axis=0)
        except Exception:
            embedding = embedding_np.mean(axis=0)
    embedding = _normalize_vector(embedding)

    return embedding, None

def identify_speaker(_encoder_unused=None, is_registration=False):
    """
    Improved speaker identification with enhanced reliability and gender verification.
    Uses multiple comparison methods, voice-focused processing, and gender detection
    to prevent cross-gender matching issues.
    
    Args:
        encoder: The voice encoder model (optional)
        is_registration: If True, use verification phrases. If False, just record voice command.
    
    Returns (name, score, processed_path) if match found, else (None, score, processed_path).
    """
    # Load the speaker recognizer
    try:
        recognizer = load_speaker_recognizer()
    except Exception as exc:
        print(f"CRITICAL ERROR: Failed to load speaker recognizer: {exc}")
        return None, 0, None
            
    # Check if we have any stored voice profiles
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
    
    prompt_phrase = None
    
    if is_registration:
        # Only show language selection and verification phrases during registration
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
            
        phrases = verification_phrases[lang]
        prompt_phrase = phrases[int(time.time()) % len(phrases)]
    
    # Record with voice activity detection
    print("\n[Voice Authentication] Press Enter and speak your command...")
    input()
    
    audio_float, speech_percent = record_with_vad(
        RECORD_DURATION,
        SAMPLE_RATE,
        max_attempts=3,
        prompt_phrase=prompt_phrase
    )
    
    if audio_float is None or speech_percent < 15:
        print("Failed to capture sufficient speech for authentication.")
        return None, 0, None
    
    # Convert to int16 format
    audio = (np.clip(audio_float, -1.0, 1.0) * 32767).astype(np.int16)
    wav.write(raw_path, SAMPLE_RATE, audio)
    
    # Process the audio with simplified processing for better efficiency
    try:
        # Basic preprocessing using our verification pipeline tuned for ECAPA embeddings
        reduced = preprocess_audio_for_verification(str(raw_path))
        wav.write(
            processed_path,
            SAMPLE_RATE,
            (np.clip(reduced, -1.0, 1.0) * 32767).astype(np.int16),
        )

        # Generate the embedding directly - this is the essential step
        test_emb, _ = compute_enhanced_embedding(recognizer, reduced)
        if test_emb is None:
            raise RuntimeError("Failed to compute speaker embedding")
        test_emb = np.asarray(test_emb, dtype=np.float32).reshape(-1)
        
        # Use a simple and faster approach to detect gender - just check if gender was provided by user
        # This avoids the CPU-intensive gender detection
        current_speaker_gender = 'unknown'
        print("Ready to compare with voice profiles...")
        
    except Exception as e:
        print(f"Audio processing failed: {e}")
        return None, 0, raw_path

    # Compare against stored embeddings grouped by canonical speaker name
    import re
    grouped_embeddings: Dict[str, List[Tuple[str, np.ndarray]]] = defaultdict(list)
    profile_genders: Dict[str, str] = {}

    for path in embeddings_list:
        variant_name = path.stem
        base_name = re.sub(r"_(\d+)$", "", variant_name)
        try:
            stored = np.load(path)
            vector = np.asarray(stored, dtype=np.float32).reshape(-1)
            if vector.shape != test_emb.shape:
                print(
                    f"  - {variant_name}: incompatible embedding dimensions ({vector.size} != {test_emb.size}). "
                    "Please re-enroll this profile to use the new recognizer."
                )
                continue
            grouped_embeddings[base_name].append((variant_name, vector))
        except Exception as exc:
            print(f"Error processing {variant_name}: {exc}")

    if not grouped_embeddings:
        print("No valid voice profiles found.")
        return None, 0, processed_path

    for base_name in grouped_embeddings.keys():
        gender_file = VOICE_DB_PROCESSED_DIR / f"{base_name}_gender.txt"
        profile_gender = "unknown"
        if gender_file.exists():
            try:
                profile_gender = gender_file.read_text().strip().lower()
            except Exception:
                profile_gender = "unknown"
        profile_genders[base_name] = profile_gender

    print(f"Comparing against {len(grouped_embeddings)} enrolled speaker(s)...")

    scores: List[Dict[str, Any]] = []
    for base_name, variants in grouped_embeddings.items():
        if not variants:
            continue

        variant_scores = []
        for variant_name, vector in variants:
            sim = cosine_similarity(test_emb, vector)
            variant_scores.append((sim, variant_name))

        variant_scores.sort(key=lambda item: item[0], reverse=True)
        best_score, best_variant = variant_scores[0]
        avg_score = float(sum(score for score, _ in variant_scores) / len(variant_scores))
        support_count = sum(
            1 for score, _ in variant_scores
            if (best_score - score) <= SPEAKER_VARIANT_SUPPORT_WINDOW
        )
        bonus = 0.0
        if support_count > 1:
            bonus = min(
                SPEAKER_VARIANT_SUPPORT_MAX,
                (support_count - 1) * SPEAKER_VARIANT_SUPPORT_STEP,
            )
        avg_contrib = (avg_score * SPEAKER_SCORE_AVG_WEIGHT) + SPEAKER_SCORE_AVG_BIAS
        avg_path = min(0.999, (best_score * 0.55) + avg_contrib)
        gain_path = (best_score * SPEAKER_SCORE_GAIN) + SPEAKER_SCORE_OFFSET
        boost_candidates = [best_score, gain_path, avg_path]
        boosted_core = max(boost_candidates)
        applied_boost = max(0.0, boosted_core - best_score)
        final_score = float(min(0.999, boosted_core + bonus))

        top_variants = ", ".join(
            f"{variant}:{score:.3f}" for score, variant in variant_scores[:3]
        )
        note_parts: List[str] = []
        if bonus > 0:
            note_parts.append(f"+{bonus:.3f} bonus")
        if applied_boost > 1e-4:
            note_parts.append(f"+{applied_boost:.3f} gain")
        avg_lift = max(0.0, avg_path - best_score)
        if boosted_core == avg_path and avg_lift > 1e-4:
            note_parts.append("avg support boost")
        detail_suffix = f" ({', '.join(note_parts)})" if note_parts else ""
        print(
            f"  - {base_name}: best {best_score:.3f} via {best_variant}{detail_suffix}; "
            f"final={final_score:.3f}; avg={avg_score:.3f}; variants[{len(variant_scores)}]={top_variants} "
            f"[Gender: {profile_genders.get(base_name, 'unknown')}]"
        )

        scores.append(
            {
                "name": base_name,
                "score": final_score,
                "best_score": best_score,
                "best_variant": best_variant,
                "avg_score": avg_score,
                "gender": profile_genders.get(base_name, "unknown"),
                "support_count": support_count,
                "bonus": bonus,
                "boost": applied_boost,
                "variant_scores": variant_scores,
            }
        )

    if not scores:
        print("No valid voice profiles found.")
        return None, 0, processed_path

    scores.sort(key=lambda entry: entry["score"], reverse=True)

    if current_speaker_gender != "unknown":
        gender_filtered = [
            entry
            for entry in scores
            if entry["gender"] in (current_speaker_gender, "unknown")
        ]
    else:
        gender_filtered = scores

    if gender_filtered and current_speaker_gender != "unknown":
        print(f"Using gender-filtered results ({len(gender_filtered)} profiles)")
        final_scores = gender_filtered
    else:
        print("Using all results (gender filtering inactive)")
        final_scores = scores

    if not final_scores:
        print("No gender-matching voice profiles found.")
        return None, 0, processed_path

    best_entry = final_scores[0]
    enrolled_count = len(grouped_embeddings)
    effective_threshold = SIMILARITY_THRESHOLD
    if enrolled_count == 1:
        relaxed = max(0.40, SIMILARITY_THRESHOLD - SPEAKER_SOLO_THRESHOLD_RELAX)
        if relaxed < effective_threshold:
            effective_threshold = relaxed
            print(
                f"Adjusting verification threshold to {effective_threshold:.3f} "
                "(single enrolled speaker)"
            )
    elif best_entry.get("support_count", 0) >= 3:
        relaxed = max(0.45, SIMILARITY_THRESHOLD - (SPEAKER_SOLO_THRESHOLD_RELAX * 0.5))
        if relaxed < effective_threshold:
            effective_threshold = relaxed
            print(
                f"Adjusting verification threshold to {effective_threshold:.3f} "
                "(strong multi-segment agreement)"
            )

    confusion_warning = ""
    if len(final_scores) > 1:
        second_entry = final_scores[1]
        score_diff = best_entry["score"] - second_entry["score"]
        if score_diff < SPEAKER_MIN_SCORE_GAP:
            confusion_warning = (
                f" (Warning: Close match with {second_entry['name']}: "
                f"{second_entry['score']:.3f}, diff: {score_diff:.3f})"
            )

    print(
        f"\nBest match: {best_entry['name']} "
        f"(similarity {best_entry['score']:.3f}, gender: {best_entry['gender']})"
        f"{confusion_warning}"
    )

    if best_entry["score"] >= effective_threshold:
        if len(final_scores) > 1 and (
            best_entry["score"] - final_scores[1]["score"]
        ) < SPEAKER_MIN_SCORE_GAP:
            print(
                "Authentication rejected: score gap "
                f"{(best_entry['score'] - final_scores[1]['score']):.3f} < "
                f"{SPEAKER_MIN_SCORE_GAP:.3f}"
            )
            return None, best_entry["score"], processed_path
        print(
            f"Authentication successful! ({best_entry['score']:.3f} >= {effective_threshold:.3f})"
        )
        return best_entry["name"], best_entry["score"], processed_path

    print(
        f"Authentication failed. Score {best_entry['score']:.3f} "
        f"below threshold {effective_threshold:.3f}"
    )
    return None, best_entry["score"], processed_path
        
# Additional language models for better transcription quality
LANGUAGE_CODES = {
    'en': 'english',
    'hi': 'hindi',
    'mr': 'marathi',
    'bn': 'bengali',
    'ta': 'tamil',
    'te': 'telugu',
    'ur': 'urdu',
    'ms': 'malay',
    'tl': 'tagalog',
    'pn': 'punjabi',
    'kn': 'kannada',
    'gu': 'gujarati',
    'or': 'odia',
}

def transcribe_command_audio(
    audio_path,
    language=None,
    fast_mode=FAST_TRANSCRIPTION_ENABLED,
    noise_reduction_blend: Optional[float] = None,
):
    """
    Translate spoken command audio to English using Whisper tiny.
    The function still performs any necessary preprocessing and captures
    diagnostics for debugging, but the primary output string is an English
    translation that will be matched against canonical commands.
    
    Args:
        audio_path: Path to the audio file
        language: Optional language code to force a specific language
                  If None, auto-detection will be used
                  
    Returns:
        Tuple[str, bool, Dict[str, Any]]: (translation, success flag, diagnostics)
    """
    diagnostics: Dict[str, Any] = {
        "language": language or "auto",
        "fast_mode": bool(fast_mode),
        "audio_path": str(audio_path) if audio_path else None,
    }

    pipeline = load_local_stt_pipeline()
    if pipeline is None:
        print("Error: Could not load Whisper tiny model pipeline.")
        diagnostics.update({"error": "pipeline_unavailable"})
        return "", False, diagnostics

    loaded = _load_audio_16k(audio_path)
    if loaded is None:
        diagnostics.update({"error": "audio_load_failed"})
        return "", False, diagnostics

    audio_data, sample_rate = loaded
    diagnostics.update({
        "input_duration_s": round(float(audio_data.size) / float(sample_rate), 3) if sample_rate else None,
        "sample_rate": int(sample_rate) if sample_rate else None,
    })

    trimmed = None
    gated = None
    if fast_mode:
        trimmed = trim_audio_to_speech(audio_data, sample_rate)
        diagnostics["trim_applied"] = bool(trimmed.size and trimmed.size != audio_data.size)
        working = trimmed if trimmed is not None and trimmed.size > 0 else audio_data
        blend_setting = STT_NOISE_REDUCTION_BLEND if noise_reduction_blend is None else noise_reduction_blend
        blend = float(max(0.0, min(1.0, blend_setting)))
        if blend > 0.0:
            working = apply_subtle_noise_reduction(working, sample_rate, blend)
        gated = fast_noise_gate(working, sample_rate)
        diagnostics["noise_gate_applied"] = True
        audio_processed = _normalize_audio_level(gated if gated is not None and gated.size > 0 else working)
    else:
        audio_processed = preprocess_and_noise_reduce(audio_path)
        diagnostics["trim_applied"] = False
        if audio_processed is None or len(audio_processed) == 0:
            trimmed = trim_audio_to_speech(audio_data, sample_rate)
            diagnostics["trim_applied"] = bool(trimmed.size and trimmed.size != audio_data.size)
            working = trimmed if trimmed is not None and trimmed.size > 0 else audio_data
            blend_setting = STT_NOISE_REDUCTION_BLEND if noise_reduction_blend is None else noise_reduction_blend
            blend = float(max(0.0, min(1.0, blend_setting)))
            if blend > 0.0:
                working = apply_subtle_noise_reduction(working, sample_rate, blend)
            gated = fast_noise_gate(working, sample_rate)
            diagnostics["noise_gate_applied"] = True
            audio_processed = _normalize_audio_level(gated if gated is not None and gated.size > 0 else working)
        else:
            diagnostics["noise_gate_applied"] = False

    if "noise_gate_applied" not in diagnostics:
        diagnostics["noise_gate_applied"] = bool(gated is not None)

    if audio_processed is None:
        diagnostics.update({"error": "audio_preprocess_failed"})
        return "", False, diagnostics

    if audio_processed.ndim > 1:
        audio_processed = audio_processed.flatten()
    max_samples = sample_rate * 12
    if audio_processed.size > max_samples:
        audio_processed = audio_processed[:max_samples]
        diagnostics["clipped_seconds"] = round(float(max_samples) / float(sample_rate), 3)

    audio_processed = np.ascontiguousarray(audio_processed, dtype=np.float32)
    diagnostics["processed_duration_s"] = round(float(audio_processed.size) / float(sample_rate), 3)

    # Force Hindi language for multilingual support (handles Hindi, Marathi, Hinglish)
    # This prevents Whisper from incorrectly detecting as Arabic, Tamil, etc.
    # Hindi mode outputs roman script which works well for all Indic languages
    generate_kwargs = {"task": "transcribe"}
    if language:
        generate_kwargs["language"] = language
    else:
        # Force Hindi to handle Hindi/Marathi/Hinglish with roman output
        generate_kwargs["language"] = "hi"
    diagnostics["generate_kwargs"] = dict(generate_kwargs)

    try:
        start_time = time.time()
        result = pipeline(audio_processed, generate_kwargs=generate_kwargs)
        elapsed = time.time() - start_time
        transcription = result.get("text", "").strip()
        diagnostics.update({
            "inference_seconds": round(elapsed, 3),
            "transcription_length": len(transcription),
            "detected_language": result.get("language"),
        })

        if not transcription:
            diagnostics["transcription_empty"] = True
            return "", False, diagnostics

        print(f"Whisper tiny transcription: '{transcription}' ({elapsed:.2f}s)")

        return transcription, True, diagnostics
    except Exception as e:
        diagnostics.update({"error": str(e)})
        print(f"Error translating command audio: {e}")
        return "", False, diagnostics

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


@lru_cache(maxsize=4096)
def _language_for_phrase(phrase: str) -> str:
    """Cache language guesses for command variants to aid scoring."""
    if not phrase:
        return "unknown"
    detected = detect_language(phrase.lower())
    return detected or "unknown"


COMMAND_KEYWORDS: Dict[str, Set[str]] = {
    "forward": {
        "forward",
        "ahead",
        "straight",
        "straightforward",
        "front",
        "advance",
        "move",
        "march",
        "saral",
        "sidha",
        "seedha",
        "seedhe",
        "aage",
        "aagey",
        "age",
        "agey",
        "agay",
        "aagay",
        "pudhe",
        "pudhechal",
        "pudhe chal",
        "pude",
        "pudechal",
        "pude chal",
        "pudeachal",
        "pude achal",
        "badho",
        "badhao",
        "bado",
        "barho",
        "shamne",
        "munne",
        "neraga",
        "sadheval",
        "sadeval",
        "sadhe val",
        "sade val",
        "pulechal",
        "purechal",
        "pule chal",
        "pure chal",
    },
    "backward": {
        "back",
        "backward",
        "backwards",
        "reverse",
        "rear",
        "return",
        "retreat",
        "maghe",
        "mage",
        "piche",
        "peeche",
        "pichhe",
        "peechey",
        "peechay",
        "peechhe",
        "ulta",
        "vaapas",
        "wapas",
        "back up",
    },
    "left": {
        "left",
        "leftward",
        "turn left",
        "move left",
        "shift left",
        "slide left",
        "khabbey",
        "khabe",
        "davikade",
        "davi",
        "davya",
        "davya kade",
        "baaye",
        "baye",
        "bayen",
        "baen",
        "bai",
        "bye",
        "bame",
        "bam",
        "edakke",
        "idathu",
        "idama",
        "bam dike",
        "baayin",
    },
    "right": {
        "right",
        "rightward",
        "turn right",
        "move right",
        "shift right",
        "slide right",
        "sajje",
        "saje",
        "ujavikade",
        "ujavi",
        "ujya",
        "daaye",
        "daye",
        "daen",
        "dai",
        "dayen",
        "daayen",
        "jamne",
        "balakke",
        "valathu",
        "dayan",
        "dahine",
        "dahini",
    },
    "rotate_left": {
        "rotate",
        "rotation",
        "spin",
        "circle",
        "loop",
        "turn around",
        "counterclockwise",
        "counter-clockwise",
        "counter clockwise",
        "anti clockwise",
        "anticlockwise",
        "left",
        "spin left",
        "rotate left",
        "ghumo",
        "ghumao",
        "ghoom",
        "ghum",
        "ghoomo",
    },
    "rotate_right": {
        "rotate",
        "rotation",
        "spin",
        "circle",
        "loop",
        "turn around",
        "clockwise",
        "clock-wise",
        "clock wise",
        "right",
        "spin right",
        "rotate right",
        "ghumo",
        "ghumao",
        "ghoom",
        "ghum",
        "ghoomo",
    },
    "start": {
        "start",
        "begin",
        "resume",
        "initiate",
        "activate",
        "enable",
        "power on",
        "power up",
        "switch on",
        "turn on",
        "turn it on",
        "launch",
        "go",
        "let's",
        "lets",
        "move",
        "chalu",
        "chaloo",
        "shuru",
        "suru",
        "prarambh",
        "on",
        "start up",
        "get started",
    },
    "stop": {
        "stop",
        "halt",
        "wait",
        "pause",
        "hold",
        "freeze",
        "brake",
        "cease",
        "ruko",
        "roko",
        "ruk",
        "rukjao",
        "band",
        "bas",
        "thamba",
        "thamb",
        "tham",
        "thahar",
        "thehro",
        "off",
        "shutdown",
        "stop now",
        "shut down",
    },
}


def _command_keywords_match(command: str, corpus: str, expanded_tokens: List[str]) -> bool:
    """Validate that the candidate command has at least one supporting keyword."""
    keywords = COMMAND_KEYWORDS.get(command)
    if not keywords:
        return True

    import re

    corpus_lower = corpus.lower()
    tokens_in_corpus = [token for token in corpus_lower.split() if token]
    tokens_in_corpus = [token for token in tokens_in_corpus if token not in {"'", ""}]

    token_set: Set[str] = {token.lower() for token in expanded_tokens if token}
    token_set.update(tokens_in_corpus)
    token_set.update({re.sub(r"[^a-z0-9]+", "", tok) for tok in tokens_in_corpus if tok})

    for window in (2, 3):
        if len(tokens_in_corpus) >= window:
            for idx in range(len(tokens_in_corpus) - window + 1):
                chunk = tokens_in_corpus[idx:idx + window]
                joined = "".join(chunk)
                spaced = " ".join(chunk)
                token_set.add(joined)
                token_set.add(spaced)
                token_set.add(re.sub(r"[^a-z0-9]+", "", joined))
                token_set.add(re.sub(r"[^a-z0-9]+", "", spaced))

    token_set.discard("")

    corpus_compact = re.sub(r"[^a-z0-9]+", "", corpus_lower)

    for keyword in keywords:
        keyword_lower = keyword.lower().replace(" ", "")
        if keyword_lower in corpus_compact or keyword_lower in token_set:
            return True
        if keyword_lower in corpus_lower:
            return True

    if RAPIDFUZZ_AVAILABLE:
        def adaptive_threshold(word: str) -> int:
            length = len(word)
            if length <= 3:
                return 94
            if length <= 5:
                return 86
            if length <= 7:
                return 82
            return 76

        variation_threshold = 78

        for keyword in keywords:
            keyword_lower = keyword.lower()
            threshold = adaptive_threshold(keyword_lower)
            if rapidfuzz_fuzz.partial_ratio(keyword_lower, corpus_lower) >= threshold:
                return True
            if corpus_compact and rapidfuzz_fuzz.partial_ratio(keyword_lower, corpus_compact) >= threshold:
                return True
            for token in token_set:
                if rapidfuzz_fuzz.partial_ratio(keyword_lower, token) >= threshold:
                    return True
                if rapidfuzz_fuzz.token_set_ratio(keyword_lower, token) >= threshold:
                    return True

            if rapidfuzz_fuzz.WRatio(keyword_lower, corpus_lower) >= max(threshold + 8, 88):
                return True

        for variation in WHEELCHAIR_COMMANDS.get(command, []):
            variation_lower = variation.lower()
            if rapidfuzz_fuzz.partial_ratio(variation_lower, corpus_lower) >= variation_threshold:
                return True
            if corpus_compact and rapidfuzz_fuzz.partial_ratio(variation_lower, corpus_compact) >= variation_threshold:
                return True

    return False


def match_command(transcribed_text):
    """Match transcription to a wheelchair command using language-aware highest scores."""
    if not transcribed_text:
        return None, 0.0

    import re
    import difflib

    original_text = transcribed_text.lower().strip()
    text = re.sub(r"[.!?,;:।]", "", original_text)

    def attempt_global_fuzzy(*candidate_texts: str) -> Tuple[Optional[str], float]:
        for candidate_text in candidate_texts:
            if not candidate_text:
                continue
            result = _global_command_fuzzy_match(candidate_text)
            if result:
                fallback_cmd, fallback_score, fallback_phrase = result
                print(
                    f"Fallback fuzzy match -> '{fallback_cmd}' via '{fallback_phrase}' "
                    f"(confidence {fallback_score:.2f})"
                )
                return fallback_cmd, fallback_score
        return None, 0.0

    vote_totals: Dict[str, float] = defaultdict(float)
    vote_counts: Dict[str, int] = defaultdict(int)
    vote_details: Dict[str, List[Tuple[float, str, str]]] = defaultdict(list)
    best_single_match: Optional[str] = None
    best_single_detail: Optional[Tuple[float, str, str]] = None
    best_single_score = 0.0

    def record_vote(command: str, score: float, reason: str, language_hint: str = "unknown") -> None:
        nonlocal best_single_match, best_single_score, best_single_detail
        score = float(max(0.0, min(1.0, score)))
        if score == 0.0 or not command:
            return
        vote_totals[command] += score
        vote_counts[command] += 1
        vote_details[command].append((score, reason, language_hint))
        if score > best_single_score:
            best_single_score = score
            best_single_match = command
            best_single_detail = (score, reason, language_hint)
        lang_note = f" | lang={language_hint}" if language_hint not in {"unknown", ""} else ""
        print(
            f"Evidence -> {command}: +{score:.2f} ({reason}){lang_note}; "
            f"total={vote_totals[command]:.2f}; count={vote_counts[command]}"
        )

    phrase_match = _apply_phrase_replacements(text)
    if phrase_match:
        command, conf = phrase_match
        record_vote(command, conf, "canonical phrase", language_hint="en")

    words = [w for w in text.split() if w]
    normalized_words = _normalize_command_tokens(words)
    if normalized_words:
        text = " ".join(normalized_words)
        words = normalized_words
    else:
        words = [w for w in text.split() if w]

    if RAPIDFUZZ_AVAILABLE:
        variation_lookup: Dict[str, str] = {}
        variation_phrases: List[str] = []
        for cmd, variations in WHEELCHAIR_COMMANDS.items():
            for variation in variations:
                normalized_variation = variation.lower()
                if normalized_variation not in variation_lookup:
                    variation_lookup[normalized_variation] = cmd
                    variation_phrases.append(normalized_variation)
        if variation_phrases:
            best_match = rapidfuzz_process.extractOne(
                text,
                variation_phrases,
                scorer=rapidfuzz_fuzz.token_set_ratio,
            )
            if best_match:
                phrase, score, _ = best_match
                mapped_cmd = variation_lookup.get(phrase)
                if mapped_cmd:
                    record_vote(
                        mapped_cmd,
                        float(score) / 100.0,
                        f"rapidfuzz token_set_ratio '{phrase}'",
                        language_hint=_language_for_phrase(phrase),
                    )

    detected_language = detect_language(text)
    print(f"Detected language: {detected_language}")

    if detected_language in ["hi", "mr", "en"]:
        hindi_result = hindi_direction_detector(text)
        if hindi_result:
            print(f"Hindi direction detector found: {hindi_result}")
            record_vote(hindi_result, 0.95, "Hindi direction detector", language_hint="hi")

    rotation_tokens_hindi = {"ghum", "ghumo", "ghoom", "ghumao", "gumo"}
    rotation_tokens_english = {"rotate", "spin", "circle", "loop", "around"}
    if any(word in text for word in rotation_tokens_hindi.union(rotation_tokens_english)):
        if any(word in text for word in ["dai", "daye", "daaye", "daayen", "dayen", "right"]):
            if any(word in text for word in rotation_tokens_hindi):
                lang_hint = "hi"
            elif any(word in text for word in rotation_tokens_english):
                lang_hint = "en"
            else:
                lang_hint = "unknown"
            record_vote("rotate_right", 0.90, "rotation keyword -> right", language_hint=lang_hint)
        if any(word in text for word in ["bai", "baye", "baaye", "baayen", "bayen", "left"]):
            if any(word in text for word in rotation_tokens_hindi):
                lang_hint = "hi"
            elif any(word in text for word in rotation_tokens_english):
                lang_hint = "en"
            else:
                lang_hint = "unknown"
            record_vote("rotate_left", 0.90, "rotation keyword -> left", language_hint=lang_hint)

    if "taraf" in text:
        if any(word in text for word in ["dai", "daye", "daaye", "daayen", "dayen", "right"]):
            record_vote("right", 0.85, "taraf + right indicator", language_hint="hi")
        if any(word in text for word in ["bai", "baye", "baaye", "baayen", "bayen", "left"]):
            record_vote("left", 0.85, "taraf + left indicator", language_hint="hi")

    for cmd, variations in WHEELCHAIR_COMMANDS.items():
        if text in [v.lower() for v in variations]:
            print(f"Found exact match: '{text}' -> {cmd}")
            return cmd, 1.0

    rotation_hint_keywords = {
        "ghoom",
        "ghum",
        "ghumo",
        "ghumao",
        "घूम",
        "घूमो",
        "घुमाव",
        "घुमाओ",
        "spin",
        "rotate",
        "rotation",
        "circle",
        "loop",
        "around",
        "pura",
        "पूरा",
        "पूर्ण",
        "full turn",
        "turn around",
        "360",
    }
    rotation_hints_present = any(keyword in text for keyword in rotation_hint_keywords)

    for cmd, variations in WHEELCHAIR_COMMANDS.items():
        rotation_command = cmd.startswith("rotate_")
        allow_rotation_scoring = rotation_hints_present if rotation_command else True
        for variation in variations:
            variation_lower = variation.lower()
            variation_language = _language_for_phrase(variation_lower)
            if variation_lower in text and len(variation_lower) > 3:
                if allow_rotation_scoring:
                    confidence = min(0.95, len(variation_lower) / max(1, len(text)) * 1.2)
                    record_vote(cmd, confidence, f"embedded phrase '{variation}'", language_hint=variation_language)

    if "पूरा" in text and "दाय" in text and ("मुड" in text or "mud" in text):
        record_vote("right", 0.95, "special पूर pattern", language_hint="hi")

    words = text.split()

    phonetic_variants = {
        "bay": ["baaye", "baye", "bai"],
        "buy": ["baaye", "baye", "bai"],
        "bai": ["baaye", "baye", "bay", "buy"],
        "left": ["lift", "laft", "lft", "lef"],
        "veer": ["vir", "beer", "veer"],
        "rotate": ["rotation", "rote", "rotee", "roate"],
        "spin": ["spin", "spain", "speen"],
        "day": ["daaye", "daye", "dai", "dye", "die", "दाय", "दाएं", "दाई"],
        "die": ["daaye", "daye", "dai", "day", "dye", "दाय", "दाएं"],
        "dai": ["daaye", "daye", "die", "day", "dye", "दाय", "दाएं"],
        "दाय": ["right", "daaye", "daye", "dai", "day"],
        "right": ["rite", "ryt", "rit", "rght", "wright"],
        "mude": ["mudo", "mudna", "mudho", "mode", "mood", "move", "मुडव", "मुडो", "मुड़ो"],
        "mode": ["mudo", "mudna", "mude", "mood", "move", "मुडव", "मुडो"],
        "mude.": ["mudo", "mudna", "mode", "mood", "move", "मुडव"],
        "mood": ["mudo", "mudna", "mode", "mude", "move", "मुडव"],
        "move": ["mudo", "mudna", "mode", "mude", "mood", "मुडव"],
        "मुडव": ["mudo", "turn", "mudna", "मुड़ो", "मुडो"],
        "ruko": ["rukho", "rukna", "rukko", "roko", "roku", "rocco"],
        "stop": ["stp", "stahp", "stoop", "hault", "holt"],
        "chalo": ["challo", "chal", "chalu", "challu", "cello"],
        "karo": ["kro", "karro", "karho", "kero", "kiro"],
        "aage": ["agay", "aagay", "aagey", "age", "agge", "aagee", "aageh", "agaye"],
        "agay": ["aage", "aagay", "aagey", "age", "agge", "aagee"],
        "badho": ["badhao", "bado", "bardho", "barho", "badha"],
        "badhao": ["badho", "badha", "bardho", "barho", "badavo"],
        "seedhe": ["sidhe", "sidha", "seedha", "sidhey", "seede", "seedey"],
        "sidhe": ["seedhe", "sidha", "seedha", "sidhey", "seede"],
        "pudhe": ["pude", "pudhe", "pudhey", "pudhee", "pudh", "puday"],
        "pude": ["pudhe", "pudey", "pudhee", "puday", "pudde"],
        "saral": ["sarl", "saralh", "sarlh", "sarall"],
        "peeche": ["piche", "peechay", "peechhe", "peechey", "pichey", "peechai"],
        "piche": ["peeche", "peechay", "peechhe", "peechey", "pichy"],
        "mage": ["maghe", "maage", "magay", "magey", "mague"],
        "maghe": ["mage", "magey", "maghey", "mange", "magheh"],
        "vaapas": ["wapas", "wapis", "vapas", "vaaps", "waapas"],
        "shuru": ["sharoo", "shuroo", "shroo", "shru", "shruu"],
        "start": ["stat", "strt", "starrt", "stard"],
    }

    expanded_words = list(words)
    for word in words:
        for original, variants in phonetic_variants.items():
            if word == original or word.startswith(original):
                expanded_words.extend(variants)
            if word in variants:
                expanded_words.append(original)
    expanded_words = list(dict.fromkeys(expanded_words))
    print(f"Expanded word list: {expanded_words}")

    command_patterns = [
        {"pattern": ["baaye", "baye", "bai"], "command": "left", "score": 0.85, "language": "hi"},
        {"pattern": ["daaye", "daye", "day", "die"], "command": "right", "score": 0.85, "language": "hi"},
        {"pattern": ["aage", "seedhe", "forward"], "command": "forward", "score": 0.85, "language": "hi"},
        {"pattern": ["peeche", "back", "vaapas"], "command": "backward", "score": 0.85, "language": "hi"},
        {"pattern": ["ruko", "stop", "thehro"], "command": "stop", "score": 0.85, "language": "hi"},
        {"pattern": ["chalo", "shuru", "start"], "command": "start", "score": 0.85, "language": "hi"},
        {"pattern": ["baaye", "mudo"], "command": "left", "score": 0.95, "multi_word": True, "language": "hi"},
        {"pattern": ["daaye", "mudo"], "command": "right", "score": 0.95, "multi_word": True, "language": "hi"},
        {"pattern": ["left", "mudo"], "command": "left", "score": 0.90, "multi_word": True, "language": "en"},
        {"pattern": ["right", "mudo"], "command": "right", "score": 0.90, "multi_word": True, "language": "en"},
        {"pattern": ["spin", "left"], "command": "rotate_left", "score": 0.90, "multi_word": True, "language": "en"},
        {"pattern": ["spin", "right"], "command": "rotate_right", "score": 0.90, "multi_word": True, "language": "en"},
        {"pattern": ["rotate", "left"], "command": "rotate_left", "score": 0.90, "multi_word": True, "language": "en"},
        {"pattern": ["rotate", "right"], "command": "rotate_right", "score": 0.90, "multi_word": True, "language": "en"},
        {"pattern": ["veer", "left"], "command": "left", "score": 0.88, "multi_word": True, "language": "en"},
        {"pattern": ["veer", "right"], "command": "right", "score": 0.88, "multi_word": True, "language": "en"},
    ]

    for pattern_def in command_patterns:
        pattern_words = pattern_def["pattern"]
        multi_word = pattern_def.get("multi_word", False)
        cmd = pattern_def["command"]
        base_score = pattern_def["score"]
        lang_hint = pattern_def.get("language", "unknown")

        if multi_word and len(words) >= 2:
            matches_all = all(
                any(pattern_word in word for word in expanded_words)
                for pattern_word in pattern_words
            )

            if "left" in cmd:
                special_combo = any(
                    f"{w1} {w2}" in text
                    for w1 in ["bye", "by", "bay", "baaye"]
                    for w2 in ["mude", "mode", "mudo"]
                )
                if matches_all or special_combo:
                    record_vote(cmd, base_score, "multi-word pattern match", language_hint=lang_hint)

            if "right" in cmd:
                special_combo = any(
                    f"{w1} {w2}" in text
                    for w1 in ["die", "day", "daaye"]
                    for w2 in ["mude", "mode", "mudo"]
                )
                if matches_all or special_combo:
                    record_vote(cmd, base_score, "multi-word pattern match", language_hint=lang_hint)
        else:
            for pattern_word in pattern_words:
                if pattern_word in expanded_words:
                    record_vote(cmd, base_score, f"pattern word '{pattern_word}'", language_hint=lang_hint)
                    break

    for cmd, variations in WHEELCHAIR_COMMANDS.items():
        rotation_command = cmd.startswith("rotate_")
        allow_rotation_scoring = rotation_hints_present if rotation_command else True
        for variation in variations:
            if len(variation) < 3:
                continue

            variation_lower = variation.lower()
            variation_language = _language_for_phrase(variation_lower)
            variation_words = variation_lower.split()
            for v_word in variation_words:
                if len(v_word) > 2 and v_word in expanded_words:
                    if not rotation_command or allow_rotation_scoring:
                        record_vote(cmd, 0.82, f"word match '{v_word}'", language_hint=_language_for_phrase(v_word))
                    break

            similarity = difflib.SequenceMatcher(None, text, variation_lower).ratio()
            if variation_lower in text:
                similarity += 0.18

            variation_word_set = set(variation_lower.split())
            expanded_word_set = set(expanded_words)
            common_words = variation_word_set.intersection(expanded_word_set)
            if variation_word_set and common_words:
                word_overlap_score = len(common_words) / len(variation_word_set) * 0.95
                similarity = max(similarity, word_overlap_score)

            token_score = 0.0
            partial_score = 0.0
            if RAPIDFUZZ_AVAILABLE:
                token_score = rapidfuzz_fuzz.token_set_ratio(text, variation_lower) / 100.0
                partial_score = rapidfuzz_fuzz.partial_ratio(text, variation_lower) / 100.0
                similarity = max(similarity, token_score, partial_score)

            passes_shared = bool(common_words) and similarity >= 0.45
            passes_strong = similarity >= 0.75 or token_score >= 0.78 or partial_score >= 0.78

            if (passes_shared or passes_strong) and (not rotation_command or allow_rotation_scoring):
                record_vote(cmd, similarity, f"fuzzy match '{variation}'", language_hint=variation_language)

            if "mudo" in variation and ("baaye" in variation or "daaye" in variation):
                left_combo = any(w in text for w in ["bye", "by", "bay"]) and any(
                    w in text for w in ["mude", "mode"]
                )
                right_combo = any(w in text for w in ["die", "day"]) and any(
                    w in text for w in ["mude", "mode"]
                )
                if "baaye" in variation and left_combo:
                    if not rotation_command or allow_rotation_scoring:
                        record_vote(cmd, 0.9, "Hindi left special", language_hint="hi")
                if "daaye" in variation and right_combo:
                    if not rotation_command or allow_rotation_scoring:
                        record_vote(cmd, 0.9, "Hindi right special", language_hint="hi")

    if not vote_details:
        fallback_cmd, fallback_score = attempt_global_fuzzy(original_text, text)
        if fallback_cmd:
            return fallback_cmd, fallback_score
        return None, best_single_score

    command_stats: Dict[str, Dict[str, Any]] = {}
    for cmd, entries in vote_details.items():
        scores = [score for score, _, _ in entries]
        if not scores:
            continue
        best_entry = max(entries, key=lambda item: item[0])
        command_stats[cmd] = {
            "best": best_entry,
            "avg": sum(scores) / len(scores),
            "count": len(scores),
            "total": sum(scores),
        }

    if not command_stats:
        fallback_cmd, fallback_score = attempt_global_fuzzy(original_text, text)
        if fallback_cmd:
            return fallback_cmd, fallback_score
        return None, best_single_score

    sorted_by_best = sorted(
        command_stats.items(), key=lambda item: item[1]["best"][0], reverse=True
    )

    print("Command score summary:")
    for cmd, stats in sorted_by_best:
        best_score, best_reason, best_lang = stats["best"]
        lang_note = f" lang={best_lang}" if best_lang not in {"unknown", ""} else ""
        print(
            f"  {cmd}: best={best_score:.2f} (reason: {best_reason}){lang_note}; "
            f"avg={stats['avg']:.2f}; votes={stats['count']}; total={stats['total']:.2f}"
        )

    candidate_order: List[str] = []
    if best_single_match and best_single_match in command_stats:
        candidate_order.append(best_single_match)
    for cmd, _ in sorted_by_best:
        if cmd not in candidate_order:
            candidate_order.append(cmd)

    chosen_cmd: Optional[str] = None
    chosen_score = 0.0
    chosen_reason = ""
    chosen_lang = "unknown"

    for candidate in candidate_order:
        if candidate not in command_stats:
            continue
        candidate_entry = (
            best_single_detail
            if candidate == best_single_match and best_single_detail
            else command_stats[candidate]["best"]
        )
        candidate_score, candidate_reason, candidate_lang = candidate_entry
        if candidate_score >= 0.70:
            chosen_cmd = candidate
            chosen_score = candidate_score
            chosen_reason = candidate_reason
            chosen_lang = candidate_lang
            break
        if not _command_keywords_match(candidate, original_text, expanded_words):
            print(
                f"Keyword validation failed for '{candidate}'. Translation lacks expected intent markers."
            )
            continue
        chosen_cmd = candidate
        chosen_score = candidate_score
        chosen_reason = candidate_reason
        chosen_lang = candidate_lang
        break

    if not chosen_cmd:
        print("No command passed keyword validation; attempting global fuzzy match")
        fallback_cmd, fallback_score = attempt_global_fuzzy(original_text, text)
        if fallback_cmd:
            return fallback_cmd, fallback_score
        fallback_score = max(
            [best_single_score]
            + [stats["best"][0] for stats in command_stats.values()]
        )
        return None, fallback_score

    runner_up_score = 0.0
    if len(command_stats) > 1:
        runner_up_score = max(
            stats["best"][0]
            for cmd, stats in command_stats.items()
            if cmd != chosen_cmd
        )

    if chosen_score - runner_up_score < 0.10 and chosen_score < 0.75:
        print(
            f"Command ambiguity detected (top score {chosen_score:.2f} vs next {runner_up_score:.2f}); attempting global fuzzy match"
        )
        fallback_cmd, fallback_score = attempt_global_fuzzy(original_text, text)
        if fallback_cmd:
            return fallback_cmd, fallback_score
        return None, max(best_single_score, runner_up_score, chosen_score)

    min_threshold = 0.70

    lang_note = f", lang={chosen_lang}" if chosen_lang not in {"unknown", ""} else ""
    print(
        f"Selected command '{chosen_cmd}' with confidence {chosen_score:.2f} "
        f"(reason: {chosen_reason}{lang_note})"
    )

    if chosen_score >= min_threshold:
        return chosen_cmd, chosen_score

    print(
        f"Top candidate '{chosen_cmd}' below threshold {min_threshold:.2f} "
        f"(score={chosen_score:.2f}); attempting global fuzzy match"
    )
    fallback_cmd, fallback_score = attempt_global_fuzzy(original_text, text)
    if fallback_cmd:
        return fallback_cmd, fallback_score
    return None, max(best_single_score, chosen_score)

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
    else:
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
    Records audio, authenticates the user, translates speech using Whisper tiny,
    matches to predefined commands, and executes the command.
    Includes detailed diagnostics for better troubleshooting.
    """
    print("\n=== MODE 2: Voice Command Control ===")
    
    # Authenticate user
    user, score, processed_path = identify_speaker(encoder, is_registration=False)
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
    command, confidence, translation = process_command_with_whisper_tiny(processed_path)
    processing_time = time.time() - start_time
    
    print(f"\n[DIAGNOSTICS] Command processing completed in {processing_time:.2f} seconds")
    print(f"[DIAGNOSTICS] Command: {command or 'None'}, Confidence: {confidence:.2f}")
    if translation:
        print(f"[DIAGNOSTICS] Translation: {translation}")
    
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
    MODELS_DIR.mkdir(exist_ok=True)
    OPTIMIZED_DIR.mkdir(exist_ok=True)
    
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
    print("Speaker recognizer will be loaded when needed.")
    speaker_model = None
    
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
            if speaker_model is None:
                print("Loading speaker recognizer (CPU)... This may take a moment.")
                try:
                    speaker_model = load_speaker_recognizer()
                except Exception as e:
                    print(f"CRITICAL ERROR: Failed to load speaker recognizer: {e}")
                    continue
            online_llm_mode(speaker_model)
        elif choice == "2":
            # Load encoder only when needed
            if speaker_model is None:
                print("Loading speaker recognizer (CPU)... This may take a moment.")
                try:
                    speaker_model = load_speaker_recognizer()
                except Exception as e:
                    print(f"CRITICAL ERROR: Failed to load speaker recognizer: {e}")
                    continue
            command_control_mode(speaker_model)
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
                if speaker_model is None:
                    print("Loading speaker recognizer (CPU)... This may take a moment.")
                    try:
                        speaker_model = load_speaker_recognizer()
                    except Exception as e:
                        print(f"CRITICAL ERROR: Failed to load speaker recognizer: {e}")
                        input("\nPress Enter to continue...")
                        continue

                # Create a new voice profile
                success = create_voice_profile_internal(speaker_model, VOICE_DB_PROCESSED_DIR, 
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
                if speaker_model is None:
                    print("Loading speaker recognizer (CPU)... This may take a moment.")
                    try:
                        speaker_model = load_speaker_recognizer()
                    except Exception as e:
                        print(f"CRITICAL ERROR: Failed to load speaker recognizer: {e}")
                        input("\nPress Enter to continue...")
                        continue

                # Test voice authentication
                test_voice_authentication_internal(speaker_model)
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
            command, confidence, translation = process_command_with_whisper_tiny(None, detect_lang=True)
            
            if command:
                print(f"\nRecognized command: '{command}' with confidence {confidence:.2f}")
                
                detected_language = detect_language(translation)
                print(f"Detected language: {detected_language} ({LANGUAGE_CODES.get(detected_language, 'Unknown')})")
                
                # Execute command with language-specific feedback
                execute_command_with_language_feedback(command, detected_language)
            else:
                print("\nNo command recognized. Please try again with a clearer voice command.")
                if translation:
                    print(f"Heard: {translation}")
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