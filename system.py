import os
import sys
import math
import time
import urllib.request
import subprocess
from pathlib import Path
from datetime import datetime, timezone # MODIFIED
import requests  # --- NEW --- For Hugging Face API calls
from dotenv import load_dotenv # --- NEW --- To load .env file

# Force CPU-only execution for all libraries
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

try:
    from pydub import AudioSegment
except ImportError:
    AudioSegment = None

import numpy as np
import sounddevice as sd
import scipy.io.wavfile as wav

# Optional noise reduction
try:
    import noisereduce as nr
except ImportError:
    nr = None

try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    from resemblyzer.audio import sampling_rate as RS_SAMPLING_RATE
except ImportError:
    print("Missing dependency: resemblyzer. Install with: pip install resemblyzer")
    sys.exit(1)

try:
    import torch
    from transformers import (
        MarianMTModel,
        MarianTokenizer,
        AutoConfig,
        AutoProcessor,
        AutoModelForSpeechSeq2Seq,
        pipeline,
        GenerationConfig,
        # Added for local CPU-only LLM without downloads
        AutoModelForCausalLM,
        AutoTokenizer,
    )
except ImportError:
    print("Missing dependency: transformers. Install with: pip install transformers")
    sys.exit(1)

# --- GLOBAL CONFIGURATION ---
VOICE_DB_PROCESSED_DIR = Path("./voice_db_processed")
VOICE_DB_EMBEDDINGS_DIR = Path("./voice_db_embeddings")
OPTIMIZED_DIR = Path("./optimized_models")
MODELS_DIR = Path("./models")
TTS_OUTPUT_DIR = Path("./tts_outputs")
TEMP_DIR = Path("./temp")
VOICE_MODEL_SEARCH_DIRS = [MODELS_DIR, OPTIMIZED_DIR]

SAMPLE_RATE = RS_SAMPLING_RATE
RECORD_DURATION = 5
SIMILARITY_THRESHOLD = 0.70

ALLOW_TRANSLATION_AUTO_DOWNLOAD = False
ASK_BEFORE_DOWNLOAD = True
ESTIMATED_MARIAN_SIZE_MB = 306

# --- NEW: HUGGING FACE LLM CONFIGURATION ---
HF_API_TOKEN = None  # Will be loaded from .env
LLM_MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct" # Powerful, recommended model
LLM_API_URL = f"https://api-inference.huggingface.co/models/{LLM_MODEL_ID}"
LLM_HEADERS = {} # Will be populated after loading token
# Fallback models that are broadly accessible on the Inference API
LLM_FALLBACK_MODELS = [
    "microsoft/Phi-3-mini-4k-instruct",
    "HuggingFaceH4/zephyr-7b-beta",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
]
# Optional local LLM (Transformers) fallback
ENABLE_LOCAL_LLM = False
LOCAL_LLM_MODEL_ID = "HuggingFaceH4/zephyr-7b-beta"
_local_llm_pipe = None
# Router (OpenAI-compatible) configuration
USE_HF_ROUTER = False
HF_ROUTER_TOKEN = None
ROUTER_API_BASE = "https://router.huggingface.co/v1"
ROUTER_MODEL_ID = "HuggingFaceH4/zephyr-7b-beta:featherless-ai"
ROUTER_MODEL_FALLBACKS = [
    "Qwen/Qwen2.5-7B-Instruct:together",
]

# System prompt for all LLM responses
SYSTEM_PROMPT = (
    "you are Cruzer, an AI bot for a Smart IOT enabled Wheelchair, product name- SmartNav "
    "and are mostly talking with a disabled or an old aged person , talk like a human and in the "
    "most simplest way possible for anyone to understand and provide short answers"
)

PIPER_VOICES = {
    "en": {
        "female": {
            "name": "en_US-lessac-medium",
            "onnx_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx",
            "json_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json"
        },
        "male": {
            "name": "en_US-ryan-medium",
            "onnx_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx",
            "json_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/ryan/medium/en_US-ryan-medium.onnx.json"
        }
    },
    "hi": {
        "female": {
            "name": "hi_IN-priyamvada-medium",
            "onnx_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/priyamvada/medium/hi_IN-priyamvada-medium.onnx",
            "json_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/priyamvada/medium/hi_IN-priyamvada-medium.onnx.json"
        },
        "male": {
            "name": "hi_IN-rohan-medium",
            "onnx_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/rohan/medium/hi_IN-rohan-medium.onnx",
            "json_url": "https://huggingface.co/rhasspy/piper-voices/resolve/main/hi/hi_IN/rohan/medium/hi_IN-rohan-medium.onnx.json"
        }
    },
    "mr": {
        "female": {
            "name": "mr_IN-kalpana-medium",
            "onnx_url": None,
            "json_url": None
        },
        "male": {
            "name": "mr_IN-kalpana-medium",
            "onnx_url": None,
            "json_url": None
        }
    }
}

TRANSLATION_MODEL_IDS = {
    "hi": "Helsinki-NLP/opus-mt-en-hi",
    "mr": "Helsinki-NLP/opus-mt-en-mr",
}

_translation_cache = {}
_stt_pipe = None

WHISPER_SMALL_MODEL_ID = "openai/whisper-small"
WHISPER_SMALL_WEIGHTS = OPTIMIZED_DIR / "whisper_small_weights_cpu_int8.pt"


# --- NEW: HUGGING FACE LLM INTEGRATION ---
def load_api_key():
    """Loads the Hugging Face API key from .env file."""
    global HF_API_TOKEN, LLM_HEADERS, LLM_MODEL_ID, LLM_API_URL, LOCAL_LLM_MODEL_ID, ENABLE_LOCAL_LLM
    global USE_HF_ROUTER, HF_ROUTER_TOKEN, ROUTER_MODEL_ID, ROUTER_MODEL_FALLBACKS
    load_dotenv()
    HF_API_TOKEN = os.getenv("hfapi")
    # Allow overriding the default LLM via env var
    env_model = os.getenv("HF_LLM_MODEL_ID")
    if env_model:
        LLM_MODEL_ID = env_model.strip()
        LLM_API_URL = f"https://api-inference.huggingface.co/models/{LLM_MODEL_ID}"
        print(f"Using LLM model from env: {LLM_MODEL_ID}")
    # Local LLM toggles
    env_local_model = os.getenv("LOCAL_LLM_MODEL_ID")
    if env_local_model:
        LOCAL_LLM_MODEL_ID = env_local_model.strip()
    ENABLE_LOCAL_LLM = os.getenv("ENABLE_LOCAL_LLM", "false").strip().lower() in ("1", "true", "yes", "on")
    if ENABLE_LOCAL_LLM:
        print(f"Local LLM enabled: {LOCAL_LLM_MODEL_ID}")
    # Router token and model id
    HF_ROUTER_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HF_ROUTER_TOKEN")
    router_model_env = os.getenv("HF_ROUTER_MODEL_ID")
    if router_model_env:
        ROUTER_MODEL_ID = router_model_env.strip()
    # Optional: comma-separated list of fallbacks
    router_fallbacks_env = os.getenv("HF_ROUTER_FALLBACK_MODELS")
    if router_fallbacks_env:
        ROUTER_MODEL_FALLBACKS = [m.strip() for m in router_fallbacks_env.split(",") if m.strip()]
    USE_HF_ROUTER = bool(HF_ROUTER_TOKEN)
    if USE_HF_ROUTER:
        print(f"HF Router enabled with model: {ROUTER_MODEL_ID}")
        if ROUTER_MODEL_FALLBACKS:
            print(f"HF Router fallbacks: {ROUTER_MODEL_FALLBACKS}")
    if not HF_API_TOKEN and not HF_ROUTER_TOKEN:
        print("ERROR: No Hugging Face token found.")
        print("Set either 'hfapi' in .env for Inference API, or 'HF_TOKEN' for Router API.")
        return False
    if HF_API_TOKEN:
        LLM_HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"}
        print("Hugging Face API token loaded successfully.")
    return True

def query_llm(text_prompt: str) -> str:
    """Sends a prompt to the Hugging Face LLM and returns the response. Tries Router first, then fallbacks."""
    # Prefer Router (OpenAI-compatible) if configured
    router_out = _query_llm_via_router(text_prompt)
    if router_out:
        return router_out

    # If router not configured or failed, try Inference API path
    if not HF_API_TOKEN:
        if ENABLE_LOCAL_LLM:
            local = _try_local_llm(text_prompt)
            if local:
                return local
        return "Sorry, my connection to the language model is not configured."

    # Include system prompt in instruction format
    formatted_prompt = f"[INST] {SYSTEM_PROMPT}\nUser: {text_prompt} [/INST]"
    models_to_try = [LLM_MODEL_ID] + [m for m in LLM_FALLBACK_MODELS if m != LLM_MODEL_ID]
    payload = {
        "inputs": formatted_prompt,
        "parameters": {
            "max_new_tokens": 150,
            "temperature": 0.7,
            "return_full_text": False,
        },
        "options": {"wait_for_model": True}
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

            if isinstance(result, list) and result and 'generated_text' in result[0]:
                return result[0]['generated_text'].strip()
            if isinstance(result, dict) and 'generated_text' in result:
                return result['generated_text'].strip()
            if isinstance(result, dict) and 'error' in result:
                err_msg = result.get('error', '')
                if 'is currently loading' in err_msg:
                    print("Model is loading on Hugging Face, please wait a moment and try again...")
                    return "The AI model is warming up. Please ask me again in a minute."
                print(f"LLM error from '{model_id}': {err_msg}. Trying fallback...")
                last_error = err_msg
                continue

            print(f"LLM '{model_id}' returned an unexpected format: {result}. Trying fallback...")
            last_error = "unexpected format"
            continue

        except requests.exceptions.RequestException as e:
            status = getattr(getattr(e, 'response', None), 'status_code', 'N/A')
            print(f"Error calling Hugging Face API for '{model_id}': {e} (status={status}). Trying fallback...")
            last_error = str(e)
            continue
        except Exception as e:
            print(f"An unexpected error occurred during LLM query for '{model_id}': {e}. Trying fallback...")
            last_error = str(e)
            continue

    if last_error:
        print(f"LLM request failed after fallbacks. Last error: {last_error}")

    if ENABLE_LOCAL_LLM:
        local = _try_local_llm(text_prompt)
        if local:
            return local

    return "I'm having trouble connecting to the language model right now."

# Local LLM helpers (Transformers pipeline)

def load_local_llm_pipeline():
    global _local_llm_pipe
    if _local_llm_pipe is not None:
        return _local_llm_pipe
    try:
        # Strict: local files only, CPU only
        tokenizer = AutoTokenizer.from_pretrained(LOCAL_LLM_MODEL_ID, local_files_only=True)
        model = AutoModelForCausalLM.from_pretrained(LOCAL_LLM_MODEL_ID, local_files_only=True)
        model.to("cpu")
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)
        _local_llm_pipe = pipe
        print(f"Local LLM loaded (CPU, offline): {LOCAL_LLM_MODEL_ID}")
        return pipe
    except Exception as e:
        print(f"Failed to load local LLM '{LOCAL_LLM_MODEL_ID}': {e}")
        return None


def _try_local_llm(text_prompt: str) -> str:
    pipe = load_local_llm_pipeline()
    if pipe is None:
        return ""
    try:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text_prompt},
        ]
        tokenizer = getattr(pipe, "tokenizer", None)
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            prompt = f"System: {SYSTEM_PROMPT}\nUser: {text_prompt}\nAssistant:"
        outputs = pipe(prompt, max_new_tokens=150, do_sample=False)
        out_text = outputs[0].get("generated_text", "")
        if prompt and out_text.startswith(prompt):
            out_text = out_text[len(prompt):]
        return out_text.strip()
    except Exception as e:
        print(f"Local LLM generation failed: {e}")
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
        for mid in models_to_try:
            payload = {
                "model": mid,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": text_prompt},
                ],
                "temperature": 0.7,
                "max_tokens": 150,
                "stream": False,
            }
            try:
                resp = requests.post(f"{ROUTER_API_BASE}/chat/completions", headers=base_headers, json=payload, timeout=60)
                resp.raise_for_status()
                data = resp.json()
            except requests.exceptions.ContentDecodingError:
                nozip = dict(base_headers)
                nozip["Accept-Encoding"] = "identity"
                resp = requests.post(f"{ROUTER_API_BASE}/chat/completions", headers=nozip, json=payload, timeout=60)
                resp.raise_for_status()
                data = resp.json()
            except Exception as e:
                status = getattr(getattr(e, 'response', None), 'status_code', 'N/A')
                print(f"Router API error for '{mid}' (status={status}): {e} (trying fallback)")
                continue

            choices = data.get("choices", [])
            if not choices:
                print(f"Router returned no choices for '{mid}'. Trying fallback...")
                continue
            msg = choices[0].get("message", {})
            content = msg.get("content", "").strip()
            if content:
                return content
        return ""
    except Exception as e:
        print(f"Router request failed: {e}")
        return ""

# --- UTILITIES ---

def ensure_dirs():
    VOICE_DB_PROCESSED_DIR.mkdir(exist_ok=True)
    VOICE_DB_EMBEDDINGS_DIR.mkdir(exist_ok=True)
    OPTIMIZED_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    TTS_OUTPUT_DIR.mkdir(exist_ok=True)
    TEMP_DIR.mkdir(exist_ok=True)

def noise_reduce(data, sr):
    if nr is None:
        return data
    try:
        return nr.reduce_noise(y=data, sr=sr)
    except Exception:
        return data

def record_audio(seconds, sr):
    print(f"Recording {seconds} seconds...")
    rec = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype='int16')
    sd.wait()
    return rec.squeeze()

def embed_audio_file(path, encoder):
    wav_data = preprocess_wav(path)
    wav_data = noise_reduce(wav_data, SAMPLE_RATE)
    return encoder.embed_utterance(wav_data)

def save_embedding(name, emb):
    np.save(VOICE_DB_EMBEDDINGS_DIR / f"{name}.npy", emb)

def list_embeddings():
    return list(VOICE_DB_EMBEDDINGS_DIR.glob("*.npy"))

def cosine_similarity(a, b):
    return float(np.dot(a, b))

def save_float_to_wav(float_audio, path, sr=SAMPLE_RATE):
    wav.write(path, sr, (np.clip(float_audio, -1.0, 1.0) * 32767).astype(np.int16))

def preprocess_and_noise_reduce(src_wav_path):
    wav_data = preprocess_wav(src_wav_path)
    reduced = noise_reduce(wav_data, SAMPLE_RATE)
    return reduced

# --- VOICE STORAGE (UPDATED) ---

def store_voice(encoder):
    print("\n[Store Voice]")
    mode = input("Choose input type: (live / file): ").strip().lower()
    speaker = input("Enter speaker name (lowercase, underscores allowed): ").strip().lower()
    if not speaker:
        print("Invalid name.")
        return

    if mode == "file":
        file_path = Path(input("Enter path to WAV or MP3 file: ").strip())
        if not file_path.exists():
            print("File not found.")
            return
        if file_path.suffix.lower() == ".mp3":
            if AudioSegment is None:
                print("pydub not installed; cannot convert mp3.")
                return
            converted = TEMP_DIR / f"{speaker}_converted.wav"
            print("Converting MP3 -> WAV...")
            AudioSegment.from_mp3(file_path).set_frame_rate(SAMPLE_RATE).set_channels(1).export(converted, format="wav")
            raw_path = converted
        else:
            raw_path = file_path
        target_raw = VOICE_DB_PROCESSED_DIR / f"{speaker}_raw.wav"
        if raw_path != target_raw:
            try:
                data_sr, data = wav.read(raw_path)
                if data_sr != SAMPLE_RATE:
                    duration = data.shape[0] / data_sr
                    new_len = int(duration * SAMPLE_RATE)
                    data = np.interp(np.linspace(0, len(data), new_len, endpoint=False),
                                     np.arange(len(data)), data).astype(np.int16)
                wav.write(target_raw, SAMPLE_RATE, data)
            except Exception:
                import shutil
                shutil.copy(raw_path, target_raw)
        raw_path = target_raw
    else:
        print("Press Enter to start recording...")
        input()
        audio = record_audio(RECORD_DURATION, SAMPLE_RATE)
        raw_path = VOICE_DB_PROCESSED_DIR / f"{speaker}_raw.wav"
        wav.write(raw_path, SAMPLE_RATE, audio.astype(np.int16))

    try:
        reduced = preprocess_and_noise_reduce(raw_path)
        processed_path = VOICE_DB_PROCESSED_DIR / f"{speaker}_processed.wav"
        save_float_to_wav(reduced, processed_path)
        emb = encoder.embed_utterance(reduced)
        save_embedding(speaker, emb)
        print(f"Stored raw: {raw_path}")
        print(f"Stored processed: {processed_path}")
        print(f"Stored embedding: {VOICE_DB_EMBEDDINGS_DIR / (speaker + '.npy')}")
    except Exception as e:
        print(f"Failed processing voice: {e}")

# --- SPEAKER IDENTIFICATION (UPDATED) ---

def identify_speaker(encoder):
    if not list_embeddings():
        print("Voice database empty. Add voices first.")
        return None, None, None, None
    print("\n[Live Test] Press Enter to start recording...")
    input()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") # MODIFIED
    raw_path = TEMP_DIR / f"live_raw_{timestamp}.wav"
    processed_path = TEMP_DIR / f"live_processed_{timestamp}.wav"

    audio = record_audio(RECORD_DURATION, SAMPLE_RATE)
    wav.write(raw_path, SAMPLE_RATE, audio.astype(np.int16))

    try:
        reduced = preprocess_and_noise_reduce(raw_path)
        save_float_to_wav(reduced, processed_path)
        test_emb = encoder.embed_utterance(reduced)
    except Exception as e:
        print(f"Processing failed: {e}")
        return None, None, raw_path, processed_path

    scores = []
    for p in list_embeddings():
        db_emb = np.load(p)
        sim = cosine_similarity(test_emb, db_emb)
        scores.append((p.stem, sim))
    if not scores:
        print("No embeddings to compare.")
        return None, None, raw_path, processed_path
    scores.sort(key=lambda x: x[1], reverse=True)
    best_name, best_score = scores[0]
    print(f"Best match: {best_name} (similarity {best_score:.3f})")
    if best_score >= SIMILARITY_THRESHOLD:
        print("Threshold passed.")
        return best_name, best_score, raw_path, processed_path
    print(f"Below threshold ({SIMILARITY_THRESHOLD}). Access denied.")
    return None, best_score, raw_path, processed_path

# --- TRANSLATION MODEL LOADING / QUANTIZATION (UPDATED) ---

def confirm_large_download(model_id, size_mb):
    if not ASK_BEFORE_DOWNLOAD:
        return ALLOW_TRANSLATION_AUTO_DOWNLOAD
    ans = input(f"\nTranslation model '{model_id}' (~{size_mb}MB) not found.\nDownload now? (y/N): ").strip().lower()
    return ans == "y"

def load_or_quantize_translation_model(target_lang):
    if target_lang == "en":
        return None, None
    model_id = TRANSLATION_MODEL_IDS.get(target_lang)
    if not model_id:
        print(f"[Translate] No model configured for '{target_lang}'.")
        return None, None
    tag = model_id.replace("/", "_")
    quant_path = OPTIMIZED_DIR / f"{tag}_int8.pt"

    if target_lang in _translation_cache:
        return _translation_cache[target_lang]

    def _load_fp32_local():
        try:
            return MarianMTModel.from_pretrained(model_id, cache_dir=str(OPTIMIZED_DIR), local_files_only=True)
        except Exception as e:
            print(f"[Translate] Local FP32 model not fully cached: {e}")
            return None

    if quant_path.exists():
        print(f"[Translate] Found quantized weights: {quant_path.name}")
        fp32_model = _load_fp32_local()
        if fp32_model is not None:
            fp32_model.eval()
            quant_model = torch.quantization.quantize_dynamic(fp32_model, {torch.nn.Linear}, dtype=torch.qint8)
            try:
                state = torch.load(quant_path, map_location="cpu")
                missing, unexpected = quant_model.load_state_dict(state, strict=False)
                if missing or unexpected:
                    print(f"[Translate] Warning: state dict mismatch (missing={len(missing)}, unexpected={len(unexpected)}). Using quantized structure anyway.")
                tokenizer = MarianTokenizer.from_pretrained(model_id, cache_dir=str(OPTIMIZED_DIR), local_files_only=True)
                quant_model.eval()
                _translation_cache[target_lang] = (quant_model, tokenizer)
                return quant_model, tokenizer
            except Exception as e:
                print(f"[Translate] Failed loading saved quantized weights: {e} (will rebuild if local cache present).")
        else:
            print("[Translate] Could not reconstruct FP32 model locally; translation unavailable.")

    # Strict: do not download any weights
    try:
        fp32_model = MarianMTModel.from_pretrained(model_id, cache_dir=str(OPTIMIZED_DIR), local_files_only=True)
        tokenizer = MarianTokenizer.from_pretrained(model_id, cache_dir=str(OPTIMIZED_DIR), local_files_only=True)
    except Exception:
        print("[Translate] Model/tokenizer not available locally. Skipping translation.")
        return None, None

    fp32_model.eval()
    quant_model = torch.quantization.quantize_dynamic(fp32_model, {torch.nn.Linear}, dtype=torch.qint8)
    del fp32_model
    if not quant_path.exists():
        try:
            torch.save(quant_model.state_dict(), quant_path)
            print(f"[Translate] Saved quantized weights -> {quant_path}")
        except Exception as e:
            print(f"[Translate] Could not save quantized weights: {e}")
    quant_model.eval()
    _translation_cache[target_lang] = (quant_model, tokenizer)
    return quant_model, tokenizer

def translate_text(text, target_lang):
    if target_lang == "en":
        return text
    model, tokenizer = load_or_quantize_translation_model(target_lang)
    if model is None or tokenizer is None:
        print("Translation unavailable. Using original English text.")
        return text
    try:
        inputs = tokenizer([text], return_tensors="pt")
        with torch.no_grad():
            generated = model.generate(**inputs, max_length=512)
        out = tokenizer.batch_decode(generated, skip_special_tokens=True)[0]
        return out
    except Exception as e:
        print(f"Translation failed: {e}. Using original English text.")
        return text

# --- STT USING EXISTING QUANTIZED WHISPER SMALL ---

try:
    import librosa
except ImportError:
    librosa = None

def debug_audio_info(arr, sr, label):
    try:
        dur = len(arr) / float(sr)
        rms = float(np.sqrt(np.mean(arr**2)))
        print(f"[DEBUG] {label}: sr={sr} len={len(arr)} dur={dur:.2f}s rms={rms:.4f}")
    except Exception:
        pass

def load_local_stt_pipeline():
    global _stt_pipe
    if _stt_pipe is not None:
        return _stt_pipe
    if not WHISPER_SMALL_WEIGHTS.exists():
        print(f"Quantized weights not found: {WHISPER_SMALL_WEIGHTS}")
        print("Run model.py first to create the quantized weights.")
        return None
    try:
        config = AutoConfig.from_pretrained(WHISPER_SMALL_MODEL_ID, cache_dir=str(MODELS_DIR), local_files_only=True)
        generation_config = GenerationConfig.from_pretrained(WHISPER_SMALL_MODEL_ID, cache_dir=str(MODELS_DIR), local_files_only=True)
        base_model = AutoModelForSpeechSeq2Seq.from_config(config)
        base_model.generation_config = generation_config
        print("Applying quantization structure to Whisper Small...")
        quant_model = torch.quantization.quantize_dynamic(
            base_model, {torch.nn.Linear}, dtype=torch.qint8
        )
        print(f"Loading quantized weights from {WHISPER_SMALL_WEIGHTS}...")
        state = torch.load(WHISPER_SMALL_WEIGHTS, map_location="cpu")
        quant_model.load_state_dict(state)
        quant_model.eval()
        print("Whisper Small (quantized) loaded successfully.")

        try:
            processor = AutoProcessor.from_pretrained(WHISPER_SMALL_MODEL_ID, cache_dir=str(MODELS_DIR), local_files_only=True)
        except Exception:
            print("Local processor files for Whisper Small not found. Please cache them locally first.")
            return None
        stt_pipe = pipeline(
            "automatic-speech-recognition",
            model=quant_model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            batch_size=1,
            return_timestamps=False,
            dtype="float32",
            device="cpu",
        )
        _stt_pipe = stt_pipe
        return stt_pipe
    except Exception as e:
        print(f"Failed to initialize STT pipeline: {e}")
        return None

def _load_audio_16k(path):
    if not Path(path).exists():
        print(f"[STT] File not found: {path}")
        return None
    try:
        if librosa:
            data, sr = librosa.load(path, sr=16000, mono=True)
        else:
            sr, data = wav.read(path)
            if data.dtype != np.float32:
                data = data.astype(np.float32) / (32768 if data.dtype == np.int16 else np.max(np.abs(data), initial=1))
            if sr != 16000:
                dur = len(data) / sr
                new_len = int(dur * 16000)
                data = np.interp(
                    np.linspace(0, len(data), new_len, endpoint=False),
                    np.arange(len(data)),
                    data
                ).astype(np.float32)
                sr = 16000
        return data, 16000
    except Exception as e:
        print(f"[STT] Failed loading audio {path}: {e}")
        return None

def transcribe_audio(audio_path):
    pipe = load_local_stt_pipeline()
    if pipe is None:
        return ""
    loaded = _load_audio_16k(audio_path)
    if loaded is None:
        return ""
    audio_array, sr = loaded
    debug_audio_info(audio_array, sr, "LoadedRaw")

    if nr is not None:
        try:
            audio_array = nr.reduce_noise(y=audio_array, sr=sr)
        except Exception:
            pass
    debug_audio_info(audio_array, sr, "PostNoiseReduce")

    try:
        result = pipe({"array": audio_array, "sampling_rate": sr}, generate_kwargs={"task": "translate"})
        text = result.get("text", "").strip()
        print(f"[STT] Used file: {audio_path}")
        return text
    except Exception as e:
        print(f"STT failed: {e}")
        return ""

# --- PIPER TTS ---
def ensure_piper_voice(lang, gender):
    info = PIPER_VOICES.get(lang, {}).get(gender)
    if not info:
        raise RuntimeError(f"No Piper mapping for {lang}/{gender}")
    name = info["name"]
    onnx_filename = f"{name}.onnx"
    json_filename = f"{name}.onnx.json"

    model_path_primary = MODELS_DIR / onnx_filename
    json_path_primary = MODELS_DIR / json_filename
    if model_path_primary.exists() and json_path_primary.exists():
        return model_path_primary, json_path_primary, name

    model_path_alt = OPTIMIZED_DIR / onnx_filename
    json_path_alt = OPTIMIZED_DIR / json_filename
    if model_path_alt.exists() and json_path_alt.exists():
        return model_path_alt, json_path_alt, name

    # Strict: Do not download voices. Instruct user to provide files locally.
    raise RuntimeError(
        f"Piper voice files missing for {lang}/{gender} ({name}). Place '{onnx_filename}' and '{json_filename}' in '{MODELS_DIR}' or '{OPTIMIZED_DIR}'."
    )

def _piper_available():
    from shutil import which
    return which("piper") is not None

def synthesize_piper(text, lang, gender):
    if not _piper_available():
        print("Piper CLI not found. pip install piper-tts")
        return
    try:
        onnx_path, _, name = ensure_piper_voice(lang, gender)
    except RuntimeError as e:
        print(e)
        return

    out_path = TTS_OUTPUT_DIR / f"tts_{lang}_{gender}.wav"
    safe_text = text.replace('"', "'")
    cmd = f'echo "{safe_text}" | piper --model "{onnx_path}" --output_file "{out_path}"'
    print(f"[Piper] Running -> {out_path} (model: {onnx_path})")
    try:
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True, check=False)
        if res.returncode != 0:
            print(f"[Piper] First attempt failed (exit {res.returncode}).")
            alt_dirs = [d for d in [OPTIMIZED_DIR, MODELS_DIR] if d != onnx_path.parent]
            for d in alt_dirs:
                alt_model = d / onnx_path.name
                alt_json = d / (onnx_path.stem + ".onnx.json")
                if alt_model.exists() and alt_json.exists():
                    print(f"[Piper] Retrying with alternate model copy: {alt_model}")
                    retry_cmd = f'echo "{safe_text}" | piper --model "{alt_model}" --output_file "{out_path}"'
                    res2 = subprocess.run(retry_cmd, shell=True, capture_output=True, text=True, check=False)
                    if res2.returncode == 0:
                        if res2.stderr.strip():
                            print(f"[Piper stderr] {res2.stderr.strip()}")
                        print(f"[Piper] Success on retry. Output: {out_path}")
                        return
                    else:
                        print(f"[Piper] Retry failed (exit {res2.returncode}). Stderr:\n{res2.stderr}")
            print(f"[Piper] Final failure. Stderr:\n{res.stderr}")
            return
        if res.stderr.strip():
            print(f"[Piper stderr] {res.stderr.strip()}")
        print(f"[Piper] Success. Output: {out_path}")
    except Exception as e:
        print(f"[Piper] Execution error: {e}")


# --- MAIN WORKFLOW (TEST FLOW) ---
def run_live_test(encoder):
    user, score, raw_path, processed_path = identify_speaker(encoder)
    if not user:
        return
    print(f"Cached raw audio: {raw_path}")
    print(f"Cached processed audio: {processed_path}")
    if not processed_path or not Path(processed_path).exists():
        print("Processed audio missing. Aborting STT.")
        return

    lang_map = {"english": "en", "en": "en", "hindi": "hi", "hi": "hi", "marathi": "mr", "mr": "mr"}
    target = input("Output language (english / hindi / marathi): ").strip().lower()
    target_lang = lang_map.get(target)
    if target_lang not in ("en", "hi", "mr"):
        print("Invalid language.")
        return
    gender_in = input("Voice gender (male / female): ").strip().lower()
    if gender_in not in ("male", "female"):
        print("Invalid gender.")
        return

    print("\n--- Running speech-to-text on processed audio...")
    recognized_text = transcribe_audio(processed_path)
    if not recognized_text:
        print("Could not recognize speech.")
        return
    print(f"STEP 1: Recognized Text (Input to LLM) -> '{recognized_text}'")

    # --- LLM PROCESSING STEP ---
    print("\n--- STEP 2: Sending text to LLM for processing...")
    llm_response_text = query_llm(recognized_text)
    print(f"STEP 2: LLM Response -> '{llm_response_text}'")
    # --- END OF STEP ---

    # --- MODIFIED: Use the LLM's response for translation and TTS ---
    text_to_speak = llm_response_text

    if target_lang == "en":
        final_text = text_to_speak
        print("\n--- STEP 3: No translation needed. Final text for TTS is from LLM.")
    else:
        print(f"\n--- STEP 3: Translating LLM response into '{target_lang}'...")
        final_text = translate_text(text_to_speak, target_lang)
        print(f"STEP 3: Translated Text -> '{final_text}'")

    print("\n--- STEP 4: Synthesizing speech with Piper...")
    synthesize_piper(final_text, target_lang, gender_in)
    # --- END OF MODIFICATIONS ---


# --- MAIN CLI ---
def main():
    ensure_dirs()
    if not load_api_key(): # --- NEW --- Load key at startup
        sys.exit(1)
    torch.set_num_threads(2)
    print("Loading voice encoder (CPU)...")
    encoder = VoiceEncoder(device="cpu")
    while True:
        cmd = input("\nEnter command (store | test | exit): ").strip().lower()
        if cmd == "store":
            store_voice(encoder)
        elif cmd == "test":
            run_live_test(encoder)
        elif cmd == "exit":
            print("Bye.")
            break
        else:
            print("Unknown command.")

if __name__ == "__main__":
    main()

