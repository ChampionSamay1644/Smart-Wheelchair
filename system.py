import os
import sys
import math
import time
import urllib.request
import subprocess  # ADDED for Piper execution
from pathlib import Path
from datetime import datetime
try:
    from pydub import AudioSegment  # For mp3 -> wav (optional)
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

# Speaker embedding (required)
try:
    from resemblyzer import VoiceEncoder, preprocess_wav
    from resemblyzer.audio import sampling_rate as RS_SAMPLING_RATE
except ImportError:
    print("Missing dependency: resemblyzer. Install with: pip install resemblyzer")
    sys.exit(1)

# Translation + STT support
try:
    import torch
    from transformers import (
        MarianMTModel,
        MarianTokenizer,
        AutoConfig,
        AutoProcessor,
        AutoModelForSpeechSeq2Seq,
        pipeline,
        GenerationConfig,  # ADDED
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
VOICE_MODEL_SEARCH_DIRS = [MODELS_DIR, OPTIMIZED_DIR]  # ADDED: search order for Piper voices

SAMPLE_RATE = RS_SAMPLING_RATE
RECORD_DURATION = 5
SIMILARITY_THRESHOLD = 0.70

# Control translation model download on constrained hardware
ALLOW_TRANSLATION_AUTO_DOWNLOAD = False  # Set True if you want auto download without asking
ASK_BEFORE_DOWNLOAD = True               # Prompt user if download needed
ESTIMATED_MARIAN_SIZE_MB = 306           # Approx size for opus-mt en->hi/mr

# Piper voice mapping (extend as needed)
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
        # Placeholder; user must provide a valid Marathi Piper voice URLs if desired.
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

# Translation models (English -> target)
TRANSLATION_MODEL_IDS = {
    "hi": "Helsinki-NLP/opus-mt-en-hi",
    "mr": "Helsinki-NLP/opus-mt-en-mr",
    # "en" needs no translation
}

# Cache loaded translation artifacts
_translation_cache = {}
# Removed: _asr_pipe (Whisper Tiny). We now use local quantized Whisper Small.
_stt_pipe = None  # cached speech-to-text pipeline (quantized whisper-small)

WHISPER_SMALL_MODEL_ID = "openai/whisper-small"
WHISPER_SMALL_WEIGHTS = OPTIMIZED_DIR / "whisper_small_weights_cpu_int8.pt"

# --- UTILITIES ---

def ensure_dirs():
    VOICE_DB_PROCESSED_DIR.mkdir(exist_ok=True)
    VOICE_DB_EMBEDDINGS_DIR.mkdir(exist_ok=True)
    OPTIMIZED_DIR.mkdir(exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    TTS_OUTPUT_DIR.mkdir(exist_ok=True)
    TEMP_DIR.mkdir(exist_ok=True)  # ADDED

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
    """Helper: saves float32 (-1..1) array as int16 wav."""
    wav.write(path, sr, (np.clip(float_audio, -1.0, 1.0) * 32767).astype(np.int16))

def preprocess_and_noise_reduce(src_wav_path):
    """
    Loads (resample/mono) via preprocess_wav, applies noise reduction,
    returns reduced float waveform and path where processed wav is stored (caller decides filename).
    """
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
        # Copy raw into voice_db for traceability
        target_raw = VOICE_DB_PROCESSED_DIR / f"{speaker}_raw.wav"
        if raw_path != target_raw:
            try:
                data_sr, data = wav.read(raw_path)
                if data_sr != SAMPLE_RATE:
                    # simple resample via numpy (optional simple approach)
                    duration = data.shape[0] / data_sr
                    new_len = int(duration * SAMPLE_RATE)
                    data = np.interp(np.linspace(0, len(data), new_len, endpoint=False),
                                     np.arange(len(data)), data).astype(np.int16)
                wav.write(target_raw, SAMPLE_RATE, data)
            except Exception:
                # fallback copy
                import shutil
                shutil.copy(raw_path, target_raw)
        raw_path = target_raw
    else:
        print("Press Enter to start recording...")
        input()
        audio = record_audio(RECORD_DURATION, SAMPLE_RATE)
        raw_path = VOICE_DB_PROCESSED_DIR / f"{speaker}_raw.wav"
        wav.write(raw_path, SAMPLE_RATE, audio.astype(np.int16))

    # Process (noise reduce) and save processed file
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
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
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
    """
    Ask user to confirm a large model download (returns True if proceed).
    """
    if not ASK_BEFORE_DOWNLOAD:
        return ALLOW_TRANSLATION_AUTO_DOWNLOAD
    ans = input(f"\nTranslation model '{model_id}' (~{size_mb}MB) not found.\nDownload now? (y/N): ").strip().lower()
    return ans == "y"

def load_or_quantize_translation_model(target_lang):
    # REPLACED implementation with robust fallback (no MarianMTModel.from_config usage)
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

    # Helper: load base FP32 model locally (no download)
    def _load_fp32_local():
        try:
            return MarianMTModel.from_pretrained(model_id, cache_dir=str(OPTIMIZED_DIR), local_files_only=True)
        except Exception as e:
            print(f"[Translate] Local FP32 model not fully cached: {e}")
            return None

    # If quantized weights exist attempt fast restore
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
                tokenizer = MarianTokenizer.from_pretrained(model_id, cache_dir=str(OPTIMIZED_DIR))
                quant_model.eval()
                _translation_cache[target_lang] = (quant_model, tokenizer)
                return quant_model, tokenizer
            except Exception as e:
                print(f"[Translate] Failed loading saved quantized weights: {e} (will rebuild).")
        else:
            print("[Translate] Could not reconstruct FP32 model locally; may need download.")

    # Download / permission gate
    if (not quant_path.exists()) and (not ALLOW_TRANSLATION_AUTO_DOWNLOAD):
        if not confirm_large_download(model_id, ESTIMATED_MARIAN_SIZE_MB):
            print("[Translate] Skipping translation (user declined download).")
            return None, None

    print(f"[Translate] Building (and quantizing if first time) model '{model_id}'...")
    try:
        fp32_model = MarianMTModel.from_pretrained(model_id, cache_dir=str(OPTIMIZED_DIR))
        tokenizer = MarianTokenizer.from_pretrained(model_id, cache_dir=str(OPTIMIZED_DIR))
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
    except Exception as e:
        print(f"[Translate] Failed to prepare model '{model_id}': {e}")
        return None, None

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
    import librosa  # For robust loading/resampling
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
    """
    Load the already quantized Whisper Small model from ./optimized_models using
    the same procedure as model.py so that 'translate' task works.
    """
    global _stt_pipe
    if _stt_pipe is not None:
        return _stt_pipe
    if not WHISPER_SMALL_WEIGHTS.exists():
        print(f"Quantized weights not found: {WHISPER_SMALL_WEIGHTS}")
        print("Run model.py first to create the quantized weights.")
        return None
    try:
        # 1. Load config + generation config
        config = AutoConfig.from_pretrained(WHISPER_SMALL_MODEL_ID, cache_dir=str(MODELS_DIR))
        generation_config = GenerationConfig.from_pretrained(WHISPER_SMALL_MODEL_ID, cache_dir=str(MODELS_DIR))
        # 2. Build model from config and attach generation config
        base_model = AutoModelForSpeechSeq2Seq.from_config(config)
        base_model.generation_config = generation_config
        # 3. Apply dynamic quantization (structure must match quantized weights)
        print("Applying quantization structure to Whisper Small...")
        quant_model = torch.quantization.quantize_dynamic(
            base_model, {torch.nn.Linear}, dtype=torch.qint8
        )
        # 4. Load pre-quantized weights
        print(f"Loading quantized weights from {WHISPER_SMALL_WEIGHTS}...")
        state = torch.load(WHISPER_SMALL_WEIGHTS, map_location="cpu")
        quant_model.load_state_dict(state)
        quant_model.eval()
        print("Whisper Small (quantized) loaded successfully.")
        # 5. Load processor
        processor = AutoProcessor.from_pretrained(WHISPER_SMALL_MODEL_ID, cache_dir=str(MODELS_DIR))
        # 6. Build pipeline (mirror model.py params)
        stt_pipe = pipeline(
            "automatic-speech-recognition",
            model=quant_model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            max_new_tokens=128,
            # Removed chunk_length_s to avoid experimental segmentation issues
            batch_size=1,
            return_timestamps=False,
            torch_dtype=torch.float32,
            device="cpu",
        )
        _stt_pipe = stt_pipe
        return stt_pipe
    except Exception as e:
        print(f"Failed to initialize STT pipeline: {e}")
        return None

def _load_audio_16k(path):
    """
    Load audio at 16k mono float32 [-1,1].
    Prefer librosa if available; fallback to scipy + naive resample.
    """
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
                # Simple linear resample
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
    """
    Transcribe & translate-to-English using quantized Whisper Small.
    Always loads from disk (./temp processed file), re-noise-reduces, and feeds array to pipeline.
    """
    pipe = load_local_stt_pipeline()
    if pipe is None:
        return ""
    loaded = _load_audio_16k(audio_path)
    if loaded is None:
        return ""
    audio_array, sr = loaded
    debug_audio_info(audio_array, sr, "LoadedRaw")

    # Re-apply noise reduction (ensures every input pass goes through it)
    if nr is not None:
        try:
            audio_array = nr.reduce_noise(y=audio_array, sr=sr)
        except Exception:
            pass
    debug_audio_info(audio_array, sr, "PostNoiseReduce")

    # Whisper expects float32 array + sampling_rate
    try:
        result = pipe({"array": audio_array, "sampling_rate": sr}, generate_kwargs={"task": "translate"})
        text = result.get("text", "").strip()
        print(f"[STT] Used file: {audio_path}")
        return text
    except Exception as e:
        print(f"STT failed: {e}")
        return ""

# --- PIPER TTS (FIXED & MOVED BEFORE run_live_test) ---
def ensure_piper_voice(lang, gender):
    """
    Ensure Piper voice exists (prefer ./models like pipertts.py; fallback to ./optimized_models).
    """
    info = PIPER_VOICES.get(lang, {}).get(gender)
    if not info:
        raise RuntimeError(f"No Piper mapping for {lang}/{gender}")
    name = info["name"]
    onnx_filename = f"{name}.onnx"
    json_filename = f"{name}.onnx.json"

    # Prefer MODELS_DIR (stable) then OPTIMIZED_DIR
    model_path_primary = MODELS_DIR / onnx_filename
    json_path_primary = MODELS_DIR / json_filename
    if model_path_primary.exists() and json_path_primary.exists():
        return model_path_primary, json_path_primary, name

    # Fallback search (optimized)
    model_path_alt = OPTIMIZED_DIR / onnx_filename
    json_path_alt = OPTIMIZED_DIR / json_filename
    if model_path_alt.exists() and json_path_alt.exists():
        return model_path_alt, json_path_alt, name

    # Need download
    if not info.get("onnx_url") or not info.get("json_url"):
        raise RuntimeError(f"Piper voice URLs missing for {lang}/{gender} ({name}).")

    MODELS_DIR.mkdir(exist_ok=True)
    print(f"[Piper] Downloading voice '{name}' into {MODELS_DIR}")
    try:
        urllib.request.urlretrieve(info["onnx_url"], model_path_primary)
        urllib.request.urlretrieve(info["json_url"], json_path_primary)
    except Exception as e:
        raise RuntimeError(f"Piper download failed: {e}")
    return model_path_primary, json_path_primary, name

def _piper_available():  # ADDED
    from shutil import which
    return which("piper") is not None

def synthesize_piper(text, lang, gender):
    """
    Piper invocation with fallback: if first attempt fails (exit 1) and an alternate
    copy exists in the other directory, retry once.
    """
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
        res = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"[Piper] First attempt failed (exit {res.returncode}).")
            # Fallback attempt: try alternate directory if available
            alt_dirs = [d for d in [OPTIMIZED_DIR, MODELS_DIR] if d != onnx_path.parent]
            for d in alt_dirs:
                alt_model = d / onnx_path.name
                alt_json = d / (onnx_path.stem + ".onnx.json")
                if alt_model.exists() and alt_json.exists():
                    print(f"[Piper] Retrying with alternate model copy: {alt_model}")
                    retry_cmd = f'echo "{safe_text}" | piper --model "{alt_model}" --output_file "{out_path}"'
                    res2 = subprocess.run(retry_cmd, shell=True, capture_output=True, text=True)
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

    print("Running speech-to-text on processed audio (exact file in ./temp)...")
    recognized_text = transcribe_audio(processed_path)
    if not recognized_text:
        print("Could not recognize speech.")
        return
    print(f"Recognized (English): {recognized_text}")

    if target_lang == "en":
        final_text = recognized_text
        print("No translation needed.")
    else:
        print(f"Translating English -> {target_lang} ...")
        final_text = translate_text(recognized_text, target_lang)
        print(f"Translated Text: {final_text}")

    print("Synthesizing speech with Piper...")
    synthesize_piper(final_text, target_lang, gender_in)

# --- MAIN CLI ---

def main():
    ensure_dirs()
    torch.set_num_threads(2)  # Constrain CPU threads for Pi
    print("Loading voice encoder (CPU)...")
    encoder = VoiceEncoder()
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
