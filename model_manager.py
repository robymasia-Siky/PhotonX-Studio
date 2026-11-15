#!/usr/bin/env python3
"""
PhotonX Studio — SOTA Model Manager & Registry
Handles discovery, hardware accelerator selection, and on-demand streaming downloads
of State-of-the-Art ONNX neural checkpoints from Hugging Face.
"""

import os
import sys
import urllib.request
from pathlib import Path
import onnxruntime as ort

APP_DIR = Path(__file__).parent.resolve()
LOCAL_MODELS_DIR = APP_DIR / "models"
SYSTEM_MODELS_DIR = Path(r"C:\ProgramData\OpenPhoton\models")

LOCAL_MODELS_DIR.mkdir(parents=True, exist_ok=True)
try:
    SYSTEM_MODELS_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass

MODEL_REGISTRY = {
    "bs_roformer": {
        "filename": "bs_roformer_4stem.onnx",
        "name": "BS-RoFormer 4-Stem (Band-Split SOTA #1)",
        "category": "Multi-Stem (RoFormer SOTA)",
        "description": "Band-Split RoFormer architecture for ultra-clean 4-stem isolation (Vocals, Drums, Bass, Other). Highest SDR benchmark in audio demixing.",
        "url": "https://huggingface.co/phuongnse/mel-roformer-kim-onnx/resolve/main/bs_roformer_4stem.onnx",
        "stems": ["Vocals", "Drums", "Bass", "Other"],
        "min_bytes": 200_000_000,
        "recommended": True
    },
    "mel_roformer": {
        "filename": "mel-roformer.fp16.onnx",
        "name": "Mel-Band RoFormer (Vocal SOTA #1)",
        "category": "Vocal Isolation (RoFormer SOTA)",
        "description": "World #1 ranked Mel-Band RoFormer for pristine vocal separation with zero acoustic bleed and natural timbre.",
        "url": "https://huggingface.co/phuongnse/mel-roformer-kim-onnx/resolve/main/mel-roformer.fp16.onnx",
        "stems": ["Vocals", "Instrumental"],
        "min_bytes": 400_000_000,
        "recommended": True
    },
    "htdemucs": {
        "filename": "htdemucs.onnx",
        "name": "Meta AI HTDemucs v4 (4-Stem SOTA)",
        "category": "Multi-Stem 4-Stems",
        "description": "Hybrid Transformer 4-Stem (Drums, Bass, Other, Vocals) with embedded STFT/iSTFT and zero phase cancellation.",
        "url": "https://huggingface.co/smank/htdemucs-onnx/resolve/main/htdemucs.onnx",
        "stems": ["Vocals", "Drums", "Bass", "Other"],
        "min_bytes": 300_000_000,
        "recommended": True
    },
    "htdemucs_6s": {
        "filename": "htdemucs_6s.onnx",
        "name": "Meta AI HTDemucs v4 (6-Stem SOTA)",
        "category": "Multi-Stem 6-Stems",
        "description": "Full 6-Stem isolation: Drums, Bass, Other, Vocals, Guitar, and Piano in a single coherent inference pass.",
        "url": "https://huggingface.co/smank/htdemucs-onnx/resolve/main/htdemucs_6s.onnx",
        "stems": ["Vocals", "Drums", "Bass", "Guitar", "Piano", "Other"],
        "min_bytes": 240_000_000,
        "recommended": True
    },
    "kim_vocal_2": {
        "filename": "Kim_Vocal_2.onnx",
        "name": "Kimberley Jensen Vocal SOTA (MDX-Net)",
        "category": "Vocal Isolation (2-Stem)",
        "description": "SOTA vocal extraction with SDR > 12.8 dB, ultra-clean separation, zero metallic bleed on instrumental.",
        "url": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/Kim_Vocal_2.onnx",
        "stems": ["Vocals", "Instrumental"],
        "min_bytes": 60_000_000,
        "recommended": True
    },
    "foxjoy_dereverb": {
        "filename": "Reverb_HQ_By_FoxJoy.onnx",
        "name": "FoxJoy De-Reverb HQ Studio",
        "category": "De-Reverb & Denoise",
        "description": "Studio-grade vocal dereverberation separating pure direct vocal from room acoustic reflection/reverb tail.",
        "url": "https://huggingface.co/raydotac/AudioSeparatorONNX/resolve/main/Reverb_HQ_By_FoxJoy.onnx",
        "stems": ["Dry Voice", "Reverb Tail"],
        "min_bytes": 60_000_000,
        "recommended": True
    }
}


def get_available_execution_providers() -> list[str]:
    """
    Returns available ONNX Runtime execution providers in priority order:
    1. CUDAExecutionProvider (NVIDIA Tensor Cores)
    2. DmlExecutionProvider (DirectML DirectX 12 GPU)
    3. CPUExecutionProvider (Multi-Threaded AVX)
    """
    all_providers = ort.get_available_providers()
    priority_order = ['CUDAExecutionProvider', 'DmlExecutionProvider', 'CPUExecutionProvider']
    available = [p for p in priority_order if p in all_providers]
    return available if available else ['CPUExecutionProvider']


def get_model_path(key_or_filename: str) -> Path | None:
    """Finds existing model in local models directory or system directory."""
    # Check env var
    clean_k = key_or_filename.upper().replace('.', '_').replace('-', '_')
    env_p = os.environ.get(f"OPENPHOTON_MODEL_{clean_k}")
    if env_p and Path(env_p).exists() and Path(env_p).stat().st_size > 50_000:
        return Path(env_p)

    # Determine filename
    filename = key_or_filename
    if key_or_filename in MODEL_REGISTRY:
        filename = MODEL_REGISTRY[key_or_filename]["filename"]

    candidates = [
        LOCAL_MODELS_DIR / filename,
        SYSTEM_MODELS_DIR / filename,
    ]

    for p in candidates:
        if p.exists() and p.stat().st_size > 50_000:
            return p

    return None


def is_model_installed(key: str) -> bool:
    """Checks whether the requested model is downloaded and valid."""
    p = get_model_path(key)
    if not p:
        return False
    min_bytes = MODEL_REGISTRY.get(key, {}).get("min_bytes", 50_000)
    return p.stat().st_size >= min_bytes


def download_model(key: str, progress_callback=None) -> Path:
    """
    Downloads model on-demand from Hugging Face into local models directory.
    Emits progress percentage (0-100) via progress_callback(pct, text).
    """
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model key: {key}")

    meta = MODEL_REGISTRY[key]
    filename = meta["filename"]
    target_path = LOCAL_MODELS_DIR / filename
    temp_path = LOCAL_MODELS_DIR / f"{filename}.download"

    if target_path.exists() and target_path.stat().st_size >= meta["min_bytes"]:
        if progress_callback:
            progress_callback(100, f"{meta['name']} già installato.")
        return target_path

    url = meta["url"]

    def reporthook(blocknum, blocksize, totalsize):
        readsofar = blocknum * blocksize
        if totalsize > 0:
            pct = min(100, int(readsofar * 100 / totalsize))
            mb_read = readsofar / (1024 * 1024)
            mb_total = totalsize / (1024 * 1024)
            if progress_callback:
                progress_callback(pct, f"Download {meta['name']}: {pct}% ({mb_read:.1f} MB / {mb_total:.1f} MB)")
        else:
            if progress_callback:
                progress_callback(50, f"Download {meta['name']} in corso...")

    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'PhotonX-Studio/1.0'})
        with urllib.request.urlopen(req) as resp, open(temp_path, 'wb') as out_f:
            total_size = int(resp.headers.get('Content-Length', 0))
            block_size = 1024 * 1024  # 1 MB blocks
            read_bytes = 0
            while True:
                chunk = resp.read(block_size)
                if not chunk:
                    break
                out_f.write(chunk)
                read_bytes += len(chunk)
                if total_size > 0 and progress_callback:
                    pct = min(100, int(read_bytes * 100 / total_size))
                    mb_read = read_bytes / (1024 * 1024)
                    mb_total = total_size / (1024 * 1024)
                    progress_callback(pct, f"Download {meta['name']}: {pct}% ({mb_read:.1f} MB / {mb_total:.1f} MB)")

        # Rename temp to target
        if temp_path.exists():
            if target_path.exists():
                target_path.unlink()
            temp_path.rename(target_path)

        # Mirror to system directory if possible
        try:
            sys_target = SYSTEM_MODELS_DIR / filename
            if not sys_target.exists():
                import shutil
                shutil.copyfile(target_path, sys_target)
        except Exception:
            pass

        if progress_callback:
            progress_callback(100, f"Download completato: {meta['name']}")

        return target_path

    except Exception as e:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass
        raise RuntimeError(f"Errore download {meta['name']}: {e}")
