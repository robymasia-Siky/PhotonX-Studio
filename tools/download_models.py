#!/usr/bin/env python3
"""
PhotonX Studio — Model Downloader & Manager
Downloads official open-source AI stem demixing checkpoints (BS-RoFormer, BandIt-v2, DeepFilterNet).
"""

import sys
import os
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_REGISTRY = {
    "Kim_Vocal_2.onnx": {
        "url": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/Kim_Vocal_2.onnx",
        "description": "Kimberley Jensen Vocal SOTA MDX-Net ONNX [63.7 MB]",
        "expected_min_bytes": 60_000_000
    },
    "kuielab_a_drums.onnx": {
        "url": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/kuielab_a_drums.onnx",
        "description": "Kuielab Drums Ultra-Fast MDX-Net ONNX [28.3 MB]",
        "expected_min_bytes": 28_000_000
    },
    "kuielab_a_bass.onnx": {
        "url": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/kuielab_a_bass.onnx",
        "description": "Kuielab Bass Ultra-Fast MDX-Net ONNX [28.3 MB]",
        "expected_min_bytes": 28_000_000
    },
    "UVR-MDX-NET-Inst_HQ_4.onnx": {
        "url": "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/UVR-MDX-NET-Inst_HQ_4.onnx",
        "description": "UVR Instrumental HQ MDX-Net ONNX [56.3 MB]",
        "expected_min_bytes": 55_000_000
    },
    "bs_roformer_4stem.onnx": {
        "url": "https://huggingface.co/silverdaw/bs-roformer-rhythm-onnx/resolve/main/bs_roformer_4stem_rhythm_fp16.onnx",
        "description": "BS-RoFormer 4-Stem Band-Split RoFormer FP16 ONNX [256.9 MB]",
        "expected_min_bytes": 260_000_000
    }
}

def reporthook(blocknum, blocksize, totalsize):
    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = readsofar * 100 / totalsize
        mb_read = readsofar / (1024 * 1024)
        mb_total = totalsize / (1024 * 1024)
        sys.stdout.write(f"\r  Descargando: {percent:5.1f}% ({mb_read:6.1f} MB / {mb_total:6.1f} MB)")
        sys.stdout.flush()

def download_models():
    print("=" * 65)
    print(" PhotonX Studio — AI Model Downloader")
    print("=" * 65)

    for filename, meta in MODEL_REGISTRY.items():
        target = MODELS_DIR / filename
        if target.exists() and target.stat().st_size >= meta["expected_min_bytes"]:
            size_mb = target.stat().st_size / (1024 * 1024)
            print(f" [OK] {filename} già presente ({size_mb:.1f} MB)")
            continue

        print(f"\n Scaricamento di: {meta['description']}")
        print(f" Target: {target}")
        try:
            urllib.request.urlretrieve(meta["url"], str(target), reporthook)
            print("\n [OK] Download completato con successo!")
        except Exception as e:
            print(f"\n [ERRORE] Impossibile scaricare {filename}: {e}")

if __name__ == "__main__":
    download_models()
