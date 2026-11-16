#!/usr/bin/env python3
"""
PhotonX Studio — GGML Model Downloader & Manager
Downloads official open-source GGML weights for Whisper and Demucs.
"""

import sys
import os
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_REGISTRY = {
    "ggml-base.bin": {
        "url": "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.bin",
        "description": "Whisper.cpp Base Model GGML [~141 MB]",
        "expected_min_bytes": 140_000_000
    },
    "ggml-model-htdemucs-f16.bin": {
        "url": "https://huggingface.co/datasets/Retrobear/demucs.cpp/resolve/main/ggml-model-htdemucs-4s-f16.bin",
        "description": "Demucs.cpp HTDemucs v4 4-Stem FP16 GGML [~80 MB]",
        "expected_min_bytes": 80_000_000
    }
}

def reporthook(blocknum, blocksize, totalsize):
    readsofar = blocknum * blocksize
    if totalsize > 0:
        percent = min(100.0, readsofar * 100.0 / totalsize)
        mb_read = readsofar / (1024 * 1024)
        mb_total = totalsize / (1024 * 1024)
        sys.stdout.write(f"\r  Download: {percent:5.1f}% ({mb_read:6.1f} MB / {mb_total:6.1f} MB)")
        sys.stdout.flush()

def download_models():
    print("=" * 65)
    print(" PhotonX Studio — AI GGML Model Downloader")
    print("=" * 65)

    for filename, meta in MODEL_REGISTRY.items():
        target = MODELS_DIR / filename
        if target.exists() and target.stat().st_size >= meta["expected_min_bytes"]:
            size_mb = target.stat().st_size / (1024 * 1024)
            print(f" [OK] {filename} gia' presente ({size_mb:.1f} MB)")
            continue

        print(f"\n Scaricamento di: {meta['description']}")
        print(f" Target: {target}")
        try:
            req = urllib.request.Request(
                meta["url"],
                headers={"User-Agent": "Mozilla/5.0"}
            )
            with urllib.request.urlopen(req) as response, open(target, 'wb') as out_file:
                total_size = int(response.info().get('Content-Length', -1))
                block_size = 1024 * 1024
                downloaded = 0
                while True:
                    buffer = response.read(block_size)
                    if not buffer:
                        break
                    downloaded += len(buffer)
                    out_file.write(buffer)
                    if total_size > 0:
                        pct = min(100.0, downloaded * 100.0 / total_size)
                        sys.stdout.write(f"\r  Download: {pct:5.1f}% ({downloaded / (1024*1024):6.1f} MB / {total_size / (1024*1024):6.1f} MB)")
                        sys.stdout.flush()
            print("\n [OK] Download completato con successo!")
        except Exception as e:
            print(f"\n [ERRORE] Impossibile scaricare {filename}: {e}")

if __name__ == "__main__":
    download_models()
