"""
Test OpenPhoton Standalone Application initialization, processing engines, real-time mixer,
DSP FX chain (Denoiser, EQ, Saturator), and multi-format export worker.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Engine & UI structure
# ---------------------------------------------------------------------------

def test_engines_definition():
    from openphoton_app import ENGINES
    assert "Music 4-Stems (Ultra)" in ENGINES
    assert "Drums 5-Stems" in ENGINES
    assert "Cinema 4-Stems" in ENGINES
    assert "Voice Restore & Denoise" in ENGINES
    assert "Voice Separator (Multi-Speaker)" in ENGINES

    music_stems = ENGINES["Music 4-Stems (Ultra)"]["stems"]
    assert music_stems == ["Vocals", "Drums", "Bass", "Other"]

    drum_stems = ENGINES["Drums 5-Stems"]["stems"]
    assert drum_stems == ["Kick", "Snare", "Hi-Hat", "Toms", "Cymbals"]


def test_stem_model_options_coverage():
    from openphoton_app import STEM_MODEL_OPTIONS, ENGINES
    for engine_info in ENGINES.values():
        for stem in engine_info["stems"]:
            opts = STEM_MODEL_OPTIONS.get(stem, [])
            assert len(opts) >= 1, f"No model options for stem: {stem}"


def test_best_quality_preset_keys():
    from openphoton_app import BEST_QUALITY_PRESET
    assert "Vocals" in BEST_QUALITY_PRESET
    assert "Drums" in BEST_QUALITY_PRESET
    assert "Instrumental" in BEST_QUALITY_PRESET


def test_app_window_creation(qapp):
    from openphoton_app import MainWindow
    window = MainWindow()
    assert "PhotonX" in window.windowTitle()
    assert window.current_engine_name == "Music 4-Stems (Ultra)"
    assert len(window.channels) == 4
    window.close()


def test_stem_channel_has_model_combo(qapp):
    from openphoton_app import StemChannel
    from PyQt6.QtGui import QColor
    ch = StemChannel(0, "Vocals", QColor(76, 201, 240))
    assert ch.model_combo is not None
    assert ch.model_combo.count() >= 1
    assert ch.fx_btn is not None


def test_stem_channel_set_model(qapp):
    from openphoton_app import StemChannel
    from PyQt6.QtGui import QColor
    ch = StemChannel(0, "Vocals", QColor(76, 201, 240))
    # Should set without crash even if option not found
    ch.set_model("kim_vocal_2 (Kim Vocal 2 MDX)")
    result = ch.get_selected_model()
    # Either it matched or fell back gracefully
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Real-time audio callback
# ---------------------------------------------------------------------------

def test_realtime_audio_callback_mixing(qapp):
    from openphoton_app import MainWindow
    window = MainWindow()

    sr = 44100
    dur_samples = 44100
    dummy_audio = np.ones((dur_samples, 2), dtype=np.float32)
    window.loaded_audio = dummy_audio
    window.sample_rate = sr

    stems = {
        "Vocals": np.full((dur_samples, 2), 1.0, dtype=np.float32),
        "Drums":  np.full((dur_samples, 2), 0.5, dtype=np.float32),
        "Bass":   np.full((dur_samples, 2), 0.2, dtype=np.float32),
        "Other":  np.full((dur_samples, 2), 0.1, dtype=np.float32),
    }
    window.separated_stems = stems

    # 1. All unmuted, gain = 1.0
    outdata = np.zeros((1024, 2), dtype=np.float32)
    window._audio_callback(outdata, 1024, None, None)
    assert np.allclose(outdata[0, 0], 1.8, atol=1e-3)

    # 2. Mute Vocals → 0.8
    window.channels[0].set_mute(True)
    window._audio_callback(outdata, 1024, None, None)
    assert np.allclose(outdata[0, 0], 0.8, atol=1e-3)

    # 3. Solo Bass → 0.2
    window.channels[2].set_solo(True)
    window._audio_callback(outdata, 1024, None, None)
    assert np.allclose(outdata[0, 0], 0.2, atol=1e-3)

    # 4. Fader gain × 2 on Bass solo → 0.4
    window.channels[2].gain = 2.0
    window._audio_callback(outdata, 1024, None, None)
    assert np.allclose(outdata[0, 0], 0.4, atol=1e-3)

    # 5. Seek
    window._on_seek(0.5)
    assert window.playback_frame_pos == dur_samples // 2

    window.close()


# ---------------------------------------------------------------------------
# All engines demix worker (fallback / no model path)
# ---------------------------------------------------------------------------

def test_all_engines_worker():
    from openphoton_app import DemixWorker, ENGINES
    sr = 44100
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    test_audio = np.stack(
        [np.sin(2 * np.pi * 440 * t), np.cos(2 * np.pi * 440 * t)], axis=-1
    ).astype(np.float32)

    for engine_name, info in ENGINES.items():
        worker = DemixWorker(ort_session=None, audio=test_audio, sr=sr,
                             engine_name=engine_name)
        result_stems = {}

        def on_done(names, stems_dict, colors, elapsed):
            result_stems.update(stems_dict)

        worker.finished_success.connect(on_done)
        worker.run()
        for stem in info["stems"]:
            assert stem in result_stems, f"Missing stem '{stem}' in engine '{engine_name}'"
            assert len(result_stems[stem]) == len(test_audio)


# ---------------------------------------------------------------------------
# DSP: Denoiser
# ---------------------------------------------------------------------------

def test_apply_denoiser_shape():
    from openphoton_app import apply_denoiser
    sr = 44100
    audio = np.random.randn(sr, 2).astype(np.float32) * 0.1
    out = apply_denoiser(audio, threshold_db=-40.0, reduction_db=12.0, sr=sr)
    assert out.shape == audio.shape
    assert out.dtype == np.float32


def test_apply_denoiser_mono():
    from openphoton_app import apply_denoiser
    sr = 44100
    audio = np.random.randn(sr).astype(np.float32) * 0.1
    out = apply_denoiser(audio, threshold_db=-40.0, reduction_db=12.0, sr=sr)
    assert out.shape[0] == sr


def test_apply_denoiser_reduces_noise():
    from openphoton_app import apply_denoiser
    sr = 44100
    # Pure noise floor
    noise = np.random.randn(sr, 2).astype(np.float32) * 0.001
    out = apply_denoiser(noise, threshold_db=-20.0, reduction_db=40.0, sr=sr)
    # Output should be quieter than input
    assert np.mean(np.abs(out)) <= np.mean(np.abs(noise)) + 1e-4


# ---------------------------------------------------------------------------
# DSP: EQ
# ---------------------------------------------------------------------------

def test_apply_eq_shape():
    from openphoton_app import apply_eq
    sr = 44100
    audio = np.random.randn(sr, 2).astype(np.float32)
    bands = [
        {"type": "lowcut",    "freq": 80.0,   "gain_db": 0.0,   "q": 0.707},
        {"type": "lowshelf",  "freq": 250.0,  "gain_db": 3.0,   "q": 0.707},
        {"type": "peak",      "freq": 1000.0, "gain_db": -2.0,  "q": 1.0},
        {"type": "highshelf", "freq": 5000.0, "gain_db": 1.5,   "q": 0.707},
        {"type": "highcut",   "freq": 12000.0,"gain_db": 0.0,   "q": 0.707},
    ]
    out = apply_eq(audio, bands, sr=sr)
    assert out.shape == audio.shape
    assert out.dtype == np.float32


def test_apply_eq_zero_gain_passthrough():
    from openphoton_app import apply_eq
    sr = 44100
    audio = np.random.randn(sr, 2).astype(np.float32)
    bands = [{"type": "peak", "freq": 1000.0, "gain_db": 0.0, "q": 1.0}]
    out = apply_eq(audio, bands, sr=sr)
    # With zero gain, peak should be a no-op
    assert np.allclose(out, audio, atol=1e-3)


# ---------------------------------------------------------------------------
# DSP: Saturator
# ---------------------------------------------------------------------------

def test_apply_saturation_shape():
    from openphoton_app import apply_saturation
    audio = np.random.randn(44100, 2).astype(np.float32) * 0.5
    for char in ["Soft Clip", "Tape", "Tube"]:
        out = apply_saturation(audio, drive=0.3, character=char, mix=0.5)
        assert out.shape == audio.shape
        assert out.dtype == np.float32


def test_apply_saturation_no_clip():
    from openphoton_app import apply_saturation
    audio = np.random.randn(44100, 2).astype(np.float32) * 2.0  # over-driven input
    out = apply_saturation(audio, drive=1.0, character="Soft Clip", mix=1.0)
    # Output should not exceed 1.0 in absolute value
    assert np.max(np.abs(out)) <= 1.0 + 1e-6


def test_apply_saturation_dry_mix():
    from openphoton_app import apply_saturation
    audio = np.ones((1000, 2), dtype=np.float32) * 0.1
    out = apply_saturation(audio, drive=0.5, character="Soft Clip", mix=0.0)
    # mix=0 → pure dry signal
    assert np.allclose(out, audio, atol=1e-5)


# ---------------------------------------------------------------------------
# Export worker (dry-run, no real ONNX)
# ---------------------------------------------------------------------------

def test_export_worker_applies_fx(tmp_path):
    from openphoton_app import ExportWorker
    sr = 44100
    stems = {
        "Vocals": np.random.randn(sr, 2).astype(np.float32) * 0.3,
        "Drums":  np.random.randn(sr, 2).astype(np.float32) * 0.3,
    }
    processing_params = {
        "Vocals": {
            "denoiser": {"enabled": True, "threshold_db": -40.0, "reduction_db": 12.0},
            "eq": {"enabled": False, "bands": []},
            "sat": {"enabled": False, "drive": 0.1, "character": "Soft Clip", "mix": 0.3},
        },
        "Drums": {}
    }
    worker = ExportWorker(stems, str(tmp_path), "WAV", sr, [], processing_params)
    errors = []
    worker.error_occurred.connect(lambda e: errors.append(e))
    worker.run()
    assert not errors, f"Export error: {errors}"
    # Check files created
    assert (tmp_path / "Vocals.wav").exists()
    assert (tmp_path / "Drums.wav").exists()


def test_export_worker_aiff(tmp_path):
    from openphoton_app import ExportWorker
    sr = 44100
    stems = {"Bass": np.random.randn(sr, 2).astype(np.float32) * 0.2}
    worker = ExportWorker(stems, str(tmp_path), "AIFF", sr, [], {})
    errors = []
    worker.error_occurred.connect(lambda e: errors.append(e))
    worker.run()
    assert not errors
    assert (tmp_path / "Bass.aiff").exists()


def test_stem_fx_worker():
    from openphoton_app import StemFXWorker
    sr = 44100
    raw = np.random.randn(sr // 2, 2).astype(np.float32) * 0.1
    params = {
        "denoiser": {"enabled": False},
        "eq": {"enabled": True, "bands": [{"type": "peak", "freq": 1000.0, "gain_db": 3.0, "q": 1.0}]},
        "sat": {"enabled": True, "drive": 0.2, "character": "Tape", "mix": 0.5}
    }
    worker = StemFXWorker("Vocals", raw, params, sr)
    result = {}
    worker.finished_stem.connect(lambda name, audio: result.update({name: audio}))
    worker.run()
    assert "Vocals" in result
    assert result["Vocals"].shape == raw.shape

