"""
Test OpenPhoton Standalone Application initialization, processing engines, and real-time mixer.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pytest
import numpy as np

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

def test_app_window_creation(qapp):
    from openphoton_app import MainWindow
    window = MainWindow()
    assert window.windowTitle().startswith("PhotonX Studio") or window.windowTitle().startswith("Open-Photon")
    assert window.current_engine_name == "Music 4-Stems (Ultra)"
    assert len(window.channels) == 4

def test_realtime_audio_callback_mixing(qapp):
    from openphoton_app import MainWindow
    window = MainWindow()

    sr = 44100
    dur_samples = 44100
    dummy_audio = np.ones((dur_samples, 2), dtype=np.float32)
    window.loaded_audio = dummy_audio
    window.sample_rate = sr

    # Populate 4 separated stems: Vocals, Drums, Bass, Other
    stems = {
        "Vocals": np.full((dur_samples, 2), 1.0, dtype=np.float32),
        "Drums": np.full((dur_samples, 2), 0.5, dtype=np.float32),
        "Bass": np.full((dur_samples, 2), 0.2, dtype=np.float32),
        "Other": np.full((dur_samples, 2), 0.1, dtype=np.float32),
    }
    window.separated_stems = stems

    # 1. Default playback (all unmuted, gain = 1.0)
    outdata = np.zeros((1024, 2), dtype=np.float32)
    window._audio_callback(outdata, 1024, None, None)
    expected_sum = 1.0 + 0.5 + 0.2 + 0.1 # 1.8
    assert np.allclose(outdata[0, 0], expected_sum, atol=1e-3)

    # 2. Test MUTE: mute Vocals
    window.channels[0].set_mute(True) # Vocals muted
    window._audio_callback(outdata, 1024, None, None)
    expected_muted = 0.5 + 0.2 + 0.1 # 0.8
    assert np.allclose(outdata[0, 0], expected_muted, atol=1e-3)

    # 3. Test SOLO: solo Bass
    window.channels[2].set_solo(True) # Bass soloed
    window._audio_callback(outdata, 1024, None, None)
    expected_solo = 0.2 # only Bass
    assert np.allclose(outdata[0, 0], expected_solo, atol=1e-3)

    # 4. Test FADER GAIN on soloed track:
    window.channels[2].gain = 2.0
    window._audio_callback(outdata, 1024, None, None)
    assert np.allclose(outdata[0, 0], 0.4, atol=1e-3)

    # 5. Test Seek
    window._on_seek(0.5)
    assert window.playback_frame_pos == dur_samples // 2

    # Clean up
    window.close()

def test_all_engines_worker():
    from openphoton_app import DemixWorker, ENGINES
    sr = 44100
    t = np.linspace(0, 0.5, int(sr * 0.5), endpoint=False)
    test_audio = np.stack([np.sin(2 * np.pi * 440 * t), np.cos(2 * np.pi * 440 * t)], axis=-1).astype(np.float32)

    for engine_name, info in ENGINES.items():
        worker = DemixWorker(ort_session=None, audio=test_audio, sr=sr, engine_name=engine_name)
        # Verify run executes and emits finished_success
        result_stems = {}
        def on_done(names, stems_dict, colors, elapsed):
            result_stems.update(stems_dict)
        worker.finished_success.connect(on_done)
        worker.run()
        for stem in info["stems"]:
            assert stem in result_stems, f"Missing stem {stem} in engine {engine_name}"
            assert len(result_stems[stem]) == len(test_audio)

