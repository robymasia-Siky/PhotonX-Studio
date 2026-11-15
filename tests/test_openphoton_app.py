"""
Test OpenPhoton Standalone Application initialization and processing engines.
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
