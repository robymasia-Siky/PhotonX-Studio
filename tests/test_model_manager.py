"""
Tests for PhotonX Model Manager and Hugging Face SOTA Registry.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
import pytest

from model_manager import (
    MODEL_REGISTRY,
    get_available_execution_providers,
    get_model_path,
    is_model_installed,
)

def test_model_registry_keys():
    assert "htdemucs" in MODEL_REGISTRY
    assert "htdemucs_6s" in MODEL_REGISTRY
    assert "kim_vocal_2" in MODEL_REGISTRY
    assert "foxjoy_dereverb" in MODEL_REGISTRY

    for key, meta in MODEL_REGISTRY.items():
        assert "filename" in meta
        assert "url" in meta
        assert "stems" in meta
        assert meta["url"].startswith("http")

def test_execution_providers_detection():
    providers = get_available_execution_providers()
    assert isinstance(providers, list)
    assert len(providers) > 0
    # CPU should always be at least fallback
    assert "CPUExecutionProvider" in providers

def test_existing_models_discovery():
    htdemucs_p = get_model_path("htdemucs")
    assert htdemucs_p is not None
    assert htdemucs_p.exists()
    assert is_model_installed("htdemucs") is True

    foxjoy_p = get_model_path("foxjoy_dereverb")
    assert foxjoy_p is not None
    assert foxjoy_p.exists()
    assert is_model_installed("foxjoy_dereverb") is True

    htdemucs_6s_p = get_model_path("htdemucs_6s")
    assert htdemucs_6s_p is not None
    assert htdemucs_6s_p.exists()
    assert is_model_installed("htdemucs_6s") is True
