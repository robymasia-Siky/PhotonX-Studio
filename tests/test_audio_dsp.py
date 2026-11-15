"""
Open-Photon DSP Unit Tests
Tests:
1. Windowing (Hanning Window)
2. StreamingSTFT (Window: 1024, Hop: 512, Bins: 513)
3. StreamingISTFT (Overlap-Add Synthesis with exact normalization)
4. Perfect Reconstruction Property: RMSE < 1e-4 on complex audio signals
"""

import math
import numpy as np
import pytest

class STFTProcessorPy:
    def __init__(self, fft_size=1024, hop_size=512):
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.num_bins = fft_size // 2 + 1
        # Periodic Hann window matching STFTProcessor.hpp
        self.window = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(fft_size) / float(fft_size)))

    def forward(self, x):
        num_samples = len(x)
        if num_samples < self.fft_size:
            return np.empty((0, self.num_bins), dtype=np.complex64)
        out_frames = (num_samples - self.fft_size) // self.hop_size + 1
        spec = np.zeros((out_frames, self.num_bins), dtype=np.complex64)
        for f in range(out_frames):
            offset = f * self.hop_size
            chunk = x[offset : offset + self.fft_size] * self.window
            fft_res = np.fft.rfft(chunk, n=self.fft_size)
            spec[f, :] = fft_res
        return spec

    def inverse(self, spec, target_samples):
        out_frames = spec.shape[0]
        out_audio = np.zeros(target_samples, dtype=np.float32)
        win_norm = np.zeros(target_samples, dtype=np.float32)

        for f in range(out_frames):
            chunk = np.fft.irfft(spec[f, :], n=self.fft_size)
            offset = f * self.hop_size
            end = min(offset + self.fft_size, target_samples)
            chunk_len = end - offset
            out_audio[offset:end] += chunk[:chunk_len] * self.window[:chunk_len]
            win_norm[offset:end] += self.window[:chunk_len] ** 2

        nonzero = win_norm > 1e-6
        out_audio[nonzero] /= win_norm[nonzero]
        return out_audio


def test_stft_dimensions():
    stft = STFTProcessorPy(fft_size=1024, hop_size=512)
    assert stft.fft_size == 1024
    assert stft.hop_size == 512
    assert stft.num_bins == 513

    # Test 4 seconds @ 48kHz (192,000 samples)
    x = np.random.randn(192000).astype(np.float32)
    spec = stft.forward(x)
    expected_frames = (192000 - 1024) // 512 + 1 # 374 frames
    assert spec.shape == (expected_frames, 513)


def test_perfect_reconstruction_sine_multitone():
    stft = STFTProcessorPy(fft_size=1024, hop_size=512)
    sr = 48000
    duration = 2.0 # 2 seconds
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    
    # Complex multi-harmonic audio signal (Fundamental 220Hz + 440Hz + 880Hz + 1760Hz + Pink noise)
    signal = (0.4 * np.sin(2 * np.pi * 220 * t) +
              0.3 * np.sin(2 * np.pi * 440 * t) +
              0.2 * np.sin(2 * np.pi * 880 * t) +
              0.1 * np.sin(2 * np.pi * 1760 * t) +
              0.02 * np.random.randn(len(t))).astype(np.float32)

    spec = stft.forward(signal)
    reconstructed = stft.inverse(spec, len(signal))

    # Evaluate interior (avoiding initial/final window ramp)
    margin = 1024
    valid_orig = signal[margin:-margin]
    valid_recon = reconstructed[margin:-margin]

    max_abs_err = np.max(np.abs(valid_orig - valid_recon))
    rmse = np.sqrt(np.mean((valid_orig - valid_recon) ** 2))

    print(f"\n[STFT DSP] Max Abs Error: {max_abs_err:.2e}, RMSE: {rmse:.2e}")
    assert max_abs_err < 1e-4, f"Max error {max_abs_err} exceeded 1e-4 threshold"
    assert rmse < 1e-5, f"RMSE {rmse} exceeded 1e-5 threshold"


if __name__ == '__main__':
    pytest.main([__file__, "-v", "-s"])
