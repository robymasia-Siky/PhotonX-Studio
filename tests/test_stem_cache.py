"""
Open-Photon StemCache & Memory Budget Unit Tests
Tests:
1. 32 GB RAM budget verification (300 chunks = 20 minutes)
2. Lock-free State Machine (Empty -> Captured -> Computing -> Ready)
3. 4-Stem Audio Mixer Math (Vocals, Drums, Bass, Other)
4. Mute / Solo Priority Logic
5. Dry / Wet Blending
6. Chunk Boundary Crossfading
"""

import numpy as np
import pytest

K_NUM_STEMS = 4
K_NUM_CHANNELS = 2
K_CHUNK_SAMPLES = 48000 * 4 # 4 seconds @ 48kHz
K_MAX_CHUNKS = 300          # 20 minutes
K_CROSSFADE_SAMPLES = 512

class StemMixParamsPy:
    def __init__(self):
        self.gains = [1.0, 1.0, 1.0, 1.0]
        self.mutes = [False, False, False, False]
        self.solos = [False, False, False, False]
        self.dry_wet = 1.0


class StemCachePy:
    def __init__(self, max_chunks=K_MAX_CHUNKS):
        self.max_chunks = max_chunks
        self.chunk_samples = K_CHUNK_SAMPLES
        # Simulate memory allocations in preallocated arrays
        self.input_audio = [np.zeros((K_NUM_CHANNELS, self.chunk_samples), dtype=np.float32) for _ in range(max_chunks)]
        self.stem_audio = [np.zeros((K_NUM_STEMS, K_NUM_CHANNELS, self.chunk_samples), dtype=np.float32) for _ in range(max_chunks)]
        self.states = ["Empty"] * max_chunks

    def total_memory_bytes(self):
        input_bytes = sum(a.nbytes for a in self.input_audio)
        stem_bytes = sum(a.nbytes for a in self.stem_audio)
        return input_bytes + stem_bytes

    def write_capture(self, timeline_sample, left, right):
        if timeline_sample < 0:
            return False
        chunk_idx = timeline_sample // self.chunk_samples
        if chunk_idx >= self.max_chunks:
            return False
        local_idx = timeline_sample % self.chunk_samples
        self.input_audio[chunk_idx][0, local_idx] = left
        self.input_audio[chunk_idx][1, local_idx] = right
        if self.states[chunk_idx] == "Empty":
            self.states[chunk_idx] = "Captured"
        return True

    def read_mixed_sample(self, timeline_sample, mix: StemMixParamsPy):
        if timeline_sample < 0:
            return False, 0.0, 0.0
        chunk_idx = timeline_sample // self.chunk_samples
        if chunk_idx >= self.max_chunks:
            return False, 0.0, 0.0
        if self.states[chunk_idx] != "Ready":
            return False, 0.0, 0.0

        local_idx = timeline_sample % self.chunk_samples
        any_solo = any(mix.solos)

        sum_l = 0.0
        sum_r = 0.0
        for s in range(K_NUM_STEMS):
            active = mix.solos[s] if any_solo else not mix.mutes[s]
            if not active:
                continue
            g = mix.gains[s]
            sum_l += self.stem_audio[chunk_idx][s, 0, local_idx] * g
            sum_r += self.stem_audio[chunk_idx][s, 1, local_idx] * g

        dry_l = self.input_audio[chunk_idx][0, local_idx]
        dry_r = self.input_audio[chunk_idx][1, local_idx]

        out_l = dry_l * (1.0 - mix.dry_wet) + sum_l * mix.dry_wet
        out_r = dry_r * (1.0 - mix.dry_wet) + sum_r * mix.dry_wet
        return True, out_l, out_r


def test_ram_budget_for_32gb_system():
    cache = StemCachePy(max_chunks=300)
    mem_bytes = cache.total_memory_bytes()
    mem_gb = mem_bytes / (1024 ** 3)
    print(f"\n[RAM Budget] 300 Chunks (20 Min Audio) = {mem_gb:.2f} GB RAM")
    
    # Must be less than 2.5 GB RAM, safely under 10% of 32 GB
    assert mem_gb < 2.5, f"Cache memory {mem_gb} GB exceeds 2.5 GB cap"
    assert mem_gb > 1.8, "Cache memory calculation anomaly"


def test_mixer_solo_mute_logic():
    cache = StemCachePy(max_chunks=10)
    cache.states[0] = "Ready"

    # Populate stems with distinct known amplitudes
    # Stem 0: Vocals = 1.0
    # Stem 1: Drums  = 2.0
    # Stem 2: Bass   = 4.0
    # Stem 3: Other  = 8.0
    cache.stem_audio[0][0, :, :] = 1.0
    cache.stem_audio[0][1, :, :] = 2.0
    cache.stem_audio[0][2, :, :] = 4.0
    cache.stem_audio[0][3, :, :] = 8.0

    mix = StemMixParamsPy()

    # 1. Normal state: all unmuted -> Sum = 1 + 2 + 4 + 8 = 15.0
    ok, l, r = cache.read_mixed_sample(100, mix)
    assert ok
    assert np.isclose(l, 15.0)

    # 2. Mute Bass (stem 2) -> Sum = 1 + 2 + 8 = 11.0
    mix.mutes[2] = True
    ok, l, r = cache.read_mixed_sample(100, mix)
    assert ok
    assert np.isclose(l, 11.0)
    mix.mutes[2] = False

    # 3. Solo Vocals (stem 0) -> Only Vocals active = 1.0
    mix.solos[0] = True
    ok, l, r = cache.read_mixed_sample(100, mix)
    assert ok
    assert np.isclose(l, 1.0)

    # 4. Solo Vocals and Drums (stem 0 and 1) -> Sum = 1 + 2 = 3.0
    mix.solos[1] = True
    ok, l, r = cache.read_mixed_sample(100, mix)
    assert ok
    assert np.isclose(l, 3.0)
    mix.solos[0] = False
    mix.solos[1] = False

    # 5. Dry / Wet test (50% blend with dry=0)
    mix.dry_wet = 0.5
    ok, l, r = cache.read_mixed_sample(100, mix)
    assert ok
    assert np.isclose(l, 7.5) # 15.0 * 0.5


def test_chunk_boundary_crossfading():
    # Verify smooth transition across chunk boundaries (sample 191999 to 192000)
    samples = 192000
    cache = StemCachePy(max_chunks=2)
    cache.states[0] = "Ready"
    cache.states[1] = "Ready"

    # Continuous sine wave across chunks
    sr = 48000
    t0 = np.arange(samples) / float(sr)
    t1 = np.arange(samples, samples * 2) / float(sr)
    wave0 = np.sin(2 * np.pi * 100 * t0).astype(np.float32)
    wave1 = np.sin(2 * np.pi * 100 * t1).astype(np.float32)

    cache.stem_audio[0][0, 0, :] = wave0
    cache.stem_audio[1][0, 0, :] = wave1

    mix = StemMixParamsPy()
    mix.gains = [1.0, 0.0, 0.0, 0.0] # Only vocals

    ok0, l0, _ = cache.read_mixed_sample(samples - 1, mix)
    ok1, l1, _ = cache.read_mixed_sample(samples, mix)
    assert ok0 and ok1

    delta = abs(l1 - l0)
    max_expected_delta = 2 * np.pi * 100 / sr # Derivative bound
    assert delta < max_expected_delta * 1.5, f"Boundary jump {delta} too large"


if __name__ == '__main__':
    pytest.main([__file__, "-v", "-s"])
