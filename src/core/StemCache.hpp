#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <vector>

namespace OpenPhoton {

inline constexpr size_t kNumStems = 4;        // 0: Vocals, 1: Drums, 2: Bass, 3: Other
inline constexpr size_t kNumChannels = 2;     // Stereo L/R
inline constexpr size_t kSampleRateDemucs = 44100;
inline constexpr size_t kChunkSamples = 44100 * 4; // 4.0 seconds per chunk
inline constexpr size_t kMaxChunks = 300;          // Up to 20 minutes of continuous audio
inline constexpr size_t kCrossfadeSamples = 2048;  // Hann crossfade window

enum class ChunkState : uint8_t {
    Empty = 0,
    Captured = 1,
    Computing = 2,
    Ready = 3
};

struct StemMixParams {
    std::array<float, kNumStems> gains{1.0f, 1.0f, 1.0f, 1.0f}; // Vocals, Drums, Bass, Other
    std::array<bool, kNumStems> mutes{false, false, false, false};
    std::array<bool, kNumStems> solos{false, false, false, false};
    float dryWet = 1.0f;
};

struct AudioChunkSlot {
    std::atomic<ChunkState> state{ChunkState::Empty};
    std::vector<float> inputAudio; // [kNumChannels * kChunkSamples]
    std::vector<float> stemAudio;  // [kNumStems * kNumChannels * kChunkSamples]

    void allocate() {
        if (inputAudio.size() != kNumChannels * kChunkSamples) {
            inputAudio.assign(kNumChannels * kChunkSamples, 0.0f);
        }
        if (stemAudio.size() != kNumStems * kNumChannels * kChunkSamples) {
            stemAudio.assign(kNumStems * kNumChannels * kChunkSamples, 0.0f);
        }
        state.store(ChunkState::Empty, std::memory_order_release);
    }
};

class StemCache {
public:
    StemCache() {
        // Pre-allocate initial working chunks (~60 MB) for instant zero-latency processing
        for (size_t i = 0; i < 8 && i < kMaxChunks; ++i) {
            chunks_[i].allocate();
        }
    }

    void clearAll() {
        for (auto& c : chunks_) {
            c.state.store(ChunkState::Empty, std::memory_order_release);
            if (!c.inputAudio.empty()) std::fill(c.inputAudio.begin(), c.inputAudio.end(), 0.0f);
            if (!c.stemAudio.empty())  std::fill(c.stemAudio.begin(), c.stemAudio.end(), 0.0f);
        }
    }

    // Audio thread: writes capture sample lock-free
    void writeCaptureSample(int64_t timelineSample, float left, float right) noexcept {
        if (timelineSample < 0) return;
        const size_t chunkIdx = static_cast<size_t>(timelineSample) / kChunkSamples;
        if (chunkIdx >= kMaxChunks) return;
        const size_t localIdx = static_cast<size_t>(timelineSample) % kChunkSamples;

        auto& slot = chunks_[chunkIdx];
        if (slot.inputAudio.empty()) {
            slot.allocate();
        }
        slot.inputAudio[localIdx] = left;
        slot.inputAudio[kChunkSamples + localIdx] = right;

        if (slot.state.load(std::memory_order_acquire) == ChunkState::Empty) {
            slot.state.store(ChunkState::Captured, std::memory_order_release);
        }
    }

    // Audio thread: reads mixed separated stems with Hann window crossfade at boundaries
    bool readMixedSample(int64_t timelineSample, const StemMixParams& mix,
                         float& outLeft, float& outRight) const noexcept
    {
        if (timelineSample < 0) return false;
        const size_t chunkIdx = static_cast<size_t>(timelineSample) / kChunkSamples;
        if (chunkIdx >= kMaxChunks) return false;

        const auto& slot = chunks_[chunkIdx];
        const ChunkState st = slot.state.load(std::memory_order_acquire);
        if (slot.stemAudio.empty() || st != ChunkState::Ready) {
            return false;
        }

        const size_t localIdx = static_cast<size_t>(timelineSample) % kChunkSamples;

        bool anySolo = false;
        for (size_t s = 0; s < kNumStems; ++s) {
            if (mix.solos[s]) { anySolo = true; break; }
        }

        // Sum active stems for current chunk
        float curL = 0.0f, curR = 0.0f;
        for (size_t s = 0; s < kNumStems; ++s) {
            const bool active = anySolo ? mix.solos[s] : !mix.mutes[s];
            if (!active) continue;
            const float g = mix.gains[s];
            const size_t base = s * kNumChannels * kChunkSamples;
            curL += slot.stemAudio[base + localIdx] * g;
            curR += slot.stemAudio[base + kChunkSamples + localIdx] * g;
        }

        float mixedL = curL;
        float mixedR = curR;

        // Apply Hann window crossfading at chunk boundaries
        if (localIdx < kCrossfadeSamples && chunkIdx > 0) {
            const auto& prevSlot = chunks_[chunkIdx - 1];
            if (prevSlot.state.load(std::memory_order_acquire) == ChunkState::Ready && !prevSlot.stemAudio.empty()) {
                const size_t prevLocalIdx = kChunkSamples - kCrossfadeSamples + localIdx;
                float prevL = 0.0f, prevR = 0.0f;
                for (size_t s = 0; s < kNumStems; ++s) {
                    const bool active = anySolo ? mix.solos[s] : !mix.mutes[s];
                    if (!active) continue;
                    const float g = mix.gains[s];
                    const size_t base = s * kNumChannels * kChunkSamples;
                    prevL += prevSlot.stemAudio[base + prevLocalIdx] * g;
                    prevR += prevSlot.stemAudio[base + kChunkSamples + prevLocalIdx] * g;
                }

                // Hann crossfade factor: 0.0 -> 1.0
                const float frac = static_cast<float>(localIdx) / static_cast<float>(kCrossfadeSamples);
                const float w = 0.5f * (1.0f - std::cos(std::numbers::pi_v<float> * frac));
                mixedL = prevL * (1.0f - w) + curL * w;
                mixedR = prevR * (1.0f - w) + curR * w;
            }
        }

        const float dryL = slot.inputAudio[localIdx];
        const float dryR = slot.inputAudio[kChunkSamples + localIdx];
        outLeft  = dryL * (1.0f - mix.dryWet) + mixedL * mix.dryWet;
        outRight = dryR * (1.0f - mix.dryWet) + mixedR * mix.dryWet;
        return true;
    }

    AudioChunkSlot& slot(size_t idx) noexcept { return chunks_[idx]; }
    const AudioChunkSlot& slot(size_t idx) const noexcept { return chunks_[idx]; }

private:
    std::array<AudioChunkSlot, kMaxChunks> chunks_;
};

} // namespace OpenPhoton
