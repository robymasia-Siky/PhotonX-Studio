#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <vector>

namespace OpenPhoton {

inline constexpr size_t kNumStems = 4;
inline constexpr size_t kNumChannels = 2;
inline constexpr size_t kChunkSamples = 48000 * 4; // 4 seconds per chunk
inline constexpr size_t kMaxChunks = 300;          // 20 minutes of audio (utilizes ~2.3 GB of 32 GB RAM)
inline constexpr size_t kCrossfadeSamples = 512;   // Smooth overlap-add boundary transition

enum class ChunkState : uint8_t {
    Empty = 0,
    Captured = 1,
    Computing = 2,
    Ready = 3
};

struct StemMixParams {
    std::array<float, kNumStems> gains{1.0f, 1.0f, 1.0f, 1.0f};
    std::array<bool, kNumStems> mutes{false, false, false, false};
    std::array<bool, kNumStems> solos{false, false, false, false};
    float dryWet = 1.0f;
};

struct AudioChunkSlot {
    std::atomic<ChunkState> state{ChunkState::Empty};
    std::vector<float> inputAudio;
    std::vector<float> stemAudio;

    void allocate() {
        inputAudio.assign(kNumChannels * kChunkSamples, 0.0f);
        stemAudio.assign(kNumStems * kNumChannels * kChunkSamples, 0.0f);
        state.store(ChunkState::Empty, std::memory_order_release);
    }
};

class StemCache {
public:
    StemCache() {
        for (auto& c : chunks_) c.allocate();
    }

    void clearAll() {
        for (auto& c : chunks_) {
            c.state.store(ChunkState::Empty, std::memory_order_release);
            std::fill(c.inputAudio.begin(), c.inputAudio.end(), 0.0f);
        }
    }

    void writeCaptureSample(int64_t timelineSample, float left, float right) noexcept {
        if (timelineSample < 0) return;
        const size_t chunkIdx = static_cast<size_t>(timelineSample) / kChunkSamples;
        if (chunkIdx >= kMaxChunks) return;
        const size_t localIdx = static_cast<size_t>(timelineSample) % kChunkSamples;

        auto& slot = chunks_[chunkIdx];
        slot.inputAudio[localIdx] = left;
        slot.inputAudio[kChunkSamples + localIdx] = right;
        if (slot.state.load(std::memory_order_acquire) == ChunkState::Empty) {
            slot.state.store(ChunkState::Captured, std::memory_order_release);
        }
    }

    bool readMixedSample(int64_t timelineSample, const StemMixParams& mix,
                         float& outLeft, float& outRight) const noexcept
    {
        if (timelineSample < 0) return false;
        const size_t chunkIdx = static_cast<size_t>(timelineSample) / kChunkSamples;
        if (chunkIdx >= kMaxChunks) return false;

        const auto& slot = chunks_[chunkIdx];
        if (slot.state.load(std::memory_order_acquire) != ChunkState::Ready) {
            return false;
        }

        const size_t localIdx = static_cast<size_t>(timelineSample) % kChunkSamples;
        bool anySolo = false;
        for (size_t s = 0; s < kNumStems; ++s) {
            if (mix.solos[s]) { anySolo = true; break; }
        }

        float sumL = 0.0f;
        float sumR = 0.0f;
        for (size_t s = 0; s < kNumStems; ++s) {
            const bool active = anySolo ? mix.solos[s] : !mix.mutes[s];
            if (!active) continue;
            const float g = mix.gains[s];
            const size_t base = s * kNumChannels * kChunkSamples;
            sumL += slot.stemAudio[base + localIdx] * g;
            sumR += slot.stemAudio[base + kChunkSamples + localIdx] * g;
        }

        const float dryL = slot.inputAudio[localIdx];
        const float dryR = slot.inputAudio[kChunkSamples + localIdx];
        outLeft  = dryL * (1.0f - mix.dryWet) + sumL * mix.dryWet;
        outRight = dryR * (1.0f - mix.dryWet) + sumR * mix.dryWet;
        return true;
    }

    AudioChunkSlot& slot(size_t idx) noexcept { return chunks_[idx]; }
    const AudioChunkSlot& slot(size_t idx) const noexcept { return chunks_[idx]; }

private:
    std::array<AudioChunkSlot, kMaxChunks> chunks_;
};

} // namespace OpenPhoton
