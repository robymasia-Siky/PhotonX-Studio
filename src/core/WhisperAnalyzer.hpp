#pragma once

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

struct whisper_context;

namespace OpenPhoton {

// Lock-free Single-Producer Single-Consumer (SPSC) Ring Buffer
template <typename T, size_t Capacity>
class LockFreeSPSC {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of two");
public:
    LockFreeSPSC() = default;

    bool push(const T& item) noexcept {
        const size_t currentTail = tail_.load(std::memory_order_relaxed);
        const size_t currentHead = head_.load(std::memory_order_acquire);
        if ((currentTail - currentHead) >= Capacity) {
            return false; // Full
        }
        buffer_[currentTail & (Capacity - 1)] = item;
        tail_.store(currentTail + 1, std::memory_order_release);
        return true;
    }

    size_t pushBulk(const T* items, size_t count) noexcept {
        const size_t currentTail = tail_.load(std::memory_order_relaxed);
        const size_t currentHead = head_.load(std::memory_order_acquire);
        const size_t available = Capacity - (currentTail - currentHead);
        const size_t toWrite = std::min(count, available);

        for (size_t i = 0; i < toWrite; ++i) {
            buffer_[(currentTail + i) & (Capacity - 1)] = items[i];
        }
        tail_.store(currentTail + toWrite, std::memory_order_release);
        return toWrite;
    }

    bool pop(T& item) noexcept {
        const size_t currentHead = head_.load(std::memory_order_relaxed);
        const size_t currentTail = tail_.load(std::memory_order_acquire);
        if (currentHead == currentTail) {
            return false; // Empty
        }
        item = buffer_[currentHead & (Capacity - 1)];
        head_.store(currentHead + 1, std::memory_order_release);
        return true;
    }

    size_t popBulk(T* items, size_t count) noexcept {
        const size_t currentHead = head_.load(std::memory_order_relaxed);
        const size_t currentTail = tail_.load(std::memory_order_acquire);
        const size_t available = currentTail - currentHead;
        const size_t toRead = std::min(count, available);

        for (size_t i = 0; i < toRead; ++i) {
            items[i] = buffer_[(currentHead + i) & (Capacity - 1)];
        }
        head_.store(currentHead + toRead, std::memory_order_release);
        return toRead;
    }

    size_t size() const noexcept {
        const size_t t = tail_.load(std::memory_order_relaxed);
        const size_t h = head_.load(std::memory_order_relaxed);
        return (t >= h) ? (t - h) : 0;
    }

    void clear() noexcept {
        head_.store(0, std::memory_order_release);
        tail_.store(0, std::memory_order_release);
    }

private:
    std::array<T, Capacity> buffer_{};
    alignas(64) std::atomic<size_t> tail_{0};
    alignas(64) std::atomic<size_t> head_{0};
};

struct AutoEqGains {
    float lowWarmthDb = 0.0f;  // ~150 Hz
    float boxinessDb  = 0.0f;  // ~500 Hz
    float presenceDb  = 0.0f;  // ~3.5 kHz
    float airDb       = 0.0f;  // ~10 kHz
};

class WhisperAnalyzer {
public:
    static constexpr size_t kRingBufferCapacity = 131072; // ~2.7s at 48kHz
    static constexpr int kTargetSampleRate = 16000;

    WhisperAnalyzer();
    ~WhisperAnalyzer();

    bool initialize(const std::string& modelPath, double sampleRate);
    void stop();

    // Real-time audio thread: lock-free, zero allocation
    void pushAudio(const float* left, const float* right, size_t numSamples) noexcept;

    // Real-time getters: lock-free atomic reads
    bool isVoiceActive() const noexcept {
        return voiceActive_.load(std::memory_order_relaxed);
    }

    float getBreathGain() const noexcept {
        return breathGain_.load(std::memory_order_relaxed);
    }

    float getSibilantIntensity() const noexcept {
        return sibilantIntensity_.load(std::memory_order_relaxed);
    }

    AutoEqGains getAutoEqGains() const noexcept {
        AutoEqGains g;
        g.lowWarmthDb = eqLow_.load(std::memory_order_relaxed);
        g.boxinessDb  = eqMid_.load(std::memory_order_relaxed);
        g.presenceDb  = eqPresence_.load(std::memory_order_relaxed);
        g.airDb       = eqAir_.load(std::memory_order_relaxed);
        return g;
    }

private:
    void workerLoop();
    void resampleAndAnalyze(const std::vector<float>& inputChunk);
    void computeSpectralProfile(const std::vector<float>& pcm16k, const std::vector<std::pair<int64_t, int64_t>>& voiceIntervals);

    whisper_context* ctx_ = nullptr;
    double sampleRate_ = 48000.0;
    std::string modelPath_;

    // SPSC Lock-free Queue for incoming mono mix
    LockFreeSPSC<float, kRingBufferCapacity> audioQueue_;

    std::thread workerThread_;
    std::atomic<bool> running_{false};
    std::mutex cvMutex_;
    std::condition_variable cv_;

    // Real-time atomic parameters read by DSP chain
    std::atomic<bool> voiceActive_{false};
    std::atomic<float> breathGain_{1.0f};          // 1.0f = bypass, ~0.25f (-12dB) during breath pause
    std::atomic<float> sibilantIntensity_{0.0f};   // 0.0 to 1.0 driving De-Esser lookahead
    std::atomic<float> eqLow_{0.0f};               // dB
    std::atomic<float> eqMid_{0.0f};               // dB
    std::atomic<float> eqPresence_{0.0f};          // dB
    std::atomic<float> eqAir_{0.0f};               // dB
};

} // namespace OpenPhoton
