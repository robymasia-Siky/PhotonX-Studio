#pragma once

#include "StemCache.hpp"
#include "OnnxEngine.hpp"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace OpenPhoton {

class ChunkScheduler {
public:
    explicit ChunkScheduler(StemCache& cache) : cache_(cache) {}
    ~ChunkScheduler() { stop(); }

    void start(const std::wstring& onnxModelPath) {
        stop();
        if (!onnxModelPath.empty()) {
            try {
                HardwareLimits limits;
                limits.maxGpuVramBytes = 3ULL * 1024 * 1024 * 1024; // Strict 3GB VRAM limit for 8GB GPU safety
                limits.cpuThreads = 4;
                engine_ = std::make_unique<OnnxEngine>(onnxModelPath, ExecutionBackend::Auto, 0, limits);
            } catch (...) {
                engine_.reset();
            }
        }
        running_.store(true, std::memory_order_release);
        worker_ = std::thread(&ChunkScheduler::workerLoop, this);
    }

    void stop() {
        running_.store(false, std::memory_order_release);
        cv_.notify_all();
        if (worker_.joinable()) worker_.join();
        engine_.reset();
    }

    void notifyPlayhead(int64_t samplePos) noexcept {
        playheadChunk_.store(samplePos >= 0 ? static_cast<size_t>(samplePos) / kChunkSamples : 0,
                             std::memory_order_relaxed);
        hasWork_.store(true, std::memory_order_release);
    }

    void wakeWorker() { cv_.notify_one(); }

private:
    size_t findBestChunkToProcess() const noexcept {
        const size_t current = playheadChunk_.load(std::memory_order_relaxed);
        size_t bestIdx = kMaxChunks;
        size_t bestDist = kMaxChunks + 1;

        for (size_t i = 0; i < kMaxChunks; ++i) {
            if (cache_.slot(i).state.load(std::memory_order_acquire) == ChunkState::Captured) {
                const size_t dist = (i >= current) ? (i - current) : (current - i + 10);
                if (dist < bestDist) {
                    bestDist = dist;
                    bestIdx = i;
                }
            }
        }
        return bestIdx;
    }

    void workerLoop() {
        std::vector<float> outBuf;
        std::vector<int64_t> outShape;

        while (running_.load(std::memory_order_acquire)) {
            const size_t idx = findBestChunkToProcess();
            if (idx >= kMaxChunks) {
                std::unique_lock<std::mutex> lk(mtx_);
                cv_.wait_for(lk, std::chrono::milliseconds(50), [&] {
                    return !running_.load() || hasWork_.exchange(false);
                });
                continue;
            }

            auto& s = cache_.slot(idx);
            s.state.store(ChunkState::Computing, std::memory_order_release);

            if (engine_) {
                try {
                    const std::vector<int64_t> inShape = {1, static_cast<int64_t>(kNumChannels),
                                                          static_cast<int64_t>(kChunkSamples)};
                    engine_->runAudioChunk(s.inputAudio.data(), inShape, outBuf, outShape);
                    if (outBuf.size() >= s.stemAudio.size()) {
                        std::copy_n(outBuf.data(), s.stemAudio.size(), s.stemAudio.data());
                    }
                } catch (...) {
                    copyFallbackBypass(s);
                }
            } else {
                copyFallbackBypass(s);
            }

            s.state.store(ChunkState::Ready, std::memory_order_release);
        }
    }

    static void copyFallbackBypass(AudioChunkSlot& s) {
        std::fill(s.stemAudio.begin(), s.stemAudio.end(), 0.0f);
        std::copy_n(s.inputAudio.data(), kNumChannels * kChunkSamples, s.stemAudio.data());
    }

    StemCache& cache_;
    std::unique_ptr<OnnxEngine> engine_;
    std::atomic<bool> running_{false};
    std::atomic<bool> hasWork_{false};
    std::atomic<size_t> playheadChunk_{0};
    std::thread worker_;
    std::mutex mtx_;
    std::condition_variable cv_;
};

} // namespace OpenPhoton
