#include "ChunkScheduler.hpp"
#include "model.hpp"
#include "tensor.hpp"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>
#include <algorithm>
#include <iostream>

namespace OpenPhoton {

class DemucsWorkerImpl {
public:
    demucscpp::demucs_model model{};
    bool loaded = false;

    bool loadModel(const std::string& path) {
        if (path.empty()) return false;
        loaded = demucscpp::load_demucs_model(path, &model);
        if (!loaded) {
            std::cerr << "[DemucsWorker] Failed to load model from " << path << std::endl;
        }
        return loaded;
    }

    void separate(const float* inL, const float* inR, size_t numSamples,
                  std::vector<float>& outStems)
    {
        if (!loaded || numSamples == 0) return;

        Eigen::MatrixXf audio(2, numSamples);
        for (size_t i = 0; i < numSamples; ++i) {
            audio(0, i) = inL[i];
            audio(1, i) = inR[i];
        }

        demucscpp::ProgressCallback noopCallback = [](float, const std::string&) {};
        Eigen::Tensor3dXf targets = demucscpp::demucs_inference(model, audio, noopCallback);

        // Demucs target mapping:
        // 0: drums  -> Stem 1
        // 1: bass   -> Stem 2
        // 2: other  -> Stem 3
        // 3: vocals -> Stem 0
        const int targetMap[4] = { 3, 0, 1, 2 };

        const size_t totalExpected = kNumStems * kNumChannels * numSamples;
        if (outStems.size() < totalExpected) {
            outStems.resize(totalExpected);
        }

        for (size_t s = 0; s < kNumStems; ++s) {
            const Eigen::DenseIndex targetIdx = static_cast<Eigen::DenseIndex>(targetMap[s]);
            const size_t stemBase = s * kNumChannels * numSamples;
            for (Eigen::DenseIndex i = 0; i < static_cast<Eigen::DenseIndex>(numSamples); ++i) {
                outStems[stemBase + static_cast<size_t>(i)] = targets(targetIdx, Eigen::DenseIndex(0), i);              // Left
                outStems[stemBase + numSamples + static_cast<size_t>(i)] = targets(targetIdx, Eigen::DenseIndex(1), i); // Right
            }
        }
    }
};

ChunkScheduler::ChunkScheduler(StemCache& cache)
    : cache_(cache), impl_(std::make_unique<DemucsWorkerImpl>())
{
}

ChunkScheduler::~ChunkScheduler() {
    stop();
}

bool ChunkScheduler::start(const std::string& modelPath) {
    stop();
    bool loaded = false;
    if (!modelPath.empty()) {
        loaded = impl_->loadModel(modelPath);
    }
    running_.store(true, std::memory_order_release);
    worker_ = std::thread(&ChunkScheduler::workerLoop, this);
    return loaded;
}

void ChunkScheduler::stop() {
    running_.store(false, std::memory_order_release);
    cv_.notify_all();
    if (worker_.joinable()) {
        worker_.join();
    }
}

void ChunkScheduler::notifyPlayhead(int64_t samplePos) noexcept {
    playheadChunk_.store(samplePos >= 0 ? static_cast<size_t>(samplePos) / kChunkSamples : 0,
                         std::memory_order_relaxed);
    hasWork_.store(true, std::memory_order_release);
}

void ChunkScheduler::wakeWorker() {
    cv_.notify_one();
}

bool ChunkScheduler::isModelLoaded() const noexcept {
    return impl_ && impl_->loaded;
}

size_t ChunkScheduler::findBestChunkToProcess() const noexcept {
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

void ChunkScheduler::workerLoop() {
    while (running_.load(std::memory_order_acquire)) {
        const size_t idx = findBestChunkToProcess();
        if (idx >= kMaxChunks) {
            std::unique_lock<std::mutex> lk(mtx_);
            cv_.wait_for(lk, std::chrono::milliseconds(50), [&] {
                return !running_.load(std::memory_order_relaxed) || hasWork_.exchange(false);
            });
            continue;
        }

        auto& s = cache_.slot(idx);
        s.state.store(ChunkState::Computing, std::memory_order_release);

        if (impl_ && impl_->loaded) {
            try {
                const float* inL = s.inputAudio.data();
                const float* inR = s.inputAudio.data() + kChunkSamples;
                impl_->separate(inL, inR, kChunkSamples, s.stemAudio);
            } catch (...) {
                // Fallback on error: copy dry input to all stems safely
                const size_t n = kChunkSamples;
                for (size_t st = 0; st < kNumStems; ++st) {
                    const size_t base = st * kNumChannels * n;
                    std::copy_n(s.inputAudio.data(), n, s.stemAudio.data() + base);
                    std::copy_n(s.inputAudio.data() + n, n, s.stemAudio.data() + base + n);
                }
            }
        } else {
            // Fallback bypass
            const size_t n = kChunkSamples;
            for (size_t st = 0; st < kNumStems; ++st) {
                const size_t base = st * kNumChannels * n;
                std::copy_n(s.inputAudio.data(), n, s.stemAudio.data() + base);
                std::copy_n(s.inputAudio.data() + n, n, s.stemAudio.data() + base + n);
            }
        }

        s.state.store(ChunkState::Ready, std::memory_order_release);
    }
}

} // namespace OpenPhoton
