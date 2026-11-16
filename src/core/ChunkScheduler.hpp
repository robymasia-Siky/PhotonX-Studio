#pragma once

#include "StemCache.hpp"
#include <atomic>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

namespace OpenPhoton {

class DemucsWorkerImpl;

class ChunkScheduler {
public:
    explicit ChunkScheduler(StemCache& cache);
    ~ChunkScheduler();

    bool start(const std::string& modelPath);
    void stop();

    void notifyPlayhead(int64_t samplePos) noexcept;
    void wakeWorker();

    bool isModelLoaded() const noexcept;

private:
    StemCache& cache_;
    std::unique_ptr<DemucsWorkerImpl> impl_;
    std::atomic<bool> running_{false};
    std::atomic<size_t> playheadChunk_{0};
    std::atomic<bool> hasWork_{false};
    std::mutex mtx_;
    std::condition_variable cv_;
    std::thread worker_;

    size_t findBestChunkToProcess() const noexcept;
    void workerLoop();
};

} // namespace OpenPhoton
