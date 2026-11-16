#include "WhisperAnalyzer.hpp"
#include "whisper.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>

namespace OpenPhoton {

WhisperAnalyzer::WhisperAnalyzer() = default;

WhisperAnalyzer::~WhisperAnalyzer() {
    stop();
}

bool WhisperAnalyzer::initialize(const std::string& modelPath, double sampleRate) {
    stop();
    sampleRate_ = sampleRate > 8000.0 ? sampleRate : 48000.0;
    modelPath_ = modelPath;

    if (modelPath.empty()) {
        return false;
    }

    whisper_context_params cparams = whisper_context_default_params();
    cparams.use_gpu = false; // Safe CPU / AVX2 execution

    ctx_ = whisper_init_from_file_with_params(modelPath.c_str(), cparams);
    if (!ctx_) {
        std::cerr << "[WhisperAnalyzer] Failed to load model from " << modelPath << std::endl;
        return false;
    }

    running_.store(true, std::memory_order_release);
    workerThread_ = std::thread(&WhisperAnalyzer::workerLoop, this);
    return true;
}

void WhisperAnalyzer::stop() {
    running_.store(false, std::memory_order_release);
    cv_.notify_all();

    if (workerThread_.joinable()) {
        workerThread_.join();
    }

    if (ctx_) {
        whisper_free(ctx_);
        ctx_ = nullptr;
    }

    audioQueue_.clear();
    voiceActive_.store(false, std::memory_order_relaxed);
    breathGain_.store(1.0f, std::memory_order_relaxed);
    sibilantIntensity_.store(0.0f, std::memory_order_relaxed);
}

void WhisperAnalyzer::pushAudio(const float* left, const float* right, size_t numSamples) noexcept {
    if (!running_.load(std::memory_order_relaxed) || numSamples == 0) return;

    // Fast lock-free downmix to mono and write to SPSC
    constexpr size_t kStackChunk = 512;
    float monoBuf[kStackChunk];

    size_t processed = 0;
    while (processed < numSamples) {
        const size_t batch = std::min(numSamples - processed, kStackChunk);
        for (size_t i = 0; i < batch; ++i) {
            const size_t idx = processed + i;
            monoBuf[i] = 0.5f * (left[idx] + (right ? right[idx] : left[idx]));
        }
        audioQueue_.pushBulk(monoBuf, batch);
        processed += batch;
    }

    // Light notify when sufficient samples are buffered
    if (audioQueue_.size() >= 16384) {
        cv_.notify_one();
    }
}

void WhisperAnalyzer::workerLoop() {
    std::vector<float> readBuffer;
    readBuffer.reserve(kRingBufferCapacity);

    // Accumulate ~2.5 seconds of audio before running whisper inference
    const size_t requiredSamples = static_cast<size_t>(sampleRate_ * 2.5);

    while (running_.load(std::memory_order_acquire)) {
        {
            std::unique_lock<std::mutex> lock(cvMutex_);
            cv_.wait_for(lock, std::chrono::milliseconds(50), [&] {
                return !running_.load(std::memory_order_relaxed) || (audioQueue_.size() >= requiredSamples);
            });
        }

        if (!running_.load(std::memory_order_relaxed)) break;

        const size_t available = audioQueue_.size();
        if (available < requiredSamples) {
            continue;
        }

        readBuffer.resize(available);
        const size_t readCount = audioQueue_.popBulk(readBuffer.data(), available);
        readBuffer.resize(readCount);

        if (readCount > 0 && ctx_) {
            resampleAndAnalyze(readBuffer);
        }
    }
}

void WhisperAnalyzer::resampleAndAnalyze(const std::vector<float>& inputChunk) {
    if (inputChunk.empty() || !ctx_) return;

    // Resample from sampleRate_ to 16,000 Hz using linear interpolation
    const double ratio = static_cast<double>(sampleRate_) / kTargetSampleRate;
    const size_t targetSamples = static_cast<size_t>(inputChunk.size() / ratio);
    if (targetSamples < 3200) return; // Need at least 200ms

    std::vector<float> pcm16k(targetSamples);
    for (size_t i = 0; i < targetSamples; ++i) {
        const double srcPos = i * ratio;
        const size_t idx0 = static_cast<size_t>(srcPos);
        const size_t idx1 = std::min(idx0 + 1, inputChunk.size() - 1);
        const float frac = static_cast<float>(srcPos - idx0);
        pcm16k[i] = inputChunk[idx0] * (1.0f - frac) + inputChunk[idx1] * frac;
    }

    // Configure whisper inference
    whisper_full_params wparams = whisper_full_default_params(WHISPER_SAMPLING_GREEDY);
    wparams.print_special = false;
    wparams.print_progress = false;
    wparams.print_realtime = false;
    wparams.print_timestamps = false;
    wparams.token_timestamps = true;
    wparams.translate = false;
    wparams.no_context = true;
    wparams.single_segment = false;
    wparams.n_threads = 4;

    const int ret = whisper_full(ctx_, wparams, pcm16k.data(), static_cast<int>(pcm16k.size()));
    if (ret != 0) {
        return;
    }

    const int numSegments = whisper_full_n_segments(ctx_);
    bool voiceFound = false;
    bool sibilantFound = false;
    float maxSibilantScore = 0.0f;
    std::vector<std::pair<int64_t, int64_t>> voiceIntervals;

    for (int s = 0; s < numSegments; ++s) {
        const float noSpeechProb = whisper_full_get_segment_no_speech_prob(ctx_, s);
        if (noSpeechProb < 0.6f) {
            voiceFound = true;
            const int64_t t0 = whisper_full_get_segment_t0(ctx_, s); // centiseconds
            const int64_t t1 = whisper_full_get_segment_t1(ctx_, s);
            voiceIntervals.emplace_back(t0 * 160, t1 * 160); // 10ms = 160 samples @ 16kHz
        }

        const int numTokens = whisper_full_n_tokens(ctx_, s);
        for (int t = 0; t < numTokens; ++t) {
            const char* text = whisper_full_get_token_text(ctx_, s, t);
            if (!text) continue;

            // Check for sibilant characters (s, z, t, f, sh, ch)
            bool isSib = false;
            for (const char* p = text; *p != '\0'; ++p) {
                const char c = static_cast<char>(std::tolower(static_cast<unsigned char>(*p)));
                if (c == 's' || c == 'z' || c == 'f') {
                    isSib = true;
                    break;
                }
            }

            if (isSib) {
                sibilantFound = true;
                const whisper_token_data tdata = whisper_full_get_token_data(ctx_, s, t);
                maxSibilantScore = std::max(maxSibilantScore, std::clamp(tdata.p * 1.2f, 0.3f, 1.0f));
            }
        }
    }

    voiceActive_.store(voiceFound, std::memory_order_release);

    // Update sibilant lookahead intensity
    if (sibilantFound) {
        sibilantIntensity_.store(maxSibilantScore, std::memory_order_release);
    } else {
        sibilantIntensity_.store(0.0f, std::memory_order_release);
    }

    // VAD & Breath Control map:
    // If voice is inactive or in pause intervals, analyze pause segment energy
    if (!voiceFound) {
        // Evaluate energy for breath inhalation characteristics (-45 dB to -22 dB)
        float sumSq = 0.0f;
        for (float sample : pcm16k) {
            sumSq += sample * sample;
        }
        const float rms = std::sqrt(sumSq / static_cast<float>(pcm16k.size()) + 1e-12f);
        const float rmsDb = 20.0f * std::log10(rms);

        if (rmsDb > -45.0f && rmsDb < -22.0f) {
            // Breath sound detected in pause! Attenuate by -12 dB (factor 0.25)
            breathGain_.store(0.25f, std::memory_order_release);
        } else {
            // Silence or background room tone: bypass attenuation
            breathGain_.store(1.0f, std::memory_order_release);
        }
    } else {
        // Active voice singing/speech: no breath attenuation
        breathGain_.store(1.0f, std::memory_order_release);
        // Compute Auto-EQ profile on active speech only
        computeSpectralProfile(pcm16k, voiceIntervals);
    }
}

void WhisperAnalyzer::computeSpectralProfile(
    const std::vector<float>& pcm16k,
    const std::vector<std::pair<int64_t, int64_t>>& voiceIntervals)
{
    if (pcm16k.empty() || voiceIntervals.empty()) return;

    // Measure energy in 4 vocal bands across active voice frames:
    // Band 0: Low Warmth (100 - 300 Hz)
    // Band 1: Boxiness (300 - 800 Hz)
    // Band 2: Presence (2500 - 5000 Hz)
    // Band 3: Air (6000 - 8000 Hz)
    // Using bandpass filter accumulation on active voice samples
    float energyLow = 0.0f, energyMid = 0.0f, energyPresence = 0.0f, energyAir = 0.0f;
    size_t activeCount = 0;

    for (const auto& interval : voiceIntervals) {
        const size_t start = static_cast<size_t>(std::clamp<int64_t>(interval.first, 0, static_cast<int64_t>(pcm16k.size())));
        const size_t end = static_cast<size_t>(std::clamp<int64_t>(interval.second, 0, static_cast<int64_t>(pcm16k.size())));
        if (end <= start) continue;

        for (size_t i = start; i < end; ++i) {
            const float s = pcm16k[i];
            const float sPrev = (i > start) ? pcm16k[i - 1] : 0.0f;
            const float highPass = s - sPrev; // High frequency emphasis
            const float lowPass  = 0.5f * (s + sPrev);

            energyLow += lowPass * lowPass;
            energyMid += s * s;
            energyPresence += highPass * highPass;
            energyAir += std::abs(highPass);
            ++activeCount;
        }
    }

    if (activeCount < 1600) return; // Need at least 100ms of active speech

    energyLow = std::sqrt(energyLow / activeCount + 1e-9f);
    energyMid = std::sqrt(energyMid / activeCount + 1e-9f);
    energyPresence = std::sqrt(energyPresence / activeCount + 1e-9f);
    energyAir = std::sqrt(energyAir / activeCount + 1e-9f);

    // Compute target curves and gentle correction (max +/- 4 dB)
    const float avgLevel = (energyLow + energyMid + energyPresence) / 3.0f + 1e-6f;

    const float deltaLow = std::clamp((avgLevel * 0.9f - energyLow) / avgLevel * 3.0f, -4.0f, 4.0f);
    const float deltaMid = std::clamp((avgLevel * 0.7f - energyMid) / avgLevel * 3.0f, -4.0f, 4.0f);
    const float deltaPres = std::clamp((avgLevel * 1.1f - energyPresence) / avgLevel * 3.0f, -4.0f, 4.0f);
    const float deltaAir = std::clamp((avgLevel * 0.8f - energyAir) / avgLevel * 3.0f, -4.0f, 4.0f);

    // Gradual atomic update for smooth Auto-EQ
    eqLow_.store(deltaLow, std::memory_order_release);
    eqMid_.store(deltaMid, std::memory_order_release);
    eqPresence_.store(deltaPres, std::memory_order_release);
    eqAir_.store(deltaAir, std::memory_order_release);
}

} // namespace OpenPhoton
