#pragma once

#include <algorithm>
#include <cmath>
#include <complex>
#include <numbers>
#include <vector>

namespace OpenPhoton {

class STFTProcessor {
public:
    STFTProcessor(size_t fftSize = 2048, size_t hopSize = 512)
        : fftSize_(fftSize), hopSize_(hopSize), window_(fftSize, 0.0f)
    {
        for (size_t i = 0; i < fftSize_; ++i) {
            window_[i] = 0.5f * (1.0f - std::cos(2.0f * std::numbers::pi_v<float> *
                         static_cast<float>(i) / static_cast<float>(fftSize_)));
        }
    }

    size_t fftSize() const noexcept { return fftSize_; }
    size_t hopSize() const noexcept { return hopSize_; }
    size_t numBins() const noexcept { return fftSize_ / 2 + 1; }

    void forward(const float* input, size_t numSamples,
                 std::vector<std::complex<float>>& outSpec, size_t& outFrames) const
    {
        if (numSamples < fftSize_) {
            outFrames = 0;
            outSpec.clear();
            return;
        }
        outFrames = (numSamples - fftSize_) / hopSize_ + 1;
        const size_t bins = numBins();
        outSpec.assign(outFrames * bins, {0.0f, 0.0f});

        std::vector<std::complex<float>> buf(fftSize_);
        for (size_t f = 0; f < outFrames; ++f) {
            const size_t offset = f * hopSize_;
            for (size_t i = 0; i < fftSize_; ++i) {
                buf[i] = { input[offset + i] * window_[i], 0.0f };
            }
            fft(buf, false);
            for (size_t b = 0; b < bins; ++b) {
                outSpec[f * bins + b] = buf[b];
            }
        }
    }

    void inverse(const std::vector<std::complex<float>>& spec, size_t numFrames,
                 size_t targetSamples, std::vector<float>& outAudio) const
    {
        outAudio.assign(targetSamples, 0.0f);
        std::vector<float> winNorm(targetSamples, 0.0f);
        std::vector<std::complex<float>> buf(fftSize_);
        const size_t bins = numBins();

        for (size_t f = 0; f < numFrames; ++f) {
            for (size_t b = 0; b < bins; ++b) {
                buf[b] = spec[f * bins + b];
            }
            for (size_t b = 1; b < bins - 1; ++b) {
                buf[fftSize_ - b] = std::conj(buf[b]);
            }
            fft(buf, true);

            const size_t offset = f * hopSize_;
            for (size_t i = 0; i < fftSize_ && (offset + i) < targetSamples; ++i) {
                outAudio[offset + i] += buf[i].real() * window_[i];
                winNorm[offset + i]  += window_[i] * window_[i];
            }
        }

        for (size_t i = 0; i < targetSamples; ++i) {
            if (winNorm[i] > 1e-6f) outAudio[i] /= winNorm[i];
        }
    }

private:
    static void fft(std::vector<std::complex<float>>& a, bool invert) {
        const size_t n = a.size();
        for (size_t i = 1, j = 0; i < n; ++i) {
            size_t bit = n >> 1;
            for (; j & bit; bit >>= 1) j ^= bit;
            j ^= bit;
            if (i < j) std::swap(a[i], a[j]);
        }
        for (size_t len = 2; len <= n; len <<= 1) {
            const float ang = 2.0f * std::numbers::pi_v<float> / static_cast<float>(len) * (invert ? 1.0f : -1.0f);
            const std::complex<float> wlen(std::cos(ang), std::sin(ang));
            for (size_t i = 0; i < n; i += len) {
                std::complex<float> w(1.0f, 0.0f);
                for (size_t j = 0; j < len / 2; ++j) {
                    const auto u = a[i + j];
                    const auto v = a[i + j + len / 2] * w;
                    a[i + j] = u + v;
                    a[i + j + len / 2] = u - v;
                    w *= wlen;
                }
            }
        }
        if (invert) {
            const float invN = 1.0f / static_cast<float>(n);
            for (auto& x : a) x *= invN;
        }
    }

    size_t fftSize_;
    size_t hopSize_;
    std::vector<float> window_;
};

} // namespace OpenPhoton
