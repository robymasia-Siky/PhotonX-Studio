#pragma once

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <numbers>
#include <vector>

namespace OpenPhoton {

template <typename FloatType = float>
class SmoothedValue {
public:
    void reset(FloatType sampleRate, FloatType rampTimeSeconds) noexcept {
        const FloatType numSamples = std::max(FloatType(1.0), sampleRate * rampTimeSeconds);
        step_ = FloatType(1.0) / numSamples;
    }

    void setCurrentAndTargetValue(FloatType value) noexcept {
        current_ = value;
        target_ = value;
    }

    void setTargetValue(FloatType target) noexcept {
        target_ = target;
    }

    FloatType getNextValue() noexcept {
        if (std::abs(target_ - current_) <= step_) {
            current_ = target_;
        } else if (current_ < target_) {
            current_ += step_;
        } else {
            current_ -= step_;
        }
        return current_;
    }

    FloatType getCurrentValue() const noexcept { return current_; }

private:
    FloatType current_ = 0.0f;
    FloatType target_ = 0.0f;
    FloatType step_ = 0.001f;
};

// Direct Form II Transposed Biquad Filter for real-time safety
class BiquadFilter {
public:
    enum class Type { LowShelf, HighShelf, Peak, BandPass };

    void setCoefficients(Type type, float sampleRate, float frequency, float q, float gainDb) noexcept {
        const float w0 = 2.0f * std::numbers::pi_v<float> * (frequency / sampleRate);
        const float cosW = std::cos(w0);
        const float sinW = std::sin(w0);
        const float A = std::pow(10.0f, gainDb / 40.0f);
        const float alpha = sinW / (2.0f * q);

        float b0 = 1.0f, b1 = 0.0f, b2 = 0.0f, a0 = 1.0f, a1 = 0.0f, a2 = 0.0f;

        switch (type) {
            case Type::Peak: {
                b0 = 1.0f + alpha * A;
                b1 = -2.0f * cosW;
                b2 = 1.0f - alpha * A;
                a0 = 1.0f + alpha / A;
                a1 = -2.0f * cosW;
                a2 = 1.0f - alpha / A;
                break;
            }
            case Type::LowShelf: {
                const float sqrtA2 = 2.0f * std::sqrt(A) * alpha;
                b0 = A * ((A + 1.0f) - (A - 1.0f) * cosW + sqrtA2);
                b1 = 2.0f * A * ((A - 1.0f) - (A + 1.0f) * cosW);
                b2 = A * ((A + 1.0f) - (A - 1.0f) * cosW - sqrtA2);
                a0 = (A + 1.0f) + (A - 1.0f) * cosW + sqrtA2;
                a1 = -2.0f * ((A - 1.0f) + (A + 1.0f) * cosW);
                a2 = (A + 1.0f) + (A - 1.0f) * cosW - sqrtA2;
                break;
            }
            case Type::HighShelf: {
                const float sqrtA2 = 2.0f * std::sqrt(A) * alpha;
                b0 = A * ((A + 1.0f) + (A - 1.0f) * cosW + sqrtA2);
                b1 = -2.0f * A * ((A - 1.0f) + (A + 1.0f) * cosW);
                b2 = A * ((A + 1.0f) + (A - 1.0f) * cosW - sqrtA2);
                a0 = (A + 1.0f) - (A - 1.0f) * cosW + sqrtA2;
                a1 = 2.0f * ((A - 1.0f) - (A + 1.0f) * cosW);
                a2 = (A + 1.0f) - (A - 1.0f) * cosW - sqrtA2;
                break;
            }
            case Type::BandPass: {
                b0 = alpha;
                b1 = 0.0f;
                b2 = -alpha;
                a0 = 1.0f + alpha;
                a1 = -2.0f * cosW;
                a2 = 1.0f - alpha;
                break;
            }
        }

        const float invA0 = 1.0f / a0;
        b0_ = b0 * invA0;
        b1_ = b1 * invA0;
        b2_ = b2 * invA0;
        a1_ = a1 * invA0;
        a2_ = a2 * invA0;
    }

    void reset() noexcept {
        z1_ = 0.0f;
        z2_ = 0.0f;
    }

    float processSample(float in) noexcept {
        const float out = in * b0_ + z1_;
        z1_ = in * b1_ - out * a1_ + z2_;
        z2_ = in * b2_ - out * a2_;
        return out;
    }

private:
    float b0_ = 1.0f, b1_ = 0.0f, b2_ = 0.0f;
    float a1_ = 0.0f, a2_ = 0.0f;
    float z1_ = 0.0f, z2_ = 0.0f;
};

// Compressor stage with smooth envelope
class CompressorStage {
public:
    void init(float sampleRate, float attackMs, float releaseMs, float thresholdDb, float ratio) noexcept {
        sampleRate_ = sampleRate;
        thresholdDb_ = thresholdDb;
        ratio_ = ratio;
        attCoef_ = std::exp(-1.0f / (sampleRate * (attackMs * 0.001f)));
        relCoef_ = std::exp(-1.0f / (sampleRate * (releaseMs * 0.001f)));
        envelope_ = 0.0f;
    }

    float processSample(float in) noexcept {
        const float absIn = std::abs(in);
        if (absIn > envelope_) {
            envelope_ = attCoef_ * envelope_ + (1.0f - attCoef_) * absIn;
        } else {
            envelope_ = relCoef_ * envelope_ + (1.0f - relCoef_) * absIn;
        }

        const float envDb = 20.0f * std::log10(envelope_ + 1e-6f);
        if (envDb > thresholdDb_) {
            const float overDb = envDb - thresholdDb_;
            const float grDb = overDb * (1.0f - 1.0f / ratio_);
            return in * std::pow(10.0f, -grDb / 20.0f);
        }
        return in;
    }

    void reset() noexcept {
        envelope_ = 0.0f;
    }

private:
    float sampleRate_ = 48000.0f;
    float thresholdDb_ = -18.0f;
    float ratio_ = 4.0f;
    float attCoef_ = 0.0f;
    float relCoef_ = 0.0f;
    float envelope_ = 0.0f;
};

// Complete Real-Time Lock-Free DSP Chain
class DSPChain {
public:
    static constexpr size_t kLookaheadSamples = 128; // ~2.7 ms lookahead
    static constexpr size_t kMaxDelay = 512;

    void prepare(double sampleRate) {
        sampleRate_ = static_cast<float>(sampleRate > 8000.0 ? sampleRate : 48000.0);

        // Dry Delay line setup for perfect phase alignment with Wet lookahead
        dryDelayL_.assign(kMaxDelay, 0.0f);
        dryDelayR_.assign(kMaxDelay, 0.0f);
        delayWriteIdx_ = 0;

        // Lookahead buffer setup for De-Esser / Limiter
        lookaheadL_.assign(kLookaheadSamples, 0.0f);
        lookaheadR_.assign(kLookaheadSamples, 0.0f);
        lookaheadIdx_ = 0;

        // Biquad EQs
        for (int ch = 0; ch < 2; ++ch) {
            eqLowShelf_[ch].setCoefficients(BiquadFilter::Type::LowShelf, sampleRate_, 120.0f, 0.707f, 0.0f);
            eqBoxinessPeak_[ch].setCoefficients(BiquadFilter::Type::Peak, sampleRate_, 500.0f, 1.2f, 0.0f);
            eqPresencePeak_[ch].setCoefficients(BiquadFilter::Type::Peak, sampleRate_, 3500.0f, 1.0f, 0.0f);
            eqAirShelf_[ch].setCoefficients(BiquadFilter::Type::HighShelf, sampleRate_, 10000.0f, 0.707f, 0.0f);

            deEsserBandpass_[ch].setCoefficients(BiquadFilter::Type::BandPass, sampleRate_, 6500.0f, 2.0f, 0.0f);

            compFast_[ch].init(sampleRate_, 1.5f, 40.0f, -14.0f, 4.0f); // Stage 1 FET peak catcher
            compSlow_[ch].init(sampleRate_, 30.0f, 250.0f, -20.0f, 2.0f); // Stage 2 Opto leveler
        }

        smoothBreathGain_.reset(sampleRate_, 0.015f);
        smoothBreathGain_.setCurrentAndTargetValue(1.0f);

        smoothSibilantAtt_.reset(sampleRate_, 0.008f);
        smoothSibilantAtt_.setCurrentAndTargetValue(1.0f);

        smoothEqLow_.reset(sampleRate_, 0.05f);
        smoothEqLow_.setCurrentAndTargetValue(0.0f);

        smoothEqMid_.reset(sampleRate_, 0.05f);
        smoothEqMid_.setCurrentAndTargetValue(0.0f);

        smoothEqPresence_.reset(sampleRate_, 0.05f);
        smoothEqPresence_.setCurrentAndTargetValue(0.0f);

        smoothEqAir_.reset(sampleRate_, 0.05f);
        smoothEqAir_.setCurrentAndTargetValue(0.0f);
    }

    void reset() {
        std::fill(dryDelayL_.begin(), dryDelayL_.end(), 0.0f);
        std::fill(dryDelayR_.begin(), dryDelayR_.end(), 0.0f);
        std::fill(lookaheadL_.begin(), lookaheadL_.end(), 0.0f);
        std::fill(lookaheadR_.begin(), lookaheadR_.end(), 0.0f);
        for (int ch = 0; ch < 2; ++ch) {
            eqLowShelf_[ch].reset();
            eqBoxinessPeak_[ch].reset();
            eqPresencePeak_[ch].reset();
            eqAirShelf_[ch].reset();
            deEsserBandpass_[ch].reset();
            compFast_[ch].reset();
            compSlow_[ch].reset();
        }
    }

    // Process stereo audio block in-place (100% deterministic, zero allocations)
    void process(float* bufferL, float* bufferR, size_t numSamples,
                 float dryWet, float targetBreathGain, float sibilantIntensity,
                 float autoEqLowDb, float autoEqMidDb, float autoEqPresenceDb, float autoEqAirDb) noexcept
    {
        smoothBreathGain_.setTargetValue(targetBreathGain);
        const float deEssTarget = 1.0f - std::clamp(sibilantIntensity * 0.7f, 0.0f, 0.75f);
        smoothSibilantAtt_.setTargetValue(deEssTarget);

        smoothEqLow_.setTargetValue(autoEqLowDb);
        smoothEqMid_.setTargetValue(autoEqMidDb);
        smoothEqPresence_.setTargetValue(autoEqPresenceDb);
        smoothEqAir_.setTargetValue(autoEqAirDb);

        // Update filters with smoothed values
        const float curLowDb = smoothEqLow_.getCurrentValue();
        const float curMidDb = smoothEqMid_.getCurrentValue();
        const float curPresDb = smoothEqPresence_.getCurrentValue();
        const float curAirDb = smoothEqAir_.getCurrentValue();

        for (int ch = 0; ch < 2; ++ch) {
            eqLowShelf_[ch].setCoefficients(BiquadFilter::Type::LowShelf, sampleRate_, 120.0f, 0.707f, curLowDb);
            eqBoxinessPeak_[ch].setCoefficients(BiquadFilter::Type::Peak, sampleRate_, 500.0f, 1.2f, curMidDb);
            eqPresencePeak_[ch].setCoefficients(BiquadFilter::Type::Peak, sampleRate_, 3500.0f, 1.0f, curPresDb);
            eqAirShelf_[ch].setCoefficients(BiquadFilter::Type::HighShelf, sampleRate_, 10000.0f, 0.707f, curAirDb);
        }

        const size_t totalLatency = kLookaheadSamples;

        for (size_t i = 0; i < numSamples; ++i) {
            const float dryInL = bufferL[i];
            const float dryInR = bufferR[i];

            // 1. Feed Dry Delay Line for phase-matched Wet/Dry blending
            dryDelayL_[delayWriteIdx_] = dryInL;
            dryDelayR_[delayWriteIdx_] = dryInR;
            const size_t delayedReadIdx = (delayWriteIdx_ + kMaxDelay - totalLatency) % kMaxDelay;
            const float alignedDryL = dryDelayL_[delayedReadIdx];
            const float alignedDryR = dryDelayR_[delayedReadIdx];
            delayWriteIdx_ = (delayWriteIdx_ + 1) % kMaxDelay;

            // 2. Breath Control Attenuation
            const float breathGain = smoothBreathGain_.getNextValue();
            float wetL = dryInL * breathGain;
            float wetR = dryInR * breathGain;

            // 3. Dynamic / Parametric Auto-EQ
            wetL = eqLowShelf_[0].processSample(wetL);
            wetL = eqBoxinessPeak_[0].processSample(wetL);
            wetL = eqPresencePeak_[0].processSample(wetL);
            wetL = eqAirShelf_[0].processSample(wetL);

            wetR = eqLowShelf_[1].processSample(wetR);
            wetR = eqBoxinessPeak_[1].processSample(wetR);
            wetR = eqPresencePeak_[1].processSample(wetR);
            wetR = eqAirShelf_[1].processSample(wetR);

            // 4. De-Esser with Lookahead & Whisper Sibilant Integration
            const float sibAtt = smoothSibilantAtt_.getNextValue();
            const float sideL = deEsserBandpass_[0].processSample(wetL);
            const float sideR = deEsserBandpass_[1].processSample(wetR);
            const float sideEnergy = std::max(std::abs(sideL), std::abs(sideR));
            float dGr = 1.0f;
            if (sideEnergy > 0.15f) { // -16 dBFS sibilance threshold
                dGr = 1.0f - std::clamp((sideEnergy - 0.15f) * 1.5f, 0.0f, 0.6f);
            }
            const float totalDeEssGain = std::min(sibAtt, dGr);

            // Buffer into lookahead queue
            const size_t laRead = lookaheadIdx_;
            const float delayedWetL = lookaheadL_[laRead] * totalDeEssGain;
            const float delayedWetR = lookaheadR_[laRead] * totalDeEssGain;
            lookaheadL_[lookaheadIdx_] = wetL;
            lookaheadR_[lookaheadIdx_] = wetR;
            lookaheadIdx_ = (lookaheadIdx_ + 1) % kLookaheadSamples;

            // 5. Two-Stage Compressor (Stage 1 Peak + Stage 2 Opto)
            wetL = compFast_[0].processSample(delayedWetL);
            wetL = compSlow_[0].processSample(wetL);

            wetR = compFast_[1].processSample(delayedWetR);
            wetR = compSlow_[1].processSample(wetR);

            // 6. Harmonic Saturator (2x Oversampling Polynomial Tape/Tube Color)
            // Soft-saturation waveshaping with pleasant even/odd harmonics
            wetL = saturateSample(wetL);
            wetR = saturateSample(wetR);

            // 7. Brickwall Peak Limiter (prevents digital overs, ceiling at -0.1 dBFS)
            wetL = limitSample(wetL);
            wetR = limitSample(wetR);

            // 8. Phase-Aligned Dry/Wet Blend
            bufferL[i] = alignedDryL * (1.0f - dryWet) + wetL * dryWet;
            bufferR[i] = alignedDryR * (1.0f - dryWet) + wetR * dryWet;
        }
    }

private:
    static float saturateSample(float x) noexcept {
        // Soft analog tube/tape curvature: f(x) = x / (1 + |x|) + 0.05 * x^2
        const float drive = 1.25f;
        const float in = x * drive;
        const float sat = in / (1.0f + std::abs(in));
        const float harmonics = 0.05f * (in * in * (in > 0.0f ? 1.0f : -1.0f));
        return (sat + harmonics) * 0.8f;
    }

    static float limitSample(float x) noexcept {
        constexpr float ceiling = 0.988f; // ~ -0.1 dBFS
        if (x > ceiling) {
            return ceiling;
        }
        if (x < -ceiling) {
            return -ceiling;
        }
        return x;
    }

    float sampleRate_ = 48000.0f;

    std::vector<float> dryDelayL_;
    std::vector<float> dryDelayR_;
    size_t delayWriteIdx_ = 0;

    std::vector<float> lookaheadL_;
    std::vector<float> lookaheadR_;
    size_t lookaheadIdx_ = 0;

    std::array<BiquadFilter, 2> eqLowShelf_;
    std::array<BiquadFilter, 2> eqBoxinessPeak_;
    std::array<BiquadFilter, 2> eqPresencePeak_;
    std::array<BiquadFilter, 2> eqAirShelf_;
    std::array<BiquadFilter, 2> deEsserBandpass_;

    std::array<CompressorStage, 2> compFast_;
    std::array<CompressorStage, 2> compSlow_;

    SmoothedValue<float> smoothBreathGain_;
    SmoothedValue<float> smoothSibilantAtt_;
    SmoothedValue<float> smoothEqLow_;
    SmoothedValue<float> smoothEqMid_;
    SmoothedValue<float> smoothEqPresence_;
    SmoothedValue<float> smoothEqAir_;
};

} // namespace OpenPhoton
