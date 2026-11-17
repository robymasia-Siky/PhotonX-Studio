#include "Processor.hpp"
#include "PluginIds.hpp"
#include "pluginterfaces/vst/ivstparameterchanges.h"
#include "pluginterfaces/vst/ivstprocesscontext.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#ifdef _WIN32
#include <windows.h>
#endif

namespace OpenPhoton {

using namespace Steinberg;
using namespace Steinberg::Vst;

namespace {

std::filesystem::path getPluginDir() {
#ifdef _WIN32
    HMODULE hMod = nullptr;
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCWSTR>(&getPluginDir),
                           &hMod)) {
        wchar_t pathBuf[MAX_PATH]{};
        GetModuleFileNameW(hMod, pathBuf, MAX_PATH);
        return std::filesystem::path(pathBuf).parent_path();
    }
#endif
    return std::filesystem::current_path();
}

std::string resolveModelPath(const std::string& filename) {
    const auto pluginDir = getPluginDir();

    // 1. Check VST3 Bundle Resources folder (self-contained)
    auto bundleResources = pluginDir.parent_path() / "Resources" / "models" / filename;
    if (std::filesystem::exists(bundleResources)) {
        return bundleResources.string();
    }
    auto bundleDirect = pluginDir.parent_path() / "Resources" / filename;
    if (std::filesystem::exists(bundleDirect)) {
        return bundleDirect.string();
    }

    // 2. Check plugin binary directory
    auto pluginModel = pluginDir / filename;
    if (std::filesystem::exists(pluginModel)) {
        return pluginModel.string();
    }
    auto pluginModelsSub = pluginDir / "models" / filename;
    if (std::filesystem::exists(pluginModelsSub)) {
        return pluginModelsSub.string();
    }

    // 3. Check ProgramData global cache
    auto pdata = std::filesystem::path("C:/ProgramData/OpenPhoton/models") / filename;
    if (std::filesystem::exists(pdata)) {
        return pdata.string();
    }

    // 4. Check Open-Photon workspace repository
    auto repoDev = std::filesystem::path("C:/Users/Utente/Desktop/Open-Photon/models") / filename;
    if (std::filesystem::exists(repoDev)) {
        return repoDev.string();
    }

    // 5. Check local current working directory
    auto localModels = std::filesystem::current_path() / "models" / filename;
    if (std::filesystem::exists(localModels)) {
        return localModels.string();
    }

    return (std::filesystem::current_path() / "models" / filename).string();
}

} // namespace

Processor::Processor() {
    setControllerClass(kControllerUID);
}

tresult PLUGIN_API Processor::initialize(FUnknown* context) {
    tresult res = AudioEffect::initialize(context);
    if (res != kResultOk) return res;

    addAudioInput(STR16("Stereo In"), SpeakerArr::kStereo);
    addAudioOutput(STR16("Stereo Out"), SpeakerArr::kStereo);
    return kResultOk;
}

tresult PLUGIN_API Processor::terminate() {
    scheduler_.stop();
    whisperAnalyzer_.stop();
    return AudioEffect::terminate();
}

tresult PLUGIN_API Processor::setBusArrangements(
    SpeakerArrangement* inputs, int32 numIns,
    SpeakerArrangement* outputs, int32 numOuts)
{
    if (numIns == 1 && numOuts == 1 &&
        inputs[0] == SpeakerArr::kStereo && outputs[0] == SpeakerArr::kStereo) {
        return AudioEffect::setBusArrangements(inputs, numIns, outputs, numOuts);
    }
    return kResultFalse;
}

tresult PLUGIN_API Processor::setupProcessing(ProcessSetup& newSetup) {
    sampleRate_ = newSetup.sampleRate > 8000.0 ? newSetup.sampleRate : 48000.0;
    dspChain_.prepare(sampleRate_);
    return AudioEffect::setupProcessing(newSetup);
}

tresult PLUGIN_API Processor::setActive(TBool state) {
    if (state) {
        const std::string demucsPath = resolveModelPath("ggml-model-htdemucs-f16.bin");
        const std::string whisperPath = resolveModelPath("ggml-base.bin");

        scheduler_.start(demucsPath);
        whisperAnalyzer_.initialize(whisperPath, sampleRate_);
        dspChain_.prepare(sampleRate_);
    } else {
        scheduler_.stop();
        whisperAnalyzer_.stop();
        dspChain_.reset();
    }
    return AudioEffect::setActive(state);
}

tresult PLUGIN_API Processor::process(ProcessData& data) {
    if (data.inputParameterChanges) {
        const int32 numParams = data.inputParameterChanges->getParameterCount();
        for (int32 i = 0; i < numParams; ++i) {
            auto* q = data.inputParameterChanges->getParameterData(i);
            if (!q) continue;
            ParamValue val = 0.0;
            int32 sampleOffset = 0;
            if (q->getPoint(q->getPointCount() - 1, sampleOffset, val) != kResultOk) continue;

            const ParamID id = q->getParameterId();
            if (id == kParamMode) mode_ = static_cast<int>(std::round(val * 2.0));
            else if (id == kParamDryWet) mix_.dryWet = static_cast<float>(val);
            else if (id >= kParamVocalsGain && id <= kParamOtherGain)
                mix_.gains[id - kParamVocalsGain] = static_cast<float>(val * 2.0);
            else if (id >= kParamVocalsMute && id <= kParamOtherMute)
                mix_.mutes[id - kParamVocalsMute] = (val >= 0.5);
            else if (id >= kParamVocalsSolo && id <= kParamOtherSolo)
                mix_.solos[id - kParamVocalsSolo] = (val >= 0.5);
            else if (id == kParamClearCache && val >= 0.5)
                cache_.clearAll();
        }
    }

    if (data.numInputs == 0 || data.numOutputs == 0 || data.numSamples <= 0)
        return kResultOk;

    if (!data.inputs[0].channelBuffers32 || !data.outputs[0].channelBuffers32)
        return kResultOk;

    float* inL  = data.inputs[0].channelBuffers32[0];
    float* inR  = (data.inputs[0].numChannels > 1 && data.inputs[0].channelBuffers32[1]) 
                  ? data.inputs[0].channelBuffers32[1] : inL;
    float* outL = data.outputs[0].channelBuffers32[0];
    float* outR = (data.outputs[0].numChannels > 1 && data.outputs[0].channelBuffers32[1]) 
                  ? data.outputs[0].channelBuffers32[1] : outL;

    if (!inL || !outL)
        return kResultOk;
    if (!inR) inR = inL;
    if (!outR) outR = outL;

    int64_t pos = fallbackCursor_;
    bool playing = true;
    if (data.processContext) {
        if (data.processContext->projectTimeSamples >= 0) {
            pos = data.processContext->projectTimeSamples;
        }
        playing = (data.processContext->state & ProcessContext::kPlaying) != 0;
    }

    // Lock-free real-time AI states
    const float breathGain = whisperAnalyzer_.getBreathGain();
    const float sibilantIntensity = whisperAnalyzer_.getSibilantIntensity();
    const auto autoEq = whisperAnalyzer_.getAutoEqGains();

    if (mode_ == 1 && playing) {
        // Mode 1: Capture timeline audio for Demucs background separation & Whisper analysis
        for (int32 s = 0; s < data.numSamples; ++s) {
            cache_.writeCaptureSample(pos + s, inL[s], inR[s]);
            outL[s] = inL[s];
            outR[s] = inR[s];
        }
        whisperAnalyzer_.pushAudio(inL, inR, static_cast<size_t>(data.numSamples));
        scheduler_.notifyPlayhead(pos);
        scheduler_.wakeWorker();
    } else if (mode_ == 2) {
        // Mode 2: Real-time Stem Separation playback + DSP Chain
        scheduler_.notifyPlayhead(pos);
        for (int32 s = 0; s < data.numSamples; ++s) {
            float mL = 0.0f, mR = 0.0f;
            if (cache_.readMixedSample(pos + s, mix_, mL, mR)) {
                outL[s] = mL;
                outR[s] = mR;
            } else {
                outL[s] = inL[s];
                outR[s] = inR[s];
            }
        }
        whisperAnalyzer_.pushAudio(outL, outR, static_cast<size_t>(data.numSamples));

        dspChain_.process(outL, outR, static_cast<size_t>(data.numSamples),
                          mix_.dryWet, breathGain, sibilantIntensity,
                          autoEq.lowWarmthDb, autoEq.boxinessDb, autoEq.presenceDb, autoEq.airDb);
    } else {
        // Mode 0: Direct Real-Time DSP Chain (zero-latency, lock-free)
        if (outL != inL) std::copy_n(inL, data.numSamples, outL);
        if (outR != inR) std::copy_n(inR, data.numSamples, outR);

        whisperAnalyzer_.pushAudio(outL, outR, static_cast<size_t>(data.numSamples));

        dspChain_.process(outL, outR, static_cast<size_t>(data.numSamples),
                          mix_.dryWet, breathGain, sibilantIntensity,
                          autoEq.lowWarmthDb, autoEq.boxinessDb, autoEq.presenceDb, autoEq.airDb);
    }

    fallbackCursor_ = pos + data.numSamples;
    return kResultOk;
}

} // namespace OpenPhoton
