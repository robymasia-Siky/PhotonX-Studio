#include "Processor.hpp"
#include "PluginIds.hpp"
#include "pluginterfaces/vst/ivstparameterchanges.h"
#include "pluginterfaces/vst/ivstprocesscontext.h"
#include <algorithm>
#include <cmath>
#include <cstdlib>
#ifdef _WIN32
#include <windows.h>
#include <delayimp.h>
#include <filesystem>

static FARPROC WINAPI onnxDelayHook(unsigned dliNotify, PDelayLoadInfo pdli) {
    if (dliNotify == dliNotePreLoadLibrary) {
        if (pdli && pdli->szDll && _stricmp(pdli->szDll, "onnxruntime.dll") == 0) {
            HMODULE hMod = nullptr;
            if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                   GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                   reinterpret_cast<LPCWSTR>(&onnxDelayHook),
                                   &hMod)) {
                wchar_t pathBuf[MAX_PATH]{};
                GetModuleFileNameW(hMod, pathBuf, MAX_PATH);
                auto dir = std::filesystem::path(pathBuf).parent_path();
                SetDllDirectoryW(dir.c_str());

                auto localShared = dir / L"onnxruntime_providers_shared.dll";
                if (std::filesystem::exists(localShared)) {
                    LoadLibraryExW(localShared.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
                }

                auto localOrt = dir / L"onnxruntime.dll";
                if (std::filesystem::exists(localOrt)) {
                    HMODULE h = LoadLibraryExW(localOrt.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
                    if (h) return reinterpret_cast<FARPROC>(h);
                }

                auto bfxOrt = std::filesystem::path(L"C:\\Program Files\\Common Files\\BorisFX\\iZotope\\Photon\\onnxruntime.dll");
                if (std::filesystem::exists(bfxOrt)) {
                    HMODULE h = LoadLibraryExW(bfxOrt.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
                    if (h) return reinterpret_cast<FARPROC>(h);
                }
            }
        }
    }
    return nullptr;
}

ExternC const PfnDliHook __pfnDliNotifyHook2 = onnxDelayHook;
#endif

namespace OpenPhoton {

using namespace Steinberg;
using namespace Steinberg::Vst;

namespace {
void preloadOnnxRuntimeFromPluginDir() {
#ifdef _WIN32
    static bool loaded = false;
    if (loaded) return;
    HMODULE hMod = nullptr;
    if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                           GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                           reinterpret_cast<LPCWSTR>(&preloadOnnxRuntimeFromPluginDir),
                           &hMod)) {
        wchar_t pathBuf[MAX_PATH]{};
        GetModuleFileNameW(hMod, pathBuf, MAX_PATH);
        auto dir = std::filesystem::path(pathBuf).parent_path();
        SetDllDirectoryW(dir.c_str());
        
        // Priority 1: Local bundle directory with LOAD_WITH_ALTERED_SEARCH_PATH
        auto localShared = dir / L"onnxruntime_providers_shared.dll";
        auto localOrt = dir / L"onnxruntime.dll";
        if (std::filesystem::exists(localOrt)) {
            LoadLibraryExW(localShared.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
            LoadLibraryExW(localOrt.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
            loaded = true;
            return;
        }

        // Priority 2: BorisFX / Photon installed modern v1.23.2 runtime
        auto bfxShared = std::filesystem::path(L"C:\\Program Files\\Common Files\\BorisFX\\iZotope\\Photon\\onnxruntime_providers_shared.dll");
        auto bfxOrt = std::filesystem::path(L"C:\\Program Files\\Common Files\\BorisFX\\iZotope\\Photon\\onnxruntime.dll");
        if (std::filesystem::exists(bfxOrt)) {
            LoadLibraryExW(bfxShared.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
            LoadLibraryExW(bfxOrt.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
            loaded = true;
            return;
        }

        // Priority 3: Standard default
        LoadLibraryW(localOrt.c_str());
        loaded = true;
    }
#endif
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
    return AudioEffect::setupProcessing(newSetup);
}

tresult PLUGIN_API Processor::setActive(TBool state) {
    if (state) {
        preloadOnnxRuntimeFromPluginDir();
        std::wstring modelPath;
        if (const char* env = std::getenv("OPENPHOTON_MODEL")) {
            std::string s(env);
            modelPath.assign(s.begin(), s.end());
        } else {
#ifdef _WIN32
            HMODULE hMod = nullptr;
            if (GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                                   GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                                   reinterpret_cast<LPCWSTR>(&preloadOnnxRuntimeFromPluginDir),
                                   &hMod)) {
                wchar_t pathBuf[MAX_PATH]{};
                GetModuleFileNameW(hMod, pathBuf, MAX_PATH);
                auto dir = std::filesystem::path(pathBuf).parent_path();
                auto candidate0 = dir / L"htdemucs.onnx";
                auto candidate1 = dir / L"bs_roformer_4stem.onnx";
                auto candidate2 = dir / L"model.onnx";
                if (std::filesystem::exists(candidate0)) {
                    modelPath = candidate0.wstring();
                } else if (std::filesystem::exists(candidate1)) {
                    modelPath = candidate1.wstring();
                } else if (std::filesystem::exists(candidate2)) {
                    modelPath = candidate2.wstring();
                } else {
                    auto pdata0 = std::filesystem::path(L"C:\\ProgramData\\OpenPhoton\\models\\htdemucs.onnx");
                    auto pdata1 = std::filesystem::path(L"C:\\ProgramData\\OpenPhoton\\models\\bs_roformer_4stem.onnx");
                    auto pdata2 = std::filesystem::path(L"C:\\ProgramData\\OpenPhoton\\models\\model.onnx");
                    if (std::filesystem::exists(pdata0)) {
                        modelPath = pdata0.wstring();
                    } else if (std::filesystem::exists(pdata1)) {
                        modelPath = pdata1.wstring();
                    } else if (std::filesystem::exists(pdata2)) {
                        modelPath = pdata2.wstring();
                    }
                }
            }
#endif
        }
        scheduler_.start(modelPath);
    } else {
        scheduler_.stop();
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

    if (mode_ == 1 && playing) {
        for (int32 s = 0; s < data.numSamples; ++s) {
            cache_.writeCaptureSample(pos + s, inL[s], inR[s]);
            outL[s] = inL[s];
            outR[s] = inR[s];
        }
        scheduler_.notifyPlayhead(pos);
        scheduler_.wakeWorker();
    } else if (mode_ == 2) {
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
    } else {
        if (outL != inL) std::copy_n(inL, data.numSamples, outL);
        if (outR != inR) std::copy_n(inR, data.numSamples, outR);
    }

    fallbackCursor_ = pos + data.numSamples;
    return kResultOk;
}

} // namespace OpenPhoton
