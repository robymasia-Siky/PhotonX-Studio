#pragma once

#include "public.sdk/source/vst/vstaudioeffect.h"
#include "core/StemCache.hpp"
#include "core/ChunkScheduler.hpp"

namespace OpenPhoton {

class Processor : public Steinberg::Vst::AudioEffect {
public:
    Processor();
    ~Processor() override = default;

    static Steinberg::FUnknown* createInstance(void*) {
        return static_cast<Steinberg::Vst::IAudioProcessor*>(new Processor());
    }

    Steinberg::tresult PLUGIN_API initialize(Steinberg::FUnknown* context) override;
    Steinberg::tresult PLUGIN_API terminate() override;
    Steinberg::tresult PLUGIN_API setBusArrangements(
        Steinberg::Vst::SpeakerArrangement* inputs, Steinberg::int32 numIns,
        Steinberg::Vst::SpeakerArrangement* outputs, Steinberg::int32 numOuts) override;
    Steinberg::tresult PLUGIN_API setupProcessing(Steinberg::Vst::ProcessSetup& newSetup) override;
    Steinberg::tresult PLUGIN_API setActive(Steinberg::TBool state) override;
    Steinberg::tresult PLUGIN_API process(Steinberg::Vst::ProcessData& data) override;

private:
    StemCache cache_;
    ChunkScheduler scheduler_{cache_};
    int mode_ = 0;
    StemMixParams mix_;
    int64_t fallbackCursor_ = 0;
};

} // namespace OpenPhoton
