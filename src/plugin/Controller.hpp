#pragma once
#include "public.sdk/source/vst/vsteditcontroller.h"

namespace OpenPhoton {

class Controller : public Steinberg::Vst::EditController {
public:
    static Steinberg::FUnknown* createInstance(void*) {
        return static_cast<Steinberg::Vst::IEditController*>(new Controller());
    }
    Steinberg::tresult PLUGIN_API initialize(Steinberg::FUnknown* context) override;
};

} // namespace OpenPhoton
