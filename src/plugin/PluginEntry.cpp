#include "public.sdk/source/main/pluginfactory.h"
#include "pluginterfaces/vst/ivstaudioprocessor.h"
#include "pluginterfaces/vst/ivsteditcontroller.h"
#include "Processor.hpp"
#include "Controller.hpp"
#include "PluginIds.hpp"

#define stringPluginName "OpenPhoton"

using namespace Steinberg;

BEGIN_FACTORY_DEF("OpenPhoton Contributors", "https://github.com/openphoton", "mailto:dev@openphoton.org")

    DEF_CLASS2(INLINE_UID_FROM_FUID(OpenPhoton::kProcessorUID),
               PClassInfo::kManyInstances,
               kVstAudioEffectClass,
               stringPluginName,
               Vst::kDistributable,
               Vst::PlugType::kFxRestoration,
               "0.1.0",
               kVstVersionString,
               OpenPhoton::Processor::createInstance)

    DEF_CLASS2(INLINE_UID_FROM_FUID(OpenPhoton::kControllerUID),
               PClassInfo::kManyInstances,
               kVstComponentControllerClass,
               stringPluginName "Controller",
               0,
               "",
               "0.1.0",
               kVstVersionString,
               OpenPhoton::Controller::createInstance)

END_FACTORY
