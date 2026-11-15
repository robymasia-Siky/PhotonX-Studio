#pragma once
#include "pluginterfaces/base/funknown.h"
#include "pluginterfaces/vst/vsttypes.h"

namespace OpenPhoton {

static const Steinberg::FUID kProcessorUID (0x9A4F12C1, 0x77124B8A, 0xB8E10019, 0x4C2D9F01);
static const Steinberg::FUID kControllerUID(0x9A4F12C2, 0x77124B8A, 0xB8E10019, 0x4C2D9F02);

enum ParamIds : Steinberg::Vst::ParamID {
    kParamMode = 0,
    kParamDryWet = 1,
    kParamVocalsGain = 10,
    kParamDrumsGain  = 11,
    kParamBassGain   = 12,
    kParamOtherGain  = 13,
    kParamVocalsMute = 20,
    kParamDrumsMute  = 21,
    kParamBassMute   = 22,
    kParamOtherMute  = 23,
    kParamVocalsSolo = 30,
    kParamDrumsSolo  = 31,
    kParamBassSolo   = 32,
    kParamOtherSolo  = 33,
    kParamClearCache = 40
};

} // namespace OpenPhoton
