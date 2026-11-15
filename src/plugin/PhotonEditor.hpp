#pragma once

#include "public.sdk/source/vst/vstguieditor.h"
#include "vstgui/vstgui.h"
#include "PluginIds.hpp"
#include <array>
#include <string>
#include <unordered_map>

namespace OpenPhoton {

class PhotonEditor final : public Steinberg::Vst::VSTGUIEditor,
                           public VSTGUI::IControlListener {
public:
    explicit PhotonEditor(Steinberg::Vst::EditController* controller);
    ~PhotonEditor() override = default;

    bool PLUGIN_API open(void* parent, const VSTGUI::PlatformType& platformType) override;
    void PLUGIN_API close() override;
    void valueChanged(VSTGUI::CControl* pControl) override;

    void syncParameterToUI(Steinberg::Vst::ParamID tag, Steinberg::Vst::ParamValue value);

    DELEGATE_REFCOUNT(VSTGUIEditor)

private:
    std::unordered_map<Steinberg::Vst::ParamID, VSTGUI::CControl*> controls_;
};

} // namespace OpenPhoton
