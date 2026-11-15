#include "PhotonEditor.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>

namespace OpenPhoton {

using namespace VSTGUI;

namespace {

inline CColor colBg()      { return CColor(15, 18, 25, 255); }
inline CColor colPanel()   { return CColor(24, 29, 40, 255); }
inline CColor colCard()    { return CColor(31, 37, 51, 255); }
inline CColor colBorder()  { return CColor(48, 57, 78, 255); }
inline CColor colText()    { return CColor(235, 240, 248, 255); }
inline CColor colMuted()   { return CColor(140, 152, 175, 255); }

inline CColor stemColor(int idx) {
    switch (idx) {
        case 0: return CColor(76, 201, 240, 255);
        case 1: return CColor(247, 37, 133, 255);
        case 2: return CColor(247, 127, 0, 255);
        default: return CColor(128, 237, 153, 255);
    }
}

class PhotonBackdropView : public CView {
public:
    explicit PhotonBackdropView(const CRect& size) : CView(size) {}

    void draw(CDrawContext* ctx) override {
        const CRect r = getViewSize();
        ctx->setFillColor(colBg());
        ctx->drawRect(r, kDrawFilled);

        CRect header(r.left, r.top, r.right, r.top + 64);
        ctx->setFillColor(colPanel());
        ctx->drawRect(header, kDrawFilled);
        ctx->setFrameColor(colBorder());
        ctx->setLineWidth(1.0);
        ctx->drawLine(CPoint(r.left, r.top + 64), CPoint(r.right, r.top + 64));

        ctx->setFont(kNormalFont, 18, kBoldFace);
        ctx->setFontColor(CColor(76, 201, 240, 255));
        ctx->drawString("OPEN", CRect(24, 16, 90, 48), kLeftText, true);
        ctx->setFontColor(colText());
        ctx->drawString("PHOTON", CRect(80, 16, 175, 48), kLeftText, true);

        ctx->setFont(kNormalFont, 10, kNormalFace);
        ctx->setFontColor(colMuted());
        ctx->drawString("AI STEM DEMIX & RESTORE  |  GPU ONNX", CRect(24, 40, 280, 56), kLeftText, true);

        CRect waveBox(24, 80, r.right - 24, 270);
        ctx->setFillColor(CColor(11, 14, 20, 255));
        ctx->drawRect(waveBox, kDrawFilled);
        ctx->setFrameColor(colBorder());
        ctx->drawRect(waveBox, kDrawStroked);

        ctx->setFrameColor(CColor(30, 36, 50, 255));
        for (int i = 1; i < 8; ++i) {
            const CCoord x = waveBox.left + (waveBox.getWidth() * i) / 8.0;
            ctx->drawLine(CPoint(x, waveBox.top + 1), CPoint(x, waveBox.bottom - 1));
        }

        const char* labels[4] = { "VOCALS", "DRUMS", "BASS", "OTHER" };
        const CCoord laneH = waveBox.getHeight() / 4.0;
        for (int s = 0; s < 4; ++s) {
            const CCoord midY = waveBox.top + laneH * (s + 0.5);
            const CColor c = stemColor(s);

            ctx->setFont(kNormalFont, 10, kBoldFace);
            ctx->setFontColor(c);
            ctx->drawString(labels[s], CRect(waveBox.left + 10, midY - 8, waveBox.left + 80, midY + 8), kLeftText, true);

            ctx->setFrameColor(CColor(c.red, c.green, c.blue, 180));
            ctx->setLineWidth(1.5);
            const CCoord startX = waveBox.left + 85;
            const CCoord endX = waveBox.right - 14;
            for (CCoord x = startX; x < endX; x += 4.0) {
                const double t = (x - startX) * 0.045 + s * 1.3;
                const double amp = (0.35 + 0.55 * std::abs(std::sin(t) * std::cos(t * 0.37))) * (laneH * 0.36);
                ctx->drawLine(CPoint(x, midY - amp), CPoint(x, midY + amp));
            }
        }

        ctx->setFont(kNormalFont, 11, kNormalFace);
        ctx->setFontColor(colMuted());
        ctx->drawString("Workflow: 1) Seleziona 'Capture & Demix' e premi Play   2) Passa a 'Stem Playback' per mixare gli stem",
                        CRect(24, r.bottom - 26, r.right - 24, r.bottom - 8), kLeftText, true);
        setDirty(false);
    }

    CLASS_METHODS_NOCOPY(PhotonBackdropView, CView)
};

class PhotonSlider : public CControl {
public:
    PhotonSlider(const CRect& size, IControlListener* listener, int32_t tag,
                 CColor accent, std::string title, bool isPercent200 = true)
        : CControl(size, listener, tag), accent_(accent), title_(std::move(title)), isPercent200_(isPercent200) {}

    void draw(CDrawContext* ctx) override {
        const CRect r = getViewSize();
        ctx->setFillColor(colCard());
        ctx->drawRect(r, kDrawFilled);
        ctx->setFrameColor(colBorder());
        ctx->setLineWidth(1.0);
        ctx->drawRect(r, kDrawStroked);

        const CRect track(r.left + r.getWidth() * 0.42, r.top + 30,
                          r.left + r.getWidth() * 0.58, r.bottom - 36);
        ctx->setFillColor(CColor(12, 15, 22, 255));
        ctx->drawRect(track, kDrawFilled);

        const float val = std::clamp(getValue(), 0.0f, 1.0f);
        const CCoord fillTop = track.bottom - track.getHeight() * val;
        CRect fillRect(track.left, fillTop, track.right, track.bottom);
        ctx->setFillColor(accent_);
        ctx->drawRect(fillRect, kDrawFilled);

        CRect knob(track.left - 8, fillTop - 5, track.right + 8, fillTop + 5);
        ctx->setFillColor(colText());
        ctx->drawRect(knob, kDrawFilled);

        ctx->setFont(kNormalFont, 11, kBoldFace);
        ctx->setFontColor(accent_);
        ctx->drawString(title_.c_str(), CRect(r.left, r.top + 6, r.right, r.top + 24), kCenterText, true);

        char buf[32]{};
        const int pct = static_cast<int>(std::round(val * (isPercent200_ ? 200.0f : 100.0f)));
        std::snprintf(buf, sizeof(buf), "%d%%", pct);
        ctx->setFont(kNormalFont, 11, kNormalFace);
        ctx->setFontColor(colText());
        ctx->drawString(buf, CRect(r.left, r.bottom - 26, r.right, r.bottom - 8), kCenterText, true);

        setDirty(false);
    }

    CMouseEventResult onMouseDown(CPoint& where, const CButtonState& buttons) override {
        if (!(buttons & kLButton)) return kMouseEventNotHandled;
        dragging_ = true;
        beginEdit();
        updateFromPoint(where);
        return kMouseEventHandled;
    }

    CMouseEventResult onMouseMoved(CPoint& where, const CButtonState& buttons) override {
        if (!dragging_) return kMouseEventNotHandled;
        updateFromPoint(where);
        return kMouseEventHandled;
    }

    CMouseEventResult onMouseUp(CPoint& where, const CButtonState& buttons) override {
        if (!dragging_) return kMouseEventNotHandled;
        updateFromPoint(where);
        dragging_ = false;
        endEdit();
        return kMouseEventHandled;
    }

    CLASS_METHODS_NOCOPY(PhotonSlider, CControl)

private:
    void updateFromPoint(const CPoint& where) {
        const CRect r = getViewSize();
        const CCoord top = r.top + 30;
        const CCoord bottom = r.bottom - 36;
        float v = static_cast<float>((bottom - where.y) / (bottom - top));
        v = std::clamp(v, 0.0f, 1.0f);
        if (v != getValue()) {
            setValue(v);
            valueChanged();
            invalid();
        }
    }

    CColor accent_;
    std::string title_;
    bool isPercent200_;
    bool dragging_ = false;
};

class PhotonButton : public CControl {
public:
    PhotonButton(const CRect& size, IControlListener* listener, int32_t tag,
                 std::string label, CColor activeColor, float targetValue = 1.0f, bool isToggle = true)
        : CControl(size, listener, tag), label_(std::move(label)),
          activeColor_(activeColor), targetValue_(targetValue), isToggle_(isToggle) {}

    void setSelected(bool sel) {
        selected_ = sel;
        invalid();
    }

    void draw(CDrawContext* ctx) override {
        const CRect r = getViewSize();
        const bool active = isToggle_ ? (getValue() >= 0.5f) : selected_;

        ctx->setFillColor(active ? activeColor_ : CColor(28, 34, 48, 255));
        ctx->drawRect(r, kDrawFilled);
        ctx->setFrameColor(active ? activeColor_ : colBorder());
        ctx->setLineWidth(1.0);
        ctx->drawRect(r, kDrawStroked);

        ctx->setFont(kNormalFont, 11, kBoldFace);
        ctx->setFontColor(active ? CColor(15, 18, 25, 255) : colText());
        ctx->drawString(label_.c_str(), r, kCenterText, true);
        setDirty(false);
    }

    CMouseEventResult onMouseDown(CPoint& where, const CButtonState& buttons) override {
        if (!(buttons & kLButton)) return kMouseEventNotHandled;
        beginEdit();
        if (isToggle_) {
            setValue(getValue() >= 0.5f ? 0.0f : 1.0f);
        } else {
            setValue(targetValue_);
        }
        valueChanged();
        endEdit();
        invalid();
        return kMouseEventHandled;
    }

    CLASS_METHODS_NOCOPY(PhotonButton, CControl)

private:
    std::string label_;
    CColor activeColor_;
    float targetValue_;
    bool isToggle_;
    bool selected_ = false;
};

} // namespace

PhotonEditor::PhotonEditor(Steinberg::Vst::EditController* controller)
    : VSTGUIEditor(controller)
{
    Steinberg::ViewRect vr(0, 0, 920, 540);
    setRect(vr);
}

bool PLUGIN_API PhotonEditor::open(void* parent, const PlatformType& platformType) {
    if (frame) {
        frame->forget();
        frame = nullptr;
    }

    const auto r = getRect();
    CRect size(r.left, r.top, r.right, r.bottom);
    frame = new CFrame(size, this);
    if (!frame || !frame->open(parent, platformType)) {
        return false;
    }

    controls_.clear();
    frame->addView(new PhotonBackdropView(size));

    const char* modeLabels[3] = { "Bypass", "Capture & Demix", "Stem Playback" };
    const float modeVals[3]   = { 0.0f, 0.5f, 1.0f };
    const CColor modeCols[3]  = { CColor(160, 170, 190, 255), CColor(247, 37, 133, 255), CColor(76, 201, 240, 255) };
    for (int m = 0; m < 3; ++m) {
        const CCoord x = 320 + m * 145;
        auto* btn = new PhotonButton(CRect(x, 16, x + 138, 48), this, kParamMode + 1000 + m,
                                     modeLabels[m], modeCols[m], modeVals[m], false);
        const float curMode = static_cast<float>(controller->getParamNormalized(kParamMode));
        btn->setSelected(std::abs(curMode - modeVals[m]) < 0.2f);
        frame->addView(btn);
        controls_[kParamMode + 1000 + m] = btn;
    }

    auto* clearBtn = new PhotonButton(CRect(776, 16, 896, 48), this, kParamClearCache,
                                      "Clear Cache", CColor(247, 127, 0, 255), 1.0f, true);
    frame->addView(clearBtn);
    controls_[kParamClearCache] = clearBtn;

    const char* stemTitles[4] = { "VOCALS", "DRUMS", "BASS", "OTHER" };
    for (int i = 0; i < 4; ++i) {
        const CCoord x = 24 + i * 176;
        const CColor c = stemColor(i);

        auto* slider = new PhotonSlider(CRect(x, 286, x + 160, 456), this, kParamVocalsGain + i, c, stemTitles[i], true);
        slider->setValue(static_cast<float>(controller->getParamNormalized(kParamVocalsGain + i)));
        frame->addView(slider);
        controls_[kParamVocalsGain + i] = slider;

        auto* muteBtn = new PhotonButton(CRect(x, 464, x + 76, 496), this, kParamVocalsMute + i,
                                         "MUTE", CColor(239, 71, 111, 255), 1.0f, true);
        muteBtn->setValue(static_cast<float>(controller->getParamNormalized(kParamVocalsMute + i)));
        frame->addView(muteBtn);
        controls_[kParamVocalsMute + i] = muteBtn;

        auto* soloBtn = new PhotonButton(CRect(x + 84, 464, x + 160, 496), this, kParamVocalsSolo + i,
                                         "SOLO", CColor(255, 209, 102, 255), 1.0f, true);
        soloBtn->setValue(static_cast<float>(controller->getParamNormalized(kParamVocalsSolo + i)));
        frame->addView(soloBtn);
        controls_[kParamVocalsSolo + i] = soloBtn;
    }

    auto* dryWet = new PhotonSlider(CRect(736, 286, 896, 496), this, kParamDryWet,
                                    CColor(120, 160, 255, 255), "DRY / WET", false);
    dryWet->setValue(static_cast<float>(controller->getParamNormalized(kParamDryWet)));
    frame->addView(dryWet);
    controls_[kParamDryWet] = dryWet;

    return true;
}

void PLUGIN_API PhotonEditor::close() {
    controls_.clear();
    VSTGUIEditor::close();
}

void PhotonEditor::valueChanged(CControl* pControl) {
    if (!pControl || !controller) return;
    int32_t tag = pControl->getTag();
    Steinberg::Vst::ParamID paramId = static_cast<Steinberg::Vst::ParamID>(tag);

    if (tag >= kParamMode + 1000 && tag <= kParamMode + 1002) {
        paramId = kParamMode;
    }

    const auto val = static_cast<Steinberg::Vst::ParamValue>(pControl->getValue());
    controller->beginEdit(paramId);
    controller->setParamNormalized(paramId, val);
    controller->performEdit(paramId, val);
    controller->endEdit(paramId);

    if (paramId == kParamMode) {
        syncParameterToUI(kParamMode, val);
    }
}

void PhotonEditor::syncParameterToUI(Steinberg::Vst::ParamID tag, Steinberg::Vst::ParamValue value) {
    if (!frame) return;
    if (tag == kParamMode) {
        const float modeVals[3] = { 0.0f, 0.5f, 1.0f };
        for (int m = 0; m < 3; ++m) {
            if (auto it = controls_.find(kParamMode + 1000 + m); it != controls_.end()) {
                if (auto* btn = dynamic_cast<PhotonButton*>(it->second)) {
                    btn->setSelected(std::abs(static_cast<float>(value) - modeVals[m]) < 0.2f);
                }
            }
        }
        return;
    }
    if (auto it = controls_.find(tag); it != controls_.end()) {
        it->second->setValue(static_cast<float>(value));
        it->second->invalid();
    }
}

} // namespace OpenPhoton
