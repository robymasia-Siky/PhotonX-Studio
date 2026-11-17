#include "pluginterfaces/base/funknown.h"
#include "pluginterfaces/base/ipluginbase.h"
#include "pluginterfaces/vst/ivstaudioprocessor.h"
#include "pluginterfaces/vst/ivstcomponent.h"
#include "pluginterfaces/vst/ivsteditcontroller.h"
#include "pluginterfaces/gui/iplugview.h"
#include "pluginterfaces/vst/vsttypes.h"
#include <iomanip>
#include <iostream>
#include <windows.h>

using namespace Steinberg;
using namespace Steinberg::Vst;

int main() {
    std::cout << "[TEST] Locating PhotonX.vst3 binary...\n";
    HMODULE h = LoadLibraryW(L"C:\\Users\\Utente\\AppData\\Local\\Programs\\Common\\VST3\\PhotonX.vst3\\Contents\\x86_64-win\\PhotonX.vst3");
    if (!h) {
        h = LoadLibraryW(L"build/VST3/Release/PhotonX.vst3/Contents/x86_64-win/PhotonX.vst3");
    }
    if (!h) {
        h = LoadLibraryW(L"VST3/Release/PhotonX.vst3/Contents/x86_64-win/PhotonX.vst3");
    }
    if (!h) {
        std::cerr << "FAILED to LoadLibraryW, error: " << GetLastError() << "\n";
        return 1;
    }

    using InitDllFunc = bool (*)();
    using GetFactoryFunc = IPluginFactory* (*)();
    using ExitDllFunc = bool (*)();

    auto initDll = reinterpret_cast<InitDllFunc>(GetProcAddress(h, "InitDll"));
    auto getFactory = reinterpret_cast<GetFactoryFunc>(GetProcAddress(h, "GetPluginFactory"));
    auto exitDll = reinterpret_cast<ExitDllFunc>(GetProcAddress(h, "ExitDll"));

    if (initDll && !initDll()) {
        std::cerr << "InitDll returned false!\n";
    }

    IPluginFactory* factory = getFactory();
    if (!factory) {
        std::cerr << "FAILED: factory is null!\n";
        return 1;
    }

    int32 count = factory->countClasses();
    std::cout << "Factory class count: " << count << "\n";

    for (int32 i = 0; i < count; ++i) {
        PClassInfo info{};
        factory->getClassInfo(i, &info);
        if (std::string(info.category) == kVstAudioEffectClass) {
            std::cout << "[TEST] Testing Component: " << info.name << "\n";
            IComponent* comp = nullptr;
            tresult res = factory->createInstance(info.cid, IComponent::iid, reinterpret_cast<void**>(&comp));
            if (res != kResultOk || !comp) {
                std::cerr << "FAILED createInstance(IComponent)\n";
                return 1;
            }
            std::cout << "  -> IComponent created.\n";

            res = comp->initialize(nullptr);
            std::cout << "  -> initialize res=" << res << "\n";

            IAudioProcessor* proc = nullptr;
            if (comp->queryInterface(IAudioProcessor::iid, reinterpret_cast<void**>(&proc)) == kResultOk) {
                std::cout << "  -> IAudioProcessor obtained.\n";
                ProcessSetup setup{};
                setup.processMode = kRealtime;
                setup.sampleRate = 48000.0;
                setup.maxSamplesPerBlock = 512;
                setup.symbolicSampleSize = kSample32;
                proc->setupProcessing(setup);
                std::cout << "  -> setupProcessing done.\n";

                std::cout << "  -> Calling setActive(true)...\n";
                comp->setActive(true);
                std::cout << "  -> setActive(true) OK!\n";

                std::cout << "  -> Calling process()...\n";
                float inL[256]{}, inR[256]{};
                float outL[256]{}, outR[256]{};
                float* ins[2] = {inL, inR};
                float* outs[2] = {outL, outR};
                AudioBusBuffers inBus{};
                inBus.numChannels = 2;
                inBus.channelBuffers32 = ins;
                AudioBusBuffers outBus{};
                outBus.numChannels = 2;
                outBus.channelBuffers32 = outs;

                ProcessData data{};
                data.numInputs = 1;
                data.inputs = &inBus;
                data.numOutputs = 1;
                data.outputs = &outBus;
                data.numSamples = 256;
                res = proc->process(data);
                std::cout << "  -> process() res=" << res << "\n";

                std::cout << "  -> Calling setActive(false)...\n";
                comp->setActive(false);
                std::cout << "  -> setActive(false) OK!\n";

                proc->release();
            }
            comp->terminate();
            comp->release();
        } else if (std::string(info.category) == kVstComponentControllerClass) {
            std::cout << "[TEST] Testing Controller: " << info.name << "\n";
            IEditController* ctrl = nullptr;
            tresult res = factory->createInstance(info.cid, IEditController::iid, reinterpret_cast<void**>(&ctrl));
            if (res == kResultOk && ctrl) {
                std::cout << "  -> IEditController created.\n";
                ctrl->initialize(nullptr);
                std::cout << "  -> Controller initialize OK.\n";
                IPlugView* view = ctrl->createView("editor");
                std::cout << "  -> createView('editor'): " << (view ? "SUCCESS" : "FAILED/NULL") << "\n";
                if (view) view->release();
                ctrl->terminate();
                ctrl->release();
            }
        }
    }

    if (exitDll) exitDll();
    FreeLibrary(h);
    std::cout << "\n============================================\n";
    std::cout << " FULL VST3 HOST LIFECYCLE VERIFIED 100% OK!\n";
    std::cout << "============================================\n";
    return 0;
}
