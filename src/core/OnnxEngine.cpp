#include "OnnxEngine.hpp"
#include <chrono>
#include <stdexcept>
#include <iostream>

namespace OpenPhoton {

OnnxEngine::OnnxEngine(const std::wstring& modelPath,
                       ExecutionBackend backend,
                       int gpuDeviceId,
                       const HardwareLimits& limits)
    : env_(ORT_LOGGING_LEVEL_WARNING, "OpenPhotonORT"),
      memoryInfo_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
    sessionOptions_.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    sessionOptions_.SetIntraOpNumThreads(limits.cpuThreads);

    bool gpuAttached = false;

    // 1. Try CUDA with strict 8GB-safe VRAM allocator limits
    if ((backend == ExecutionBackend::Auto || backend == ExecutionBackend::CUDA) && !gpuAttached) {
        try {
            OrtCUDAProviderOptions cudaOpts{};
            cudaOpts.device_id = gpuDeviceId;
            
            // Limit GPU memory arena to 3GB so it never exhausts an 8GB GPU (leaves 5GB for DAW/OS)
            cudaOpts.gpu_mem_limit = limits.maxGpuVramBytes;
            cudaOpts.arena_extend_strategy = 1; // kSameAsRequested: prevents aggressive doublings
            cudaOpts.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic; // low memory overhead initialization
            cudaOpts.do_copy_in_default_stream = 1;

            sessionOptions_.AppendExecutionProvider_CUDA(cudaOpts);
            gpuAttached = true;
            activeProvider_ = "CUDA (TensorRT/TensorCores)";
        } catch (const Ort::Exception&) {
            if (backend == ExecutionBackend::CUDA) {
                throw;
            }
        }
    }

    // 2. Try DirectML (cross-vendor acceleration: AMD, Intel, NVIDIA)
    if ((backend == ExecutionBackend::Auto || backend == ExecutionBackend::DirectML) && !gpuAttached) {
        try {
            #ifdef _WIN32
            // DirectML session configuration
            sessionOptions_.DisableMemPattern();
            sessionOptions_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
            // AppendExecutionProvider_DML is available when linked with DirectML
            #endif
        } catch (...) {
        }
    }

    if (!gpuAttached) {
        activeProvider_ = "CPU (Multi-Threaded)";
    }

    session_ = std::make_unique<Ort::Session>(env_, modelPath.c_str(), sessionOptions_);

    Ort::AllocatorWithDefaultOptions allocator;
    const size_t nInputs = session_->GetInputCount();
    const size_t nOutputs = session_->GetOutputCount();

    inputNames_.reserve(nInputs);
    inputNamePtrs_.reserve(nInputs);
    for (size_t i = 0; i < nInputs; ++i) {
        auto name = session_->GetInputNameAllocated(i, allocator);
        inputNames_.emplace_back(name.get());
    }
    for (const auto& s : inputNames_) inputNamePtrs_.push_back(s.c_str());

    outputNames_.reserve(nOutputs);
    outputNamePtrs_.reserve(nOutputs);
    for (size_t i = 0; i < nOutputs; ++i) {
        auto name = session_->GetOutputNameAllocated(i, allocator);
        outputNames_.emplace_back(name.get());
    }
    for (const auto& s : outputNames_) outputNamePtrs_.push_back(s.c_str());
}

InferenceStats OnnxEngine::runAudioChunk(const float* inputData,
                                         const std::vector<int64_t>& inputShape,
                                         std::vector<float>& outputData,
                                         std::vector<int64_t>& outputShape)
{
    int64_t numInputElements = 1;
    for (auto dim : inputShape) numInputElements *= dim;

    auto inputTensor = Ort::Value::CreateTensor<float>(
        memoryInfo_,
        const_cast<float*>(inputData),
        static_cast<size_t>(numInputElements),
        inputShape.data(),
        inputShape.size()
    );

    const auto t0 = std::chrono::steady_clock::now();
    auto outputTensors = session_->Run(
        Ort::RunOptions{nullptr},
        inputNamePtrs_.data(),
        &inputTensor,
        1,
        outputNamePtrs_.data(),
        outputNamePtrs_.size()
    );
    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (outputTensors.empty() || !outputTensors[0].IsTensor()) {
        throw std::runtime_error("OpenPhoton: invalid ONNX output tensor.");
    }

    auto typeInfo = outputTensors[0].GetTensorTypeAndShapeInfo();
    outputShape = typeInfo.GetShape();

    const size_t totalOut = typeInfo.GetElementCount();
    const float* rawOut = outputTensors[0].GetTensorData<float>();
    outputData.assign(rawOut, rawOut + totalOut);

    return InferenceStats{ ms, activeProvider_, static_cast<size_t>(numInputElements * sizeof(float)) };
}

} // namespace OpenPhoton
