#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include "../src/core/STFTProcessor.hpp"

using namespace OpenPhoton;

void testSTFTDimensions() {
    std::cout << "[TEST] Running testSTFTDimensions...\n";
    STFTProcessor stft(1024, 512);
    assert(stft.fftSize() == 1024);
    assert(stft.hopSize() == 512);
    assert(stft.numBins() == 513); // 1024 / 2 + 1
    std::cout << "  -> PASSED: STFT Dimensions (fftSize=1024, hopSize=512, bins=513)\n";
}

void testSTFTReconstructionSineWave() {
    std::cout << "[TEST] Running testSTFTReconstructionSineWave...\n";
    STFTProcessor stft(1024, 512);

    const size_t numSamples = 48000; // 1 second @ 48kHz
    std::vector<float> input(numSamples, 0.0f);
    
    // Generate 440 Hz pure sine wave
    const float freq = 440.0f;
    const float sampleRate = 48000.0f;
    for (size_t i = 0; i < numSamples; ++i) {
        input[i] = 0.8f * std::sin(2.0f * 3.14159265358979323846f * freq * static_cast<float>(i) / sampleRate);
    }

    std::vector<std::complex<float>> spec;
    size_t outFrames = 0;
    stft.forward(input.data(), numSamples, spec, outFrames);
    assert(outFrames > 0);
    assert(spec.size() == outFrames * 513);

    std::vector<float> reconstructed;
    stft.inverse(spec, outFrames, numSamples, reconstructed);
    assert(reconstructed.size() == numSamples);

    // Verify reconstruction accuracy (skip boundary transients at start/end)
    const size_t skipBoundary = 1024;
    float maxErr = 0.0f;
    float sumSqErr = 0.0f;
    size_t count = 0;

    for (size_t i = skipBoundary; i < numSamples - skipBoundary; ++i) {
        const float err = std::abs(input[i] - reconstructed[i]);
        if (err > maxErr) maxErr = err;
        sumSqErr += err * err;
        count++;
    }

    const float rmse = std::sqrt(sumSqErr / static_cast<float>(count));
    std::cout << "  -> Max Error: " << maxErr << ", RMSE: " << rmse << "\n";
    assert(rmse < 1e-3f);
    std::cout << "  -> PASSED: Perfect reconstruction RMSE < 1e-3\n";
}

int main() {
    std::cout << "========================================\n";
    std::cout << "  Open-Photon Unit Test: STFT DSP Engine\n";
    std::cout << "========================================\n";
    try {
        testSTFTDimensions();
        testSTFTReconstructionSineWave();
        std::cout << "ALL STFT TESTS PASSED SUCCESSFULLY!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "TEST FAILED with exception: " << e.what() << "\n";
        return 1;
    }
}
