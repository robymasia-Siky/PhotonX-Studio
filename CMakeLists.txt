cmake_minimum_required(VERSION 3.24)
project(PhotonX VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

# Whisper.cpp configuration
set(WHISPER_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(WHISPER_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_CUDA OFF CACHE BOOL "" FORCE)

add_subdirectory(third_party/whisper.cpp EXCLUDE_FROM_ALL)

# Demucs.cpp static library
file(GLOB DEMUCS_SOURCES 
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/demucs.cpp/src/*.cpp"
)
add_library(demucs STATIC ${DEMUCS_SOURCES})
target_include_directories(demucs PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/demucs.cpp/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/demucs.cpp/vendor/eigen"
)
target_compile_features(demucs PUBLIC cxx_std_17)
if(MSVC)
    target_compile_definitions(demucs PRIVATE _USE_MATH_DEFINES NOMINMAX)
    target_compile_options(demucs PRIVATE /O2 /fp:fast)
endif()

# OpenPhoton Core Library
add_library(OpenPhotonCore STATIC
    src/core/WhisperAnalyzer.hpp
    src/core/WhisperAnalyzer.cpp
    src/core/StemCache.hpp
    src/core/ChunkScheduler.hpp
    src/core/ChunkScheduler.cpp
    src/core/STFTProcessor.hpp
    src/core/DSPChain.hpp
)
target_include_directories(OpenPhotonCore PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/whisper.cpp/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/whisper.cpp/ggml/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/demucs.cpp/src"
    "${CMAKE_CURRENT_SOURCE_DIR}/third_party/demucs.cpp/vendor/eigen"
)
target_link_libraries(OpenPhotonCore PUBLIC whisper demucs)
if(MSVC)
    target_compile_definitions(OpenPhotonCore PRIVATE _USE_MATH_DEFINES NOMINMAX)
endif()

# Standalone Test & Benchmark CLI
add_executable(PhotonX-Runner
    src/main.cpp
)
target_link_libraries(PhotonX-Runner PRIVATE OpenPhotonCore)

add_executable(test_stft
    tests/test_stft.cpp
)
target_link_libraries(test_stft PRIVATE OpenPhotonCore)

# VST3 Plugin
set(vst3sdk_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/vst3sdk")
if(EXISTS "${vst3sdk_SOURCE_DIR}/CMakeLists.txt")
    set(SMTG_CREATE_BUNDLE_FOR_WINDOWS ON CACHE BOOL "" FORCE)
    set(SMTG_CREATE_PLUGIN_LINK OFF CACHE BOOL "" FORCE)
    set(SMTG_ENABLE_VST3_PLUGIN_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(SMTG_ENABLE_VST3_HOSTING_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(SMTG_ENABLE_VSTGUI_SUPPORT ON CACHE BOOL "" FORCE)
    set(SMTG_RUN_VST_VALIDATOR OFF CACHE BOOL "" FORCE)

    add_subdirectory(${vst3sdk_SOURCE_DIR} ${PROJECT_BINARY_DIR}/vst3sdk EXCLUDE_FROM_ALL)
    smtg_enable_vst3_sdk()

    smtg_add_vst3plugin(PhotonX
        src/plugin/PluginIds.hpp
        src/plugin/PluginEntry.cpp
        src/plugin/Processor.hpp
        src/plugin/Processor.cpp
        src/plugin/Controller.hpp
        src/plugin/Controller.cpp
        src/plugin/PhotonEditor.hpp
        src/plugin/PhotonEditor.cpp
    )

    target_link_libraries(PhotonX PRIVATE
        sdk
        vstgui_support
        OpenPhotonCore
    )
endif()
