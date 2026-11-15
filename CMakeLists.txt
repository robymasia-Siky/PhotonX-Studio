cmake_minimum_required(VERSION 3.24)
project(OpenPhoton VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

set(ORT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/third_party/onnxruntime")
add_library(onnxruntime SHARED IMPORTED)
set_target_properties(onnxruntime PROPERTIES
    IMPORTED_IMPLIB "${ORT_ROOT}/lib/onnxruntime.lib"
    IMPORTED_LOCATION "${ORT_ROOT}/lib/onnxruntime.dll"
    INTERFACE_INCLUDE_DIRECTORIES "${ORT_ROOT}/include"
)

add_library(OpenPhotonCore STATIC
    src/core/OnnxEngine.cpp
    src/core/OnnxEngine.hpp
    src/core/STFTProcessor.hpp
    src/core/StemCache.hpp
    src/core/ChunkScheduler.hpp
)
target_include_directories(OpenPhotonCore PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(OpenPhotonCore PUBLIC onnxruntime)

set(SMTG_CREATE_BUNDLE_FOR_WINDOWS ON CACHE BOOL "" FORCE)
set(SMTG_ENABLE_VST3_PLUGIN_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SMTG_ENABLE_VST3_HOSTING_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SMTG_ENABLE_VSTGUI_SUPPORT OFF CACHE BOOL "" FORCE)
set(SMTG_RUN_VST_VALIDATOR OFF CACHE BOOL "" FORCE)

set(vst3sdk_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/vst3sdk")
add_subdirectory(${vst3sdk_SOURCE_DIR} ${PROJECT_BINARY_DIR}/vst3sdk EXCLUDE_FROM_ALL)
smtg_enable_vst3_sdk()

smtg_add_vst3plugin(OpenPhoton
    src/plugin/PluginIds.hpp
    src/plugin/PluginEntry.cpp
    src/plugin/Processor.hpp
    src/plugin/Processor.cpp
    src/plugin/Controller.hpp
    src/plugin/Controller.cpp
)

target_link_libraries(OpenPhoton PRIVATE
    sdk
    OpenPhotonCore
)

if(WIN32)
    target_link_libraries(OpenPhoton PRIVATE delayimp)
    target_link_options(OpenPhoton PRIVATE "/DELAYLOAD:onnxruntime.dll")
    add_custom_command(TARGET OpenPhoton POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
            "${ORT_ROOT}/lib/onnxruntime.dll"
            "${ORT_ROOT}/lib/onnxruntime_providers_shared.dll"
            "${ORT_ROOT}/lib/onnxruntime_providers_cuda.dll"
            "$<TARGET_FILE_DIR:OpenPhoton>"
    )
endif()
