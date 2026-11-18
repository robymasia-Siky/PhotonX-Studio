# cmake/post_build.cmake
# Safely bundles available GGML models into the VST3 bundle Resources/models
# and deploys to user local VST3 directory (local dev only).
#
# Arguments expected:
#   -DBUNDLE_DIR="<path/to/PhotonX.vst3>"
#   -DMODELS_DIR="<path/to/models>"
#   -DDEPLOY_DIR="<path/to/user/vst3>" (optional)

if(NOT DEFINED BUNDLE_DIR OR BUNDLE_DIR STREQUAL "")
    message(STATUS "[post_build] BUNDLE_DIR was not specified; skipping.")
    return()
endif()

# Clean quotes if passed from shell
string(REPLACE "\"" "" BUNDLE_DIR "${BUNDLE_DIR}")
if(DEFINED MODELS_DIR)
    string(REPLACE "\"" "" MODELS_DIR "${MODELS_DIR}")
endif()
if(DEFINED DEPLOY_DIR)
    string(REPLACE "\"" "" DEPLOY_DIR "${DEPLOY_DIR}")
endif()

# Resolve absolute path to bundle
get_filename_component(BUNDLE_DIR_ABS "${BUNDLE_DIR}" ABSOLUTE)

# 1. Create Resources/models directory inside bundle
set(BUNDLE_MODELS_DIR "${BUNDLE_DIR_ABS}/Contents/Resources/models")
file(MAKE_DIRECTORY "${BUNDLE_MODELS_DIR}")

# 2. Safely copy any available .bin models from models/ directory
if(DEFINED MODELS_DIR AND EXISTS "${MODELS_DIR}")
    file(GLOB MODEL_FILES "${MODELS_DIR}/*.bin")
    list(LENGTH MODEL_FILES NUM_MODELS)
    if(NUM_MODELS GREATER 0)
        message(STATUS "[post_build] Bundling ${NUM_MODELS} GGML model(s) into: ${BUNDLE_MODELS_DIR}")
        foreach(MODEL_FILE ${MODEL_FILES})
            get_filename_component(MODEL_NAME "${MODEL_FILE}" NAME)
            execute_process(
                COMMAND ${CMAKE_COMMAND} -E copy_if_different "${MODEL_FILE}" "${BUNDLE_MODELS_DIR}/${MODEL_NAME}"
                RESULT_VARIABLE COPY_RES
            )
            if(COPY_RES EQUAL 0)
                message(STATUS "[post_build]   -> Copied ${MODEL_NAME}")
            else()
                message(STATUS "[post_build]   -> Notice: Failed to copy ${MODEL_NAME}")
            endif()
        endforeach()
    else()
        message(STATUS "[post_build] Notice: No .bin models found in ${MODELS_DIR}. Skipping model bundling.")
    endif()
else()
    message(STATUS "[post_build] Notice: Models directory not found at '${MODELS_DIR}'. Skipping model bundling.")
endif()

# 3. Auto-deploy to user local VST3 folder (only if DEPLOY_DIR is set and not running in CI)
if(DEFINED DEPLOY_DIR AND NOT DEPLOY_DIR STREQUAL "" AND NOT DEFINED ENV{CI})
    get_filename_component(DEPLOY_DIR_ABS "${DEPLOY_DIR}" ABSOLUTE)
    file(MAKE_DIRECTORY "${DEPLOY_DIR_ABS}")
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E copy_directory "${BUNDLE_DIR_ABS}" "${DEPLOY_DIR_ABS}"
        RESULT_VARIABLE DEPLOY_RES
    )
    if(DEPLOY_RES EQUAL 0)
        message(STATUS "[post_build] Auto-deployed PhotonX.vst3 bundle to: ${DEPLOY_DIR_ABS}")
    else()
        message(STATUS "[post_build] Notice: Failed to auto-deploy to: ${DEPLOY_DIR_ABS}")
    endif()
endif()
