# SPDX-FileCopyrightText: 2024 STT-Runner-Moonshine contributors
# SPDX-License-Identifier: Apache-2.0
#
# cmake/FetchMoonshineModels.cmake
#
# Downloads pre-exported ONNX models from HuggingFace for Moonshine.
# Called from the top-level CMakeLists.txt when MOONSHINE_FETCH_MODELS=ON.

function(fetch_moonshine_models MODEL_SIZE DEST_DIR)
    set(HF_BASE "https://huggingface.co/moonshine-ai/moonshine-${MODEL_SIZE}-onnx/resolve/main")
    set(MODEL_DIR "${DEST_DIR}/moonshine-${MODEL_SIZE}")

    set(FILES
        "encoder_model_int8.onnx"
        "decoder_model_merged_int8.onnx"
        "tokenizer.json"
        "tokenizer_config.json"
    )

    message(STATUS "Moonshine: fetching ${MODEL_SIZE} ONNX models into ${MODEL_DIR}")
    file(MAKE_DIRECTORY "${MODEL_DIR}")

    foreach(F IN LISTS FILES)
        set(DEST_FILE "${MODEL_DIR}/${F}")
        if(NOT EXISTS "${DEST_FILE}")
            message(STATUS "  Downloading ${F}...")
            file(DOWNLOAD
                "${HF_BASE}/${F}"
                "${DEST_FILE}"
                SHOW_PROGRESS
                STATUS DL_STATUS
                TLS_VERIFY ON
            )
            list(GET DL_STATUS 0 DL_CODE)
            list(GET DL_STATUS 1 DL_MSG)
            if(NOT DL_CODE EQUAL 0)
                message(WARNING "  Failed to download ${F}: ${DL_MSG}. "
                    "Run scripts/export_onnx.py manually and place files in ${MODEL_DIR}/")
            else()
                message(STATUS "  OK: ${F}")
            endif()
        else()
            message(STATUS "  Cached: ${F}")
        endif()
    endforeach()

    # Expose model dir as a cache variable so targets can find it at runtime
    set(MOONSHINE_MODEL_DIR "${MODEL_DIR}" CACHE PATH "Moonshine ONNX model directory" FORCE)
    message(STATUS "Moonshine: model dir = ${MOONSHINE_MODEL_DIR}")
endfunction()
