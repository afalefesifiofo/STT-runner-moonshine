cmake_minimum_required(VERSION 3.27)
project(MoonshineRunner VERSION 0.1.0 LANGUAGES CXX C)

# CMake 4.x: allow subprojects that declare an older cmake_minimum_required
set(CMAKE_POLICY_VERSION_MINIMUM 3.5)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(BUILD_EXECUTABLE   "Build CLI executables (transcribe, live)"   ON)
option(BUILD_BENCHMARK    "Build benchmark executable"                  ON)
option(BUILD_TESTS        "Build unit tests"                            ON)
option(MOONSHINE_ARM_BACKEND "Enable Arm XNNPACK execution provider"   OFF)
option(MOONSHINE_USE_PORTAUDIO "Enable PortAudio for live mic input"   OFF)
option(MOONSHINE_FETCH_MODELS "Download ONNX models at configure time" ON)

set(MOONSHINE_MODEL_SIZE "tiny" CACHE STRING "Model variant: tiny or base")
set_property(CACHE MOONSHINE_MODEL_SIZE PROPERTY STRINGS tiny base)

# ---------------------------------------------------------------------------
# Dependencies via FetchContent
# ---------------------------------------------------------------------------
include(FetchContent)

# ---- ONNX Runtime ---------------------------------------------------------
# We use a pre-built release to avoid the multi-hour ORT build.
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
    if(ANDROID)
        set(ORT_PLATFORM "android-arm64-v8a")
        set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-android-arm64-v8a-1.20.1.zip")
        set(ORT_SHA256 "SKIP")  # set a real hash for production
    else()
        set(ORT_PLATFORM "linux-aarch64")
        set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-aarch64-1.20.1.tgz")
        set(ORT_SHA256 "SKIP")
    endif()
elseif(APPLE)
    set(ORT_PLATFORM "osx-arm64")
    set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-osx-arm64-1.20.1.tgz")
    set(ORT_SHA256 "SKIP")
else()
    set(ORT_PLATFORM "linux-x64")
    set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.20.1/onnxruntime-linux-x64-1.20.1.tgz")
    set(ORT_SHA256 "SKIP")
endif()

FetchContent_Declare(onnxruntime
    URL "${ORT_URL}"
    DOWNLOAD_EXTRACT_TIMESTAMP TRUE
)
FetchContent_MakeAvailable(onnxruntime)

# Locate ORT headers/lib inside the extracted archive
file(GLOB_RECURSE ORT_INCLUDE_DIRS LIST_DIRECTORIES true
    "${onnxruntime_SOURCE_DIR}/include" "${onnxruntime_SOURCE_DIR}/*/include")
list(GET ORT_INCLUDE_DIRS 0 ORT_INCLUDE_DIR)

file(GLOB ORT_LIBS
    "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime*.so"
    "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime*.dylib"
    "${onnxruntime_SOURCE_DIR}/lib/libonnxruntime*.a"
    "${onnxruntime_SOURCE_DIR}/*/lib/libonnxruntime*")
list(GET ORT_LIBS 0 ORT_LIB)

add_library(onnxruntime_imported SHARED IMPORTED)
set_target_properties(onnxruntime_imported PROPERTIES
    IMPORTED_LOCATION "${ORT_LIB}"
    INTERFACE_INCLUDE_DIRECTORIES "${ORT_INCLUDE_DIR}"
)


# ---- libsndfile (WAV reading) ---------------------------------------------
# Try pkg-config first, then fall back to find_library, then FetchContent.
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
    pkg_check_modules(SNDFILE QUIET sndfile)
endif()

if(NOT SNDFILE_FOUND)
    # Try system install without pkg-config
    find_library(SNDFILE_LIB sndfile)
    find_path(SNDFILE_INCLUDE_DIR sndfile.h)
    if(SNDFILE_LIB AND SNDFILE_INCLUDE_DIR)
        set(SNDFILE_FOUND TRUE)
        set(SNDFILE_LIBRARIES ${SNDFILE_LIB})
        set(SNDFILE_INCLUDE_DIRS ${SNDFILE_INCLUDE_DIR})
        message(STATUS "Found sndfile via find_library: ${SNDFILE_LIB}")
    endif()
endif()

if(NOT SNDFILE_FOUND)
    message(STATUS "libsndfile not found on system — fetching from source")
    FetchContent_Declare(libsndfile
        GIT_REPOSITORY https://github.com/libsndfile/libsndfile.git
        GIT_TAG        1.2.2
        GIT_SHALLOW    TRUE
    )
    set(BUILD_PROGRAMS OFF CACHE BOOL "" FORCE)
    set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
    set(BUILD_TESTING  OFF CACHE BOOL "" FORCE)
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
    FetchContent_MakeAvailable(libsndfile)
    unset(CMAKE_POLICY_VERSION_MINIMUM)
    set(SNDFILE_LIBS SndFile::sndfile)
    # Make sndfile.h available: it lives in the source tree, not the build tree
    set(SNDFILE_INCLUDE_DIRS "${libsndfile_SOURCE_DIR}/include")
else()
    # Create an imported target from the found library
    add_library(sndfile_imported UNKNOWN IMPORTED)
    set_target_properties(sndfile_imported PROPERTIES
        IMPORTED_LOCATION "${SNDFILE_LIBRARIES}"
        INTERFACE_INCLUDE_DIRECTORIES "${SNDFILE_INCLUDE_DIRS}"
    )
    set(SNDFILE_LIBS sndfile_imported)
endif()


# ---- PortAudio (optional) -------------------------------------------------
if(MOONSHINE_USE_PORTAUDIO)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(PORTAUDIO REQUIRED portaudio-2.0)
endif()

# ---- nlohmann/json (for tokenizer.hpp) ------------------------------------
FetchContent_Declare(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

# ---------------------------------------------------------------------------
# Model download
# ---------------------------------------------------------------------------
if(MOONSHINE_FETCH_MODELS)
    include(cmake/FetchMoonshineModels.cmake)
    fetch_moonshine_models("${MOONSHINE_MODEL_SIZE}" "${CMAKE_SOURCE_DIR}/models")
endif()

# ---------------------------------------------------------------------------
# Library target
# ---------------------------------------------------------------------------
add_library(moonshine_runner STATIC
    src/runner.cpp
    src/audio_utils.cpp
    src/tokenizer.cpp
)

target_include_directories(moonshine_runner PUBLIC
    include
    ${SNDFILE_INCLUDE_DIRS}
)

target_link_libraries(moonshine_runner PUBLIC
    onnxruntime_imported
    nlohmann_json::nlohmann_json
    ${SNDFILE_LIBS}
)

if(MOONSHINE_ARM_BACKEND)
    target_compile_definitions(moonshine_runner PUBLIC MOONSHINE_ARM_XNNPACK=1)
    message(STATUS "Moonshine: Arm XNNPACK EP enabled")
endif()

# ---------------------------------------------------------------------------
# CLI executables
# ---------------------------------------------------------------------------
if(BUILD_EXECUTABLE)
    add_executable(transcribe app/transcribe.cpp)
    target_link_libraries(transcribe PRIVATE moonshine_runner ${SNDFILE_LIBS})

    if(MOONSHINE_USE_PORTAUDIO)
        add_executable(live app/live.cpp)
        target_link_libraries(live PRIVATE moonshine_runner ${PORTAUDIO_LIBRARIES})
        target_include_directories(live PRIVATE ${PORTAUDIO_INCLUDE_DIRS})
        target_compile_definitions(live PRIVATE HAVE_PORTAUDIO=1)
    endif()
endif()

# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------
if(BUILD_BENCHMARK)
    add_executable(bench benchmark/bench.cpp)
    target_link_libraries(bench PRIVATE moonshine_runner ${SNDFILE_LIBS})
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
if(BUILD_TESTS)
    include(CTest)
    enable_testing()
    add_subdirectory(tests)
endif()

# ---------------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------------
install(TARGETS moonshine_runner
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
install(DIRECTORY include/moonshine_runner DESTINATION include)
