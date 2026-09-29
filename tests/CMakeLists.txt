cmake_minimum_required(VERSION 3.27)

# ---- nlohmann/json (needed by tokenizer.hpp) ------------------------------
include(FetchContent)
FetchContent_Declare(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json.git
    GIT_TAG        v3.11.3
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(nlohmann_json)

# ---- Test executable -------------------------------------------------------
add_executable(test_runner
    test_runner.cpp
)
target_link_libraries(test_runner PRIVATE
    moonshine_runner
    nlohmann_json::nlohmann_json
)
target_include_directories(test_runner PRIVATE
    ${CMAKE_SOURCE_DIR}/src   # for tokenizer.hpp
)

add_executable(test_audio_utils
    test_audio_utils.cpp
)
target_link_libraries(test_audio_utils PRIVATE
    moonshine_runner
)

# ---- CTest entries --------------------------------------------------------
add_test(NAME RunnerTest     COMMAND test_runner)
add_test(NAME AudioUtilsTest COMMAND test_audio_utils)
