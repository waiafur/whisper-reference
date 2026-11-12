cmake_minimum_required(VERSION 3.20)

# Toolchain and Android ABI must be set BEFORE project().
# Prefer passing them on the command line:
#   cmake -S . -B build \
#     -DCMAKE_TOOLCHAIN_FILE=$HOME/openvino-android/android-ndk/build/cmake/android.toolchain.cmake \
#     -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-34 \
#     -DCMAKE_BUILD_TYPE=Release
if(NOT CMAKE_TOOLCHAIN_FILE)
    set(CMAKE_TOOLCHAIN_FILE
        "$ENV{HOME}/openvino-android/android-ndk/build/cmake/android.toolchain.cmake"
        CACHE FILEPATH "Android NDK toolchain")
endif()
if(NOT ANDROID_ABI)
    set(ANDROID_ABI "arm64-v8a" CACHE STRING "")
endif()
if(NOT ANDROID_PLATFORM)
    set(ANDROID_PLATFORM "android-34" CACHE STRING "")
endif()

project(reference_runner CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# ... остальное без изменений ...

set(OV_INSTALL "$ENV{HOME}/openvino-android/openvino-install")
set(TBB_INSTALL "$ENV{HOME}/openvino-android/one-tbb-install")
set(DR_LIBS_DIR "$ENV{HOME}/openvino-android/openvino-build/_deps/dr_libs-src")

list(APPEND CMAKE_PREFIX_PATH "${OV_INSTALL}/runtime/cmake")
find_package(OpenVINO REQUIRED HINTS "${OV_INSTALL}/runtime/cmake")
find_package(OpenVINOGenAI REQUIRED HINTS "${OV_INSTALL}/runtime/cmake")
add_executable(reference_runner reference_runner.cpp)
target_include_directories(reference_runner PRIVATE "${DR_LIBS_DIR}")
target_link_libraries(reference_runner PRIVATE openvino::genai openvino::runtime)

# Ship shared libs next to the binary
set(STAGE_DIR "${CMAKE_BINARY_DIR}/stage")
add_custom_command(TARGET reference_runner POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${STAGE_DIR}"
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:reference_runner> "${STAGE_DIR}/"
    COMMAND ${CMAKE_COMMAND} -E copy
        "$ENV{HOME}/openvino-android/android-ndk/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/lib/aarch64-linux-android/libc++_shared.so"
        "${STAGE_DIR}/"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${TBB_INSTALL}/lib/libtbb.so" "${TBB_INSTALL}/lib/libtbbmalloc.so"
        "${TBB_INSTALL}/lib/libtbbmalloc_proxy.so" "${STAGE_DIR}/"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${OV_INSTALL}/runtime/lib/aarch64/libopenvino.so"
        "${OV_INSTALL}/runtime/lib/aarch64/libopenvino_genai.so"
        "${STAGE_DIR}/"
    VERBATIM)