# SwiftSim CUDA Build Configuration
# CUDA Physics + LibTorch PPO + Raylib/ImGui UI

cmake_minimum_required(VERSION 3.18)
project(SwiftSimCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ============================================================================
# CUDA
# ============================================================================
find_package(CUDAToolkit REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES 75 86 89)  # Turing, Ampere, Ada

# ============================================================================
# LibTorch (PyTorch C++)
# ============================================================================
# Download from: https://pytorch.org/get-started/locally/
# Extract to: C:/libtorch or set TORCH_DIR

if(NOT DEFINED TORCH_DIR)
    if(EXISTS "C:/libtorch")
        set(TORCH_DIR "C:/libtorch")
    elseif(EXISTS "$ENV{USERPROFILE}/libtorch")
        set(TORCH_DIR "$ENV{USERPROFILE}/libtorch")
    else()
        message(FATAL_ERROR "LibTorch not found. Set TORCH_DIR or download from pytorch.org")
    endif()
endif()

set(CMAKE_PREFIX_PATH ${TORCH_DIR})
find_package(Torch REQUIRED)

# ============================================================================
# Raylib
# ============================================================================
# Install via vcpkg: vcpkg install raylib
# Or download from: https://github.com/raysan5/raylib/releases

find_package(raylib CONFIG QUIET)
if(NOT raylib_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        raylib
        GIT_REPOSITORY https://github.com/raysan5/raylib.git
        GIT_TAG 5.0
    )
    FetchContent_MakeAvailable(raylib)
endif()

# ============================================================================
# ImGui + rlImGui
# ============================================================================
include(FetchContent)

FetchContent_Declare(
    imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG v1.90.1
)
FetchContent_MakeAvailable(imgui)

FetchContent_Declare(
    rlimgui
    GIT_REPOSITORY https://github.com/raylib-extras/rlImGui.git
    GIT_TAG main
)
FetchContent_MakeAvailable(rlimgui)

# ImGui library
add_library(imgui STATIC
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${rlimgui_SOURCE_DIR}/rlImGui.cpp
)
target_include_directories(imgui PUBLIC
    ${imgui_SOURCE_DIR}
    ${rlimgui_SOURCE_DIR}
)
target_link_libraries(imgui PUBLIC raylib)

# ============================================================================
# SwiftSim CUDA Training
# ============================================================================
add_executable(swiftsim_cuda
    main_cuda.cpp
    swarm_physics_cuda.cu
)

target_include_directories(swiftsim_cuda PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../ui
    ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(swiftsim_cuda PRIVATE
    ${TORCH_LIBRARIES}
    CUDA::cudart
    raylib
    imgui
)

# CUDA flags
target_compile_options(swiftsim_cuda PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -O3
        --expt-relaxed-constexpr
    >
)

# MSVC specific
if(MSVC)
    target_compile_options(swiftsim_cuda PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:/O2 /W4>
    )
    # Copy DLLs
    file(GLOB TORCH_DLLS "${TORCH_DIR}/lib/*.dll")
    add_custom_command(TARGET swiftsim_cuda POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS}
        $<TARGET_FILE_DIR:swiftsim_cuda>
    )
endif()

# ============================================================================
# CUDA Benchmark (no UI, max performance)
# ============================================================================
add_executable(swiftsim_cuda_benchmark
    benchmark_cuda.cu
)

target_include_directories(swiftsim_cuda_benchmark PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(swiftsim_cuda_benchmark PRIVATE
    CUDA::cudart
)

target_compile_options(swiftsim_cuda_benchmark PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --use_fast_math
        -O3
    >
)

# ============================================================================
# Info
# ============================================================================
message(STATUS "")
message(STATUS "=== SwiftSim CUDA Build ===")
message(STATUS "  CUDA: ${CUDAToolkit_VERSION}")
message(STATUS "  LibTorch: ${TORCH_DIR}")
message(STATUS "  Raylib: ${raylib_VERSION}")
message(STATUS "  CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "")
