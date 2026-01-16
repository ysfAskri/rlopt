cmake_minimum_required(VERSION 3.16)
project(SwiftSim LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenMP
find_package(OpenMP)

# Optimizations
if(MSVC)
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob3 /DNDEBUG /arch:AVX2 /fp:fast /GL")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "/LTCG")
    add_compile_options(/W4)
    add_compile_definitions(__AVX2__=1)
else()
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -mavx2 -ffast-math -flto -DNDEBUG")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Physics tests
add_executable(test_physics test_physics.cpp)
target_include_directories(test_physics PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

add_executable(test_complete test_complete.cpp)
target_include_directories(test_complete PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# SoA Benchmark
add_executable(benchmark_soa benchmark_soa.cpp)
target_include_directories(benchmark_soa PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
if(OpenMP_CXX_FOUND)
    target_link_libraries(benchmark_soa PUBLIC OpenMP::OpenMP_CXX)
endif()

# Python bindings (optional)
option(BUILD_PYTHON "Build Python bindings" ON)

if(BUILD_PYTHON)
    find_package(Python COMPONENTS Interpreter Development QUIET)
    find_package(pybind11 CONFIG QUIET)

    if(pybind11_FOUND AND Python_FOUND)
        pybind11_add_module(swiftsim python/swiftsim_bindings.cpp)
        target_include_directories(swiftsim PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        if(OpenMP_CXX_FOUND)
            target_link_libraries(swiftsim PRIVATE OpenMP::OpenMP_CXX)
        endif()

        # Install (scikit-build-core handles the destination)
        install(TARGETS swiftsim LIBRARY DESTINATION .)

        message(STATUS "  Python bindings: ENABLED")
    else()
        message(STATUS "  Python bindings: DISABLED (pybind11 or Python not found)")
        message(STATUS "  Install with: pip install pybind11")
    endif()
endif()

message(STATUS "")
message(STATUS "=== SwiftSim Build ===")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "  Compiler: ${CMAKE_CXX_COMPILER_ID}")
if(OpenMP_CXX_FOUND)
    message(STATUS "  OpenMP: ENABLED")
else()
    message(STATUS "  OpenMP: DISABLED")
endif()
message(STATUS "")
