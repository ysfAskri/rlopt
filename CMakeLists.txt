cmake_minimum_required(VERSION 3.16)
project(SoA_AoS_Benchmark LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Optimizations maximales pour le benchmark
if(MSVC)
    # MSVC optimizations
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /Ob3 /DNDEBUG /arch:AVX2 /fp:fast /GL")
    set(CMAKE_EXE_LINKER_FLAGS_RELEASE "/LTCG")
    add_compile_options(/W4)
else()
    # GCC/Clang optimizations
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math -flto -DNDEBUG")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Benchmark principal
add_executable(benchmark_soa_aos benchmark_soa_aos.cpp)

# Version avec OpenMP si disponible
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    add_executable(benchmark_soa_aos_omp benchmark_soa_aos.cpp)
    target_link_libraries(benchmark_soa_aos_omp PUBLIC OpenMP::OpenMP_CXX)
    target_compile_definitions(benchmark_soa_aos_omp PRIVATE USE_OPENMP=1)
    message(STATUS "OpenMP found - building parallel version")
else()
    message(STATUS "OpenMP not found - only building single-threaded version")
endif()

# Instructions de build
message(STATUS "")
message(STATUS "=== Build Instructions ===")
message(STATUS "  mkdir build && cd build")
message(STATUS "  cmake .. -DCMAKE_BUILD_TYPE=Release")
message(STATUS "  cmake --build . --config Release")
message(STATUS "  ./Release/benchmark_soa_aos.exe (Windows)")
message(STATUS "  ./benchmark_soa_aos (Linux/Mac)")
message(STATUS "")
