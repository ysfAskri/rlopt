@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
cd /d "%~dp0"
echo Building SwiftSim CUDA Benchmark...
nvcc -O3 --use_fast_math -arch=sm_61 -o benchmark_cuda.exe benchmark_cuda.cu -I. -Wno-deprecated-gpu-targets
if %ERRORLEVEL% EQU 0 (
    echo Build successful!
    echo Running benchmark...
    benchmark_cuda.exe
) else (
    echo Build failed!
)
