@echo off
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
cd /d "C:\Users\kingy\rlopt\swiftsim\cuda"

set RAYLIB_DIR=C:\Users\kingy\rlopt\deps\raylib-5.0_win64_msvc16

echo.
echo Building SwiftSim Raylib + CUDA Demo...
echo RAYLIB_DIR: %RAYLIB_DIR%
echo.

echo Step 1: Compiling CUDA physics wrapper...
nvcc -O3 --use_fast_math -arch=sm_61 -c physics_wrapper.cu -o physics_wrapper.obj -I. -Wno-deprecated-gpu-targets -Xcompiler "/MD /EHsc"
if %ERRORLEVEL% NEQ 0 goto :error

echo Step 2: Compiling Raylib main...
cl /O2 /MD /EHsc /c raylib_main.cpp /I. /I"%RAYLIB_DIR%\include"
if %ERRORLEVEL% NEQ 0 goto :error

echo Step 3: Linking...
link /OUT:swiftsim_demo.exe raylib_main.obj physics_wrapper.obj ^
    "%RAYLIB_DIR%\lib\raylib.lib" ^
    cudart.lib ^
    opengl32.lib gdi32.lib winmm.lib user32.lib shell32.lib ^
    /LIBPATH:"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\lib\x64"
if %ERRORLEVEL% NEQ 0 goto :error

echo.
echo Build successful!
echo.
echo Copying raylib.dll...
copy "%RAYLIB_DIR%\lib\raylib.dll" . >nul 2>&1

echo.
echo Running demo...
echo.
swiftsim_demo.exe
goto :end

:error
echo.
echo Build failed!

:end
