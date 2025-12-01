@echo off
REM Cross3D Preprocessing - MSVC Build Script
REM Author: Cross3D C Implementation
REM Date: 2024

echo ============================================
echo Cross3D Preprocessing - MSVC Build Script
echo ============================================
echo.

REM Try to find Visual Studio
where cl >nul 2>nul
if %errorlevel% neq 0 (
    echo Looking for Visual Studio...
    
    REM Try VS 2022
    if exist "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat"
        goto compile
    )
    if exist "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files\Microsoft Visual Studio\2022\Professional\VC\Auxiliary\Build\vcvars64.bat"
        goto compile
    )
    
    REM Try VS 2019
    if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
        goto compile
    )
    if exist "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvars64.bat" (
        call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvars64.bat"
        goto compile
    )
    
    echo.
    echo ERROR: Visual Studio not found!
    echo Please install Visual Studio with C++ development tools
    echo Or install MinGW-w64 and use build.bat instead
    echo.
    echo Alternative: You can compile manually with:
    echo   cl /Fe:cross3d_preprocess.exe /I include src\*.c
    exit /b 1
)

:compile
REM Create directories
if not exist obj mkdir obj
if not exist bin mkdir bin
if not exist output mkdir output

echo Compiling with MSVC...
echo.

REM Compile all source files
cl /nologo /O2 /W3 /I include /Fe:bin\cross3d_preprocess.exe ^
   src\main.c ^
   src\audio_reader.c ^
   src\fft.c ^
   src\gcc_phat.c ^
   src\srp_map.c ^
   src\test_data.c

if errorlevel 1 goto error

REM Clean up object files
del *.obj 2>nul

echo.
echo ============================================
echo Build successful!
echo Executable: bin\cross3d_preprocess.exe
echo ============================================
echo.
echo Run with: bin\cross3d_preprocess.exe
goto end

:error
echo.
echo ============================================
echo Build failed!
echo ============================================
exit /b 1

:end
