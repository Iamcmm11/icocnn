@echo off
setlocal

cd /d %~dp0

set "PRESET=%~1"
if "%PRESET%"=="" set "PRESET=baseline"

set "DEFS="
set "TARGET_SUFFIX="

if /I "%PRESET%"=="baseline" goto :build
if /I "%PRESET%"=="c8t6" (
    set "DEFS=-DICO_LAYER2_5_TIME_STEPS=6 -DICO_LAYER2_5_CIN=8 -DICO_LAYER2_5_COUT=8"
    set "TARGET_SUFFIX=_c8t6"
    goto :build
)

echo Unknown preset: %PRESET%
echo Usage: build_layer2_5.bat [baseline^|c8t6]
exit /b 1

:build
echo [Layer2-5] Preset: %PRESET%
echo [Layer2-5] Defines: %DEFS%

echo [Layer2-5] Building release test...
g++ -std=c++11 -O2 -I. -I.. -Wall %DEFS% -o test_ico_conv_layer2_5%TARGET_SUFFIX%.exe ico_conv_layer2_5.cpp test_ico_conv_layer2_5.cpp
if errorlevel 1 goto :err

if /I "%PRESET%"=="c8t6" goto :done

echo [Layer2-5] Building debug intermediate test...
g++ -std=c++11 -O2 -I. -I.. -Wall -o test_ico_conv_layer2_5_debug.exe ico_conv_layer2_5.cpp test_ico_conv_layer2_5_debug.cpp
if errorlevel 1 goto :err

:done
echo Build done.
exit /b 0

:err
echo Build failed.
exit /b 1
