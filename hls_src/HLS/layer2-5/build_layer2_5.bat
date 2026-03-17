@echo off
setlocal

cd /d %~dp0

echo [Layer2-5] Building release test...
g++ -std=c++11 -O2 -I. -I.. -Wall -o test_ico_conv_layer2_5.exe ico_conv_layer2_5.cpp test_ico_conv_layer2_5.cpp
if errorlevel 1 goto :err

echo [Layer2-5] Building debug intermediate test...
g++ -std=c++11 -O2 -I. -I.. -Wall -o test_ico_conv_layer2_5_debug.exe ico_conv_layer2_5.cpp test_ico_conv_layer2_5_debug.cpp
if errorlevel 1 goto :err

echo Build done.
exit /b 0

:err
echo Build failed.
exit /b 1
