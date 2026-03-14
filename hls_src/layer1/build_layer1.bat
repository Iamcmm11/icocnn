@echo off
setlocal

cd /d %~dp0

echo [Layer1] Building release test...
g++ -std=c++11 -O2 -I. -I.. -Wall -o test_ico_conv_layer1.exe ico_conv_layer1.cpp test_ico_conv_layer1.cpp
if errorlevel 1 goto :err

echo [Layer1] Building debug intermediate test...
g++ -std=c++11 -O2 -I. -I.. -Wall -o test_ico_conv_layer1_debug.exe ico_conv_layer1.cpp test_ico_conv_layer1_debug.cpp
if errorlevel 1 goto :err

echo Build done.
exit /b 0

:err
echo Build failed.
exit /b 1
