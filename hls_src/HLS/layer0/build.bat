@echo off
chcp 65001 >nul
REM HLS Layer0 Compile Script
echo ================================
echo HLS Layer0 C++ Compilation
echo ================================
echo.

echo [1/2] Compiling...
g++ -std=c++11 -O2 -I. -Wall ico_conv_layer0.cpp test_ico_conv.cpp -o test_ico_conv.exe

if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Compilation failed!
    pause
    exit /b 1
)

echo.
echo [2/2] Compilation successful!
echo Output: test_ico_conv.exe
echo.
echo ================================
echo Run test_ico_conv.exe to execute test
echo ================================
pause
