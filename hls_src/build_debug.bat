@echo off
REM 编译 Layer0 中间层调试程序

echo ========================================
echo 编译 Layer0 中间层调试程序
echo ========================================

g++ -std=c++11 -O2 -I. -Wall -o test_ico_conv_debug.exe ico_conv_layer0.cpp test_ico_conv_debug.cpp

if %ERRORLEVEL% EQU 0 (
    echo.
    echo ✓ 编译成功！
    echo.
    echo 运行命令: test_ico_conv_debug.exe
) else (
    echo.
    echo ✗ 编译失败！
)

pause
