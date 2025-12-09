@echo off
REM HLS Implementation 简易编译脚本
echo ========================================
echo   HLS Layer0 C++ 编译（简易版）
echo ========================================
echo.

REM 创建输出目录
if not exist "bin" mkdir "bin"
if not exist "build" mkdir "build"

echo [1/2] 编译代码...
cd src
g++ -std=c++11 -O2 -Wall -I..\include ico_conv_layer0.cpp utils.cpp test_ico_conv.cpp -o ..\bin\test_ico_conv.exe -lm

if %errorlevel% neq 0 (
    echo.
    echo [错误] 编译失败！
    cd ..
    pause
    exit /b 1
)

cd ..
echo.
echo [2/2] 编译成功！
echo 生成文件: bin\test_ico_conv.exe
echo.
echo ========================================
echo 提示：运行 bin\test_ico_conv.exe 来执行测试
echo ========================================
pause
