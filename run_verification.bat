@echo off
REM 快速验证脚本 - Windows 版本

echo ===================================
echo IcoConv Layer 0 HLS 快速验证
echo ===================================

echo.
echo [1/3] 生成测试数据...
python generate_layer0_data.py
if %ERRORLEVEL% NEQ 0 (
    echo 错误：数据生成失败！
    pause
    exit /b 1
)

echo.
echo [2/3] 编译 C++ 测试程序...
cd hls_src
g++ -std=c++11 -O2 -I. -Wall -o test_ico_conv.exe ico_conv_layer0.cpp test_ico_conv.cpp
if %ERRORLEVEL% NEQ 0 (
    echo 错误：编译失败！请检查 g++ 是否已安装。
    cd ..
    pause
    exit /b 1
)

echo.
echo [3/3] 运行验证...
test_ico_conv.exe
cd ..

echo.
echo ===================================
echo 验证完成！
echo ===================================
pause
