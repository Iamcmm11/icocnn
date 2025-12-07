@echo off
REM Layer0 C++ 验证 - 编译并测试

echo ========================================
echo   Layer0 IcoConv C++ 验证
echo ========================================
echo.

REM 检查数据文件
echo [检查] 验证数据文件是否存在...
if not exist "..\hls_testdata\layer0\input.txt" (
    echo 错误: 缺少 input.txt
    echo 请先将测试数据放入 ..\hls_testdata\layer0\ 目录
    echo 详见: 数据文件说明.md
    pause
    exit /b 1
)
if not exist "..\hls_testdata\layer0\weight.txt" (
    echo 错误: 缺少 weight.txt
    pause
    exit /b 1
)
if not exist "..\hls_testdata\layer0\output.txt" (
    echo 错误: 缺少 output.txt (PyTorch 参考输出)
    pause
    exit /b 1
)
echo ✓ 数据文件检查通过
echo.

REM 编译
echo [编译] 编译 C++ 代码...
cd src
mingw32-make clean
mingw32-make
if errorlevel 1 (
    echo.
    echo 编译失败! 请检查编译器是否安装 (MinGW/MSVC)
    cd ..
    pause
    exit /b 1
)
cd ..
echo ✓ 编译成功
echo.

REM 运行测试
echo [测试] 运行验证程序...
echo ========================================
bin\test_ico_conv.exe
set TEST_RESULT=%errorlevel%
echo ========================================
echo.

REM 显示结果
if %TEST_RESULT% equ 0 (
    echo ✓✓✓ 测试通过! C++ 实现与 PyTorch 一致 ✓✓✓
    echo.
    echo 下一步: 可以开始 Vivado HLS 综合了
) else (
    echo ✗✗✗ 测试失败! 请检查算法逻辑 ✗✗✗
    echo.
    echo 调试建议:
    echo 1. 检查数据文件格式是否正确
    echo 2. 查看输出的误差值
    echo 3. 对比 output_hls.txt 和 output.txt
)

echo.
pause
