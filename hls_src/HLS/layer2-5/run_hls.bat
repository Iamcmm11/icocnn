@echo off
setlocal
chcp 65001 >nul

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%"

for /f "tokens=* delims= " %%A in ("%XILINX_HLS%") do set "XILINX_HLS=%%A"
for /f "tokens=* delims= " %%A in ("%XILINX_VITIS%") do set "XILINX_VITIS=%%A"
for /f "tokens=* delims= " %%A in ("%XILINX_VIVADO%") do set "XILINX_VIVADO=%%A"

set "MODE=%~1"
if "%MODE%"=="" set "MODE=quick"
set "PART=%~2"
if "%PART%"=="" set "PART=xc7k325tffg900-2"
set "CLOCK=%~3"
if "%CLOCK%"=="" set "CLOCK=5.0"
set "PROJECT=%~4"
if "%PROJECT%"=="" set "PROJECT=layer2_5_hls_prj"
set "SOLUTION=%~5"
if "%SOLUTION%"=="" set "SOLUTION=sol1"
set "TOP=%~6"
if "%TOP%"=="" set "TOP=conv_ico_layer2_5"

echo ========================================
echo Vitis HLS Terminal Runner
echo ========================================
echo Mode     : %MODE%
echo Part     : %PART%
echo Clock(ns): %CLOCK%
echo Project  : %PROJECT%
echo Solution : %SOLUTION%
echo Top      : %TOP%
echo ========================================

if exist "%XILINX_HLS%\bin" set "PATH=%XILINX_HLS%\bin;%PATH%"
if exist "%XILINX_VITIS%\bin" set "PATH=%XILINX_VITIS%\bin;%PATH%"
if exist "%XILINX_VIVADO%\bin" set "PATH=%XILINX_VIVADO%\bin;%PATH%"
if exist "%XILINX_HLS%\tps\win64\msys64\usr\bin" set "PATH=%XILINX_HLS%\tps\win64\msys64\usr\bin;%PATH%"
if exist "%XILINX_VITIS%\tps\win64\msys64\usr\bin" set "PATH=%XILINX_VITIS%\tps\win64\msys64\usr\bin;%PATH%"
if exist "%XILINX_VIVADO%\tps\win64\msys64\usr\bin" set "PATH=%XILINX_VIVADO%\tps\win64\msys64\usr\bin;%PATH%"
if exist "C:\Program Files\Git\usr\bin" set "PATH=C:\Program Files\Git\usr\bin;%PATH%"

where vitis_hls >nul 2>nul
if errorlevel 1 (
    echo [ERROR] vitis_hls command not found in PATH.
    echo Please source Xilinx environment first.
    popd
    exit /b 1
)

set "ICO_HLS_MODE=%MODE%"
set "ICO_HLS_PART=%PART%"
set "ICO_HLS_CLOCK=%CLOCK%"
set "ICO_HLS_PROJECT=%PROJECT%"
set "ICO_HLS_SOLUTION=%SOLUTION%"
set "ICO_HLS_TOP=%TOP%"

echo.
echo [1/2] Running Vitis HLS flow...
cmd /c vitis_hls -f run_hls.tcl
if errorlevel 1 (
    echo.
    echo [ERROR] Vitis HLS flow failed.
    popd
    exit /b 1
)

if /I "%MODE%"=="csim" (
    echo.
    echo [DONE] C simulation finished. No synthesis report generated in csim mode.
    popd
    exit /b 0
)

echo.
echo [2/2] Parsing reports...
python parse_hls_report.py --project "%PROJECT%" --solution "%SOLUTION%" --top "%TOP%"
if errorlevel 1 (
    echo [WARN] Report parser failed. Raw reports still exist in %PROJECT%\%SOLUTION%\syn\report
    popd
    exit /b 0
)

echo.
echo [DONE] Flow finished.
echo Summary: ..\..\hls_reports\layer2_5_latest_summary.md

popd
exit /b 0
