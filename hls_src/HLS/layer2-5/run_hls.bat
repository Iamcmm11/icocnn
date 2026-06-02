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
set "PRESET=%~2"
set "CPPFLAGS="

if /I "%PRESET%"=="c8t6" (
    set "CPPFLAGS=-DICO_LAYER2_5_TIME_STEPS=6 -DICO_LAYER2_5_CIN=8 -DICO_LAYER2_5_COUT=8"
    if "%~3"=="" (set "PART=xc7k325tffg900-2") else set "PART=%~3"
    if "%~4"=="" (set "CLOCK=5.0") else set "CLOCK=%~4"
    if "%~5"=="" (set "PROJECT=layer2_5_c8t6_hls_prj") else set "PROJECT=%~5"
    if "%~6"=="" (set "SOLUTION=sol1") else set "SOLUTION=%~6"
    if "%~7"=="" (set "TOP=conv_ico_layer2_5") else set "TOP=%~7"
) else if /I "%PRESET%"=="baseline" (
    if "%~3"=="" (set "PART=xc7k325tffg900-2") else set "PART=%~3"
    if "%~4"=="" (set "CLOCK=5.0") else set "CLOCK=%~4"
    if "%~5"=="" (set "PROJECT=layer2_5_hls_prj") else set "PROJECT=%~5"
    if "%~6"=="" (set "SOLUTION=sol1") else set "SOLUTION=%~6"
    if "%~7"=="" (set "TOP=conv_ico_layer2_5") else set "TOP=%~7"
) else (
    set "PRESET=custom"
    if "%~2"=="" (set "PART=xc7k325tffg900-2") else set "PART=%~2"
    if "%~3"=="" (set "CLOCK=5.0") else set "CLOCK=%~3"
    if "%~4"=="" (set "PROJECT=layer2_5_hls_prj") else set "PROJECT=%~4"
    if "%~5"=="" (set "SOLUTION=sol1") else set "SOLUTION=%~5"
    if "%~6"=="" (set "TOP=conv_ico_layer2_5") else set "TOP=%~6"
)

echo ========================================
echo Vitis HLS Terminal Runner
echo ========================================
echo Mode     : %MODE%
echo Part     : %PART%
echo Clock(ns): %CLOCK%
echo Project  : %PROJECT%
echo Solution : %SOLUTION%
echo Top      : %TOP%
echo Preset   : %PRESET%
echo CppFlags : %CPPFLAGS%
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
if not "%CPPFLAGS%"=="" (
    if "%ICO_HLS_CPPFLAGS%"=="" (
        set "ICO_HLS_CPPFLAGS=%CPPFLAGS%"
    ) else (
        set "ICO_HLS_CPPFLAGS=%ICO_HLS_CPPFLAGS% %CPPFLAGS%"
    )
)

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
