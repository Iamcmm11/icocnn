@echo off
setlocal EnableDelayedExpansion
chcp 65001 >nul

set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%"

for /f "tokens=* delims= " %%A in ("%XILINX_HLS%") do set "XILINX_HLS=%%A"
for /f "tokens=* delims= " %%A in ("%XILINX_VITIS%") do set "XILINX_VITIS=%%A"
for /f "tokens=* delims= " %%A in ("%XILINX_VIVADO%") do set "XILINX_VIVADO=%%A"

set "MODE=%~1"
if "%MODE%"=="" set "MODE=csim"
set "PART=%~2"
if "%PART%"=="" set "PART=xc7k325tffg900-2"
set "CLOCK=%~3"
if "%CLOCK%"=="" set "CLOCK=5.0"
set "PROJECT=%~4"
if "%PROJECT%"=="" set "PROJECT=stage1_ifan_c8_r2_frontend_hls_prj"
set "SOLUTION=%~5"
if "%SOLUTION%"=="" set "SOLUTION=sol1"
set "TOP=%~6"
if "%TOP%"=="" set "TOP=ifan_dual_frontend_top"
set "PROJECT_ROOT=%~7"
if "%PROJECT_ROOT%"=="" set "PROJECT_ROOT=%SCRIPT_DIR%_hls_work"

echo ========================================
echo Vitis HLS Terminal Runner
echo ========================================
echo Mode     : %MODE%
echo Part     : %PART%
echo Clock(ns): %CLOCK%
echo Project  : %PROJECT%
echo ProjRoot : %PROJECT_ROOT%
echo Solution : %SOLUTION%
echo Top      : %TOP%
echo ========================================
echo.

if exist "%XILINX_HLS%\bin" set "PATH=%XILINX_HLS%\bin;%PATH%"
if exist "%XILINX_VITIS%\bin" set "PATH=%XILINX_VITIS%\bin;%PATH%"
if exist "%XILINX_VIVADO%\bin" set "PATH=%XILINX_VIVADO%\bin;%PATH%"
if exist "%XILINX_HLS%\tps\win64\msys64\usr\bin" set "PATH=%XILINX_HLS%\tps\win64\msys64\usr\bin;%PATH%"
if exist "%XILINX_VITIS%\tps\win64\msys64\usr\bin" set "PATH=%XILINX_VITIS%\tps\win64\msys64\usr\bin;%PATH%"
if exist "%XILINX_VIVADO%\tps\win64\msys64\usr\bin" set "PATH=%XILINX_VIVADO%\tps\win64\msys64\usr\bin;%PATH%"
if exist "C:\Program Files\Git\usr\bin" set "PATH=C:\Program Files\Git\usr\bin;%PATH%"
if exist "G:\PostGraduateFile\Git\usr\bin" set "PATH=G:\PostGraduateFile\Git\usr\bin;%PATH%"

set "LOG_DIR=%SCRIPT_DIR%hls_eval_logs"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"
set "ENV_LOG=%LOG_DIR%\preflight_%MODE%.log"
(
    echo === HLS Preflight ===
    echo Mode=!MODE!
    echo ProjectRoot=!PROJECT_ROOT!
    echo ScriptDir=!SCRIPT_DIR!
    echo.
    echo --- PATH ---
    echo(!PATH!
    echo.
    echo --- where vitis_hls ---
    where vitis_hls
    echo --- where g++ ---
    where g++
    echo --- where python ---
    where python
    echo --- where tee.exe ---
    where tee.exe
)> "%ENV_LOG%" 2>&1

type "%ENV_LOG%"

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
set "ICO_HLS_PROJECT_ROOT=%PROJECT_ROOT%"
set "ICO_HLS_SOURCE_DIR=%SCRIPT_DIR%"

echo.
echo [1/1] Running Vitis HLS flow...
cmd /c vitis_hls -f run_hls.tcl
set "HLS_EXIT=%ERRORLEVEL%"

if not "%HLS_EXIT%"=="0" (
    echo.
    echo [WARN] Vitis HLS flow returned exit code %HLS_EXIT%.
)

if /I not "%MODE%"=="csim" (
    echo.
    echo [Parse] Exporting latest report summary...
    python parse_hls_report.py --project "%PROJECT_ROOT%\%PROJECT%" --solution "%SOLUTION%" --top "%TOP%"
    if errorlevel 1 (
        echo [WARN] Report parser failed. Raw reports still exist in %PROJECT%\%SOLUTION%\syn\report
    )
)

echo.
echo [DONE] Flow finished.

popd
exit /b %HLS_EXIT%
