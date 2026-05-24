@echo off
setlocal

set MODE=%1
if "%MODE%"=="" set MODE=synth

vitis_hls -f run_hls.tcl %MODE%
exit /b %ERRORLEVEL%
