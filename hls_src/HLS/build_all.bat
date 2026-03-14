@echo off
setlocal

set "ROOT=%~dp0"

echo [HLS] Build layer0
pushd "%ROOT%layer0"
make clean
if errorlevel 1 goto :err
make test_ico_conv
if errorlevel 1 goto :err
popd

echo [HLS] Build layer1
pushd "%ROOT%layer1"
make clean
if errorlevel 1 goto :err
make test_ico_conv_layer1 test_ico_conv_layer1_debug
if errorlevel 1 goto :err
popd

echo [HLS] Build complete
exit /b 0

:err
echo [HLS] Build failed
exit /b 1
