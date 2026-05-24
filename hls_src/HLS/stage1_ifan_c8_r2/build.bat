@echo off
setlocal

set CXX=g++
set CXXFLAGS=-std=c++11 -O2 -I. -Wall -Wextra -Wno-unknown-pragmas
set TARGET=test_ifan_stage1.exe

%CXX% %CXXFLAGS% ifan_stage1.cpp ifan_stage1_engines.cpp test_ifan_stage1.cpp -o %TARGET%
if errorlevel 1 exit /b 1

echo Built %TARGET%
endlocal
