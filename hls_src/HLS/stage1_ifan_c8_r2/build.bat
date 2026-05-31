@echo off
setlocal

set CXX=g++
set CXXFLAGS=-std=c++11 -O2 -I. -Wall -Wextra -Wno-unknown-pragmas

if "%1"=="maba" goto build_maba
if "%1"=="post" goto build_post

set TARGET=test_ifan_stage1.exe

%CXX% %CXXFLAGS% ifan_stage1.cpp ifan_stage1_engines.cpp test_ifan_stage1.cpp -o %TARGET%
if errorlevel 1 exit /b 1

echo Built %TARGET%
goto done

:build_maba
set TARGET=test_feature_maba.exe
%CXX% %CXXFLAGS% ifan_stage1_maba.cpp test_feature_maba.cpp -o %TARGET%
if errorlevel 1 exit /b 1

echo Built %TARGET%
goto done

:build_post
set TARGET=test_post_maba.exe
%CXX% %CXXFLAGS% ifan_stage1_post.cpp test_post_maba.cpp -o %TARGET%
if errorlevel 1 exit /b 1

echo Built %TARGET%

:done
endlocal
