@echo off
setlocal

set CXX=g++
set CXXFLAGS=-std=c++11 -O2 -I. -Ifull_stage1_legacy -Ifrontend_dual_feature -Ifeature_maba -Ipost_maba -Itemporal_r1 -Wall -Wextra -Wno-unknown-pragmas
set DEFS=

if not "%IFAN_BUILD_DEFS%"=="" set "DEFS=%IFAN_BUILD_DEFS%"

if "%1"=="frontend" goto build_frontend
if "%1"=="full" goto build_full
if "%1"=="maba" goto build_maba
if "%1"=="post" goto build_post
if "%1"=="temporal" goto build_temporal

:build_frontend
set TARGET=test_ifan_dual_frontend.exe

%CXX% %CXXFLAGS% %DEFS% frontend_dual_feature\ifan_dual_frontend.cpp full_stage1_legacy\ifan_stage1_engines.cpp frontend_dual_feature\test_ifan_dual_frontend.cpp -o %TARGET%
if errorlevel 1 exit /b 1

echo Built %TARGET%
goto done

:build_full
set TARGET=test_ifan_stage1.exe
%CXX% %CXXFLAGS% %DEFS% full_stage1_legacy\ifan_stage1.cpp full_stage1_legacy\ifan_stage1_engines.cpp full_stage1_legacy\test_ifan_stage1.cpp -o %TARGET%
if errorlevel 1 exit /b 1

echo Built %TARGET%
goto done

:build_maba
set TARGET=test_feature_maba.exe
%CXX% %CXXFLAGS% feature_maba\ifan_stage1_maba.cpp feature_maba\test_feature_maba.cpp -o %TARGET%
if errorlevel 1 exit /b 1

echo Built %TARGET%
goto done

:build_post
set TARGET=test_post_maba.exe
%CXX% %CXXFLAGS% post_maba\ifan_stage1_post.cpp post_maba\test_post_maba.cpp -o %TARGET%
if errorlevel 1 exit /b 1

echo Built %TARGET%
goto done

:build_temporal
set TARGET=test_ifan_temporal_r1.exe
%CXX% %CXXFLAGS% temporal_r1\ifan_temporal_r1.cpp temporal_r1\test_ifan_temporal_r1.cpp -o %TARGET%
if errorlevel 1 exit /b 1

echo Built %TARGET%

:done
endlocal
