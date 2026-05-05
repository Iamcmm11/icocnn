@echo off
REM Cross3D Preprocessing - Windows Build Script
REM Author: Cross3D C Implementation
REM Date: 2024

echo ============================================
echo Cross3D Preprocessing - Build Script
echo ============================================
echo.

REM Create directories
if not exist obj mkdir obj
if not exist bin mkdir bin
if not exist output mkdir output

REM Compiler settings
set CC=D:\install\mingw64\bin\gcc.exe
set CFLAGS=-Wall -Wextra -O2 -std=c99
set LDFLAGS=-lm
set INC=-Iinclude

echo Compiling source files...

REM Compile each source file
%CC% %CFLAGS% %INC% -c src/audio_reader.c -o obj/audio_reader.o
if errorlevel 1 goto error
echo   audio_reader.c - OK

%CC% %CFLAGS% %INC% -c src/fft.c -o obj/fft.o
if errorlevel 1 goto error
echo   fft.c - OK

%CC% %CFLAGS% %INC% -c src/gcc_phat.c -o obj/gcc_phat.o
if errorlevel 1 goto error
echo   gcc_phat.c - OK

%CC% %CFLAGS% %INC% -c src/srp_map.c -o obj/srp_map.o
if errorlevel 1 goto error
echo   srp_map.c - OK

%CC% %CFLAGS% %INC% -c src/test_data.c -o obj/test_data.o
if errorlevel 1 goto error
echo   test_data.c - OK

%CC% %CFLAGS% %INC% -c src/main.c -o obj/main.o
if errorlevel 1 goto error
echo   main.c - OK

echo.
echo Linking...

REM Link all object files
%CC% obj/main.o obj/audio_reader.o obj/fft.o obj/gcc_phat.o obj/srp_map.o obj/test_data.o -o bin/cross3d_preprocess.exe %LDFLAGS%
if errorlevel 1 goto error

echo.
echo ============================================
echo Build successful!
echo Executable: bin\cross3d_preprocess.exe
echo ============================================
echo.
echo Run with: bin\cross3d_preprocess.exe
goto end

:error
echo.
echo ============================================
echo Build failed!
echo ============================================
exit /b 1

:end
