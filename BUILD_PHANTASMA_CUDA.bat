@echo off
setlocal EnableExtensions
title phantasma.cpp - CUDA BUILD
cd /d "%~dp0"

echo.
echo  ===============================================================
echo    phantasma.cpp  //  CUDA RELEASE BUILD
echo  ===============================================================
echo.

cmake --build build-cuda-vs --config Release --target llama-server --parallel 1
if errorlevel 1 goto :failed

echo.
echo [phantasma] CUDA Release build OK.
echo [phantasma] binary: %CD%\build-cuda-vs\bin\Release\llama-server.exe
pause
exit /b 0

:failed
echo.
echo [ERROR] Build failed. Read the compiler output above.
pause
exit /b 1
