@echo off
setlocal enabledelayedexpansion
title Constrained Graph Kinematics - Presentation Live Demo Launcher

:: Detect Python executable in virtual environment or system PATH
if exist "%~dp0\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=%~dp0\.venv\Scripts\python.exe"
) else (
    set "PYTHON_EXE=python"
)

:MENU
cls
echo ==============================================================================
echo   CONSTRAINED GRAPH KINEMATICS - LIVE DEMO LAUNCHER (Nov 17, 2026)
echo   Foundations of Data Science Final Project Presentation
echo ==============================================================================
echo.
echo   [1] Instant Offline Replay Demo (50%% Tool Occlusion, Guaranteed Zero-Wi-Fi)
echo   [2] Live Webcam / Continuity Camera Demo (720p HD, Press T for Tool Box)
echo   [3] Live iPhone ARKit UDP Telemetry Stream (Port 5005 + 30Hz Fallback)
echo   [4] Extreme 70%% Tool Occlusion Stress Test (15/21 Joints Masked)
echo   [5] Run Full Kinematics & Model Architecture Verification (7/7 Unit Tests)
echo   [6] Open Presentation Slides (Final_Presentation_Slides.pdf)
echo   [7] Open IEEE Project Report (Final_Project_Report.pdf)
echo   [8] Open Examiner 1-Page Cheat Sheet (docs/EXAMINER_CHEATSHEET.md)
echo   [9] Open Slide-by-Slide Speaker Script (docs/Slide_By_Slide_Presentation_Guide.md)
echo   [Q] Quit
echo.
echo ==============================================================================
set /p "CHOICE=Enter choice [1-9, default=1, Q to quit]: "

if "%CHOICE%"=="" set "CHOICE=1"
if /i "%CHOICE%"=="Q" goto :EXIT

if "%CHOICE%"=="1" (
    echo.
    echo [*] Launching Offline Replay Demo (50%% Occlusion)...
    "%PYTHON_EXE%" "%~dp0run_presentation_demo.py" --mode replay --occlusion 0.50
    pause
    goto :MENU
)

if "%CHOICE%"=="2" (
    echo.
    echo [*] Launching Live Webcam Demo (Camera 0)...
    echo [*] Controls: Press 'T' for surgical tool box, 'O' to cycle occlusion, 'Q' to quit.
    "%PYTHON_EXE%" "%~dp0run_presentation_demo.py" --mode webcam --camera 0
    pause
    goto :MENU
)

if "%CHOICE%"=="3" (
    echo.
    echo [*] Launching Mode B UDP Telemetry Receiver (Port 5005)...
    "%PYTHON_EXE%" "%~dp0run_presentation_demo.py" --mode udp --udp-port 5005 --occlusion 0.50
    pause
    goto :MENU
)

if "%CHOICE%"=="4" (
    echo.
    echo [*] Launching Extreme 70%% Occlusion Stress Test...
    "%PYTHON_EXE%" "%~dp0run_presentation_demo.py" --mode replay --occlusion 0.70
    pause
    goto :MENU
)

if "%CHOICE%"=="5" (
    echo.
    echo [*] Running Kinematics Unit Tests...
    "%PYTHON_EXE%" "%~dp0tests\test_kinematics.py"
    echo.
    echo [*] Running Mode B UDP Diagnostics...
    "%PYTHON_EXE%" "%~dp0data\iphone_parser.py" --mode test
    pause
    goto :MENU
)

if "%CHOICE%"=="6" (
    echo.
    echo [*] Opening Presentation Slides...
    start "" "%~dp0Final_Presentation_Slides.pdf"
    goto :MENU
)

if "%CHOICE%"=="7" (
    echo.
    echo [*] Opening IEEE Project Report...
    start "" "%~dp0Final_Project_Report.pdf"
    goto :MENU
)

if "%CHOICE%"=="8" (
    echo.
    echo [*] Opening Examiner 1-Page Cheat Sheet...
    start "" "%~dp0docs\EXAMINER_CHEATSHEET.md"
    goto :MENU
)

if "%CHOICE%"=="9" (
    echo.
    echo [*] Opening Slide-by-Slide Speaker Script...
    start "" "%~dp0docs\Slide_By_Slide_Presentation_Guide.md"
    goto :MENU
)

goto :MENU

:EXIT
echo Exiting launcher. Good luck with the presentation!
exit /b 0
