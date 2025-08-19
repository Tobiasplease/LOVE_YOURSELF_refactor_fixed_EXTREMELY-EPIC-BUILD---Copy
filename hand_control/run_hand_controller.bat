@echo off
echo Starting Hand Controller (Standalone Mode)...
echo.

REM Activate the virtual environment
call .venv\Scripts\activate.bat

REM Change to hand_control directory
cd hand_control

REM Run the hand controller
echo 🤖 Launching Hand Controller Interface...
python hand_control_interface.py

REM Keep window open if there's an error
if errorlevel 1 (
    echo.
    echo ❌ Hand controller exited with error. Press any key to close...
    pause >nul
)
