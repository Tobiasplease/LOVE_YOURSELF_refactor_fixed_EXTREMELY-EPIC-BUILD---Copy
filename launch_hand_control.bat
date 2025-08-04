@echo off
echo 🚀 Starting Standalone Hand Control System...
echo.
echo 📊 Features:
echo    - Simplified UI (no overengineered dataset management)
echo    - Wave control sliders restored
echo    - Emotion-based recording and Markov generation  
echo    - Can receive mood data from machine.py (optional)
echo.

python launch_standalone_hand_control.py

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo ❌ Error occurred. Press any key to exit...
    pause >nul
)
