@echo off
REM INFINITE GENERATION RECOVERY SCRIPT
REM Use this script to quickly restore the working infinite generation version

echo ========================================
echo  INFINITE GENERATION RECOVERY SCRIPT
echo ========================================
echo.
echo This script will restore the working infinite generation version
echo from the golden master backup.
echo.

set /p confirm="Are you sure you want to restore? (y/N): "
if /i not "%confirm%"=="y" (
    echo Recovery cancelled.
    pause
    exit /b
)

echo.
echo Backing up current version...
copy "conscious_cursor_interface_PURE_MARKOV.py" "conscious_cursor_interface_PURE_MARKOV_BACKUP_BEFORE_RECOVERY_%date:~-4,4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%.py" >nul 2>&1

echo Restoring golden master version...
copy "conscious_cursor_interface_PURE_MARKOV_GOLDEN_MASTER.py" "conscious_cursor_interface_PURE_MARKOV.py" >nul 2>&1

if %errorlevel% equ 0 (
    echo.
    echo ✅ SUCCESS: Infinite generation version restored!
    echo.
    echo The working infinite generation system has been restored.
    echo You can now run: python conscious_cursor_interface_PURE_MARKOV.py
    echo.
    echo Key features restored:
    echo - Infinite Markov chain generation
    echo - Dead-end state recovery
    echo - Robust error handling
    echo - Never-stopping timer system
    echo.
) else (
    echo.
    echo ❌ ERROR: Failed to restore backup!
    echo Please manually copy the golden master file.
    echo.
)

pause
