@echo off
echo Killing any stuck Python processes that might be holding the camera...
taskkill /F /IM python.exe 2>nul
taskkill /F /IM pythonw.exe 2>nul
echo Done! Camera should be free now.
pause