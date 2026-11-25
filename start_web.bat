@echo off
title Face Recognition Web App
cd /d "%~dp0"

echo ===================================================
echo   Quality-Aware Face Recognition Web Application
echo ===================================================
echo.

:: 1. Activate Virtual Environment
if exist "venv\Scripts\activate.bat" (
    echo Activating Python virtual environment...
    call venv\Scripts\activate.bat
) else (
    echo WARNING: Virtual environment not found at venv\
)

:: 2. Install web server dependencies if needed
echo Checking web server dependencies...
pip install -r web\requirements.txt --quiet

:: 3. Check if frontend build exists
if not exist "web\frontend\dist\index.html" (
    echo.
    echo Building React frontend for production...
    cd web\frontend
    call npm install
    call npm run build
    cd ..\..
)

echo.
echo ===================================================
echo   Opening Website at: http://localhost:8000
echo   (Do NOT use 0.0.0.0:8000 in your browser)
echo ===================================================
echo.

:: Automatically open browser after 2 seconds
start "" cmd /c "timeout /t 2 /nobreak >nul && start http://localhost:8000"

:: Start Uvicorn bound to 127.0.0.1 so clicking links in terminal works directly
python -m uvicorn web.server:app --host 127.0.0.1 --port 8000 --reload

pause
