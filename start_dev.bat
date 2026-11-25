@echo off
title Face Recognition - Development Mode
cd /d "%~dp0"

echo ===================================================
echo   Starting Development Mode (Backend + Frontend)
echo ===================================================
echo.

:: Terminal 1: Backend
start "Backend (FastAPI)" cmd /k "call venv\Scripts\activate.bat && cd web && python -m uvicorn server:app --host 0.0.0.0 --port 8000 --reload"

:: Terminal 2: Frontend
start "Frontend (Vite)" cmd /k "cd web\frontend && npm install && npm run dev"

echo.
echo Backend running on:  http://localhost:8000
echo Frontend running on: http://localhost:3000
echo.
echo Press any key to close this launcher window...
pause >nul
