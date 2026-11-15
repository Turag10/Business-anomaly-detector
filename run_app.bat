@echo off
title FinShield - Autonomous Financial Anomaly Surveillance
cls
echo ===================================================================
echo     FinShield - Autonomous Financial Anomaly Surveillance
echo ===================================================================
echo.
echo  [1] Launch Interactive Web Dashboard (Streamlit - Recommended)
echo  [2] Launch Production REST API (FastAPI + Swagger Docs)
echo  [3] Launch BOTH (Web Dashboard + REST API)
echo.
set /p choice="Select an option (1, 2, or 3) [Default is 1]: "

if "%choice%"=="" set choice=1

if not exist ".venv\Scripts\python.exe" (
    echo Error: Python virtual environment (.venv) not found.
    pause
    exit /b
)

if "%choice%"=="1" (
    echo.
    echo Starting FinShield Web Dashboard at http://localhost:8501 ...
    start "" "http://localhost:8501"
    ".venv\Scripts\streamlit.exe" run app.py --server.port=8501
    pause
    exit /b
)

if "%choice%"=="2" (
    echo.
    echo Starting FinShield REST API at http://localhost:8000/docs ...
    start "" "http://localhost:8000/docs"
    ".venv\Scripts\uvicorn.exe" api:app --reload --port 8000
    pause
    exit /b
)

if "%choice%"=="3" (
    echo.
    echo Starting REST API in background...
    start "FinShield REST API" ".venv\Scripts\uvicorn.exe" api:app --port 8000
    timeout /t 2 /nobreak >nul
    echo Starting Web Dashboard...
    start "" "http://localhost:8501"
    start "" "http://localhost:8000/docs"
    ".venv\Scripts\streamlit.exe" run app.py --server.port=8501
    pause
    exit /b
)

echo Invalid choice. Starting default Web Dashboard...
start "" "http://localhost:8501"
".venv\Scripts\streamlit.exe" run app.py --server.port=8501
pause
