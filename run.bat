@echo off
echo ========================================
echo   VitalFlow AI - Setup and Run
echo ========================================

cd /d "%~dp0"

if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat

echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Starting VitalFlow AI server...
echo API will be available at: http://localhost:8000
echo API Docs at:              http://localhost:8000/docs
echo.

cd /d "%~dp0backend"
python main.py

pause