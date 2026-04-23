@echo off
cd /d "%~dp0"
call .venv\Scripts\activate.bat
echo Starting backend on http://127.0.0.1:8000 ...
python main.py
pause
