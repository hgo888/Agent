@echo off
chcp 65001 >nul
cd /d "%~dp0"

echo ============================================
echo   Multimodal Agentic RAG - System Agent
echo ============================================
echo.

if not exist ".venv312\Scripts\python.exe" (
    echo [ERROR] Python venv not found: .venv312\Scripts\python.exe
    pause
    exit /b 1
)

.venv312\Scripts\python.exe System_Agent.py

echo.
echo [Process exited]
pause
