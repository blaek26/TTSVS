@echo off
rem Launch the TTSVS web UI (http://127.0.0.1:7860)
cd /d "%~dp0.."
if not exist .venv\Scripts\ttsvs.exe (
  echo Not installed yet. Run scripts\install.ps1 first.
  exit /b 1
)
.venv\Scripts\ttsvs.exe ui %*
