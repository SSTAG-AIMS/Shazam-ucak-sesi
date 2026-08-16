@echo off
cd /d "%~dp0"
venv\Scripts\python.exe app_database.py
if errorlevel 1 pause
