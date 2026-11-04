@echo off
cd /d "%~dp0"
python classify_guobiao.py %*
if errorlevel 1 pause
