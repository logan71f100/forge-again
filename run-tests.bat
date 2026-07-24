@echo off
REM Pre-merge test harness -- run before folding `testing` into `main`.
REM Usage:  run-tests.bat            all checks
REM         run-tests.bat --static   fast static checks only
setlocal
cd /d "%~dp0"

set PY=venv\Scripts\python.exe
if not exist "%PY%" set PY=python\python.exe
if not exist "%PY%" set PY=python

"%PY%" tests\run_tests.py %*
exit /b %ERRORLEVEL%
