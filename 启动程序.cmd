@echo off
setlocal
cd /d "%~dp0"

set "LAUNCHER_EXE=%~dp0jixiebijiaoben_launcher.exe"
set "RUNTIME_PYTHON=%~dp0.runtime_env\Scripts\python.exe"

if exist "%LAUNCHER_EXE%" goto start_exe
if exist "%RUNTIME_PYTHON%" goto run_runtime
where py >nul 2>nul
if %ERRORLEVEL%==0 goto run_py
where python >nul 2>nul
if %ERRORLEVEL%==0 goto run_system_python
echo Python was not found.
pause
exit /b 1

:start_exe
start "" "%LAUNCHER_EXE%"
exit /b 0

:run_runtime
set "PYTHONNOUSERSITE=1"
"%RUNTIME_PYTHON%" -s -u "%~dp0main.py"
exit /b %ERRORLEVEL%

:run_py
set "PYTHONNOUSERSITE=1"
py -3 -s -u "%~dp0main.py"
exit /b %ERRORLEVEL%

:run_system_python
set "PYTHONNOUSERSITE=1"
python -s -u "%~dp0main.py"
exit /b %ERRORLEVEL%
