@echo off
setlocal
cd /d "%~dp0"

set "INSTALLER_EXE=%~dp0jixiebijiaoben_installer.exe"
set "BOOTSTRAP_PYTHON=%~dp0_bootstrap\python\python.exe"
set "EXIT_CODE=0"

echo [JIXIEBIJIAOBEN] Installing OCR runtime...
echo [JIXIEBIJIAOBEN] This may take several minutes.
echo.

if exist "%INSTALLER_EXE%" goto run_installer
if exist "%BOOTSTRAP_PYTHON%" goto run_bootstrap
echo Missing installer executable and bootstrap Python.
set "EXIT_CODE=1"
goto finish

:run_installer
"%INSTALLER_EXE%" %*
set "EXIT_CODE=%ERRORLEVEL%"
goto finish

:run_bootstrap
set "PYTHONNOUSERSITE=1"
"%BOOTSTRAP_PYTHON%" -s -u "%~dp0install_runtime.py" %*
set "EXIT_CODE=%ERRORLEVEL%"

:finish
if not "%EXIT_CODE%"=="0" (
  echo.
  echo Install failed. Exit code: %EXIT_CODE%
  pause
  exit /b %EXIT_CODE%
)

echo.
echo Install completed.
pause
exit /b 0
