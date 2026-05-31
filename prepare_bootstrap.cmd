@echo off
setlocal
cd /d "%~dp0"

echo [BUILD] Preparing bootstrap python + get-pip.py ...

where py >nul 2>nul
if %ERRORLEVEL%==0 (
  py -3 prepare_bootstrap.py %*
  set "EXIT_CODE=%ERRORLEVEL%"
) else (
  python prepare_bootstrap.py %*
  set "EXIT_CODE=%ERRORLEVEL%"
)

if not "%EXIT_CODE%"=="0" (
  echo.
  echo Prepare failed. Exit code: %EXIT_CODE%
  pause
  exit /b %EXIT_CODE%
)

echo.
echo Bootstrap ready. Next: python build.py
pause
exit /b 0
