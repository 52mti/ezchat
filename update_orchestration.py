"""更新编排的唯一事实来源。

主 UI（写临时 ps1 后拉起）和客户手动更新工具（make_manual_update_kit.py）
都必须从这里取 PowerShell / bat 模板，禁止手工拷贝改写——历史上手工
拼出的 ps1 出现过 `EndsWith('\\\\')` 转义损坏，导致相对路径首字母被吃、
文件写错位置还报"更新成功"。

PowerShell 职责只做编排：等待宿主退出 -> 清理残留引擎进程 -> 用引擎
Python 拉起 update_apply.py。所有文件删除/复制逻辑只在 update_apply.py
一份实现。模板使用原始字符串 + 单反斜杠，并通过 validate_ps1_text
拒绝任何连续两个反斜杠（转义损坏的指纹）。
"""

from __future__ import annotations

from pathlib import Path


WINDOWS_UPDATE_PS1 = r"""param(
    [Parameter(Mandatory = $true)][string]$PackageRoot,
    [Parameter(Mandatory = $true)][string]$ArchivePath,
    [Parameter(Mandatory = $true)][int]$PidToWait,
    [int]$WaitTimeoutSeconds = 120
)

$ErrorActionPreference = 'Stop'

function Show-Message([string]$message, [bool]$isError) {
    try {
        $shell = New-Object -ComObject WScript.Shell
        $flags = 64
        if ($isError) { $flags = 16 }
        $null = $shell.Popup($message, 60, 'jixiebijiaoben updater', $flags)
    } catch {
        Write-Host $message
    }
}

function Wait-ForProcessExit([int]$processId, [int]$timeoutSeconds) {
    if ($processId -le 0) {
        return
    }
    $deadline = (Get-Date).AddSeconds($timeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
        if ($null -eq $process) {
            return
        }
        Start-Sleep -Milliseconds 250
    }
    throw 'Timed out waiting for the running program to exit.'
}

function Stop-AppPythonProcesses([string]$rootPath) {
    $prefixes = @(
        ([System.IO.Path]::GetFullPath((Join-Path $rootPath 'engine'))).TrimEnd('\'),
        ([System.IO.Path]::GetFullPath((Join-Path $rootPath 'runtime'))).TrimEnd('\')
    )
    $targets = Get-CimInstance Win32_Process -Filter "Name='python.exe' OR Name='pythonw.exe'" -ErrorAction SilentlyContinue
    foreach ($process in $targets) {
        if (-not $process.ExecutablePath) { continue }
        $executable = $process.ExecutablePath.ToLower()
        foreach ($prefix in $prefixes) {
            if ($executable.StartsWith($prefix.ToLower() + '\')) {
                Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
                break
            }
        }
    }
}

try {
    $packageRoot = ([System.IO.Path]::GetFullPath($PackageRoot)).TrimEnd('\')
    $manifestPath = Join-Path $packageRoot 'package_manifest.json'
    $launcherPath = Join-Path $packageRoot 'jixiebijiaoben_launcher.exe'
    if (-not (Test-Path -LiteralPath $manifestPath) -and -not (Test-Path -LiteralPath $launcherPath)) {
        throw ('PackageRoot does not look like an installation directory: ' + $packageRoot)
    }
    Wait-ForProcessExit -processId $PidToWait -timeoutSeconds $WaitTimeoutSeconds
    Stop-AppPythonProcesses -rootPath $packageRoot

    $enginePython = Join-Path $packageRoot 'engine\app\python.exe'
    if (-not (Test-Path -LiteralPath $enginePython)) {
        $enginePython = Join-Path $packageRoot 'runtime\app\python.exe'
    }
    if (-not (Test-Path -LiteralPath $enginePython)) {
        throw ('Engine python not found in ' + $packageRoot + '. Extract the full package over the install directory manually.')
    }

    & $enginePython -s (Join-Path $packageRoot 'update_apply.py') --package-root $packageRoot --archive $ArchivePath --pid 0 --quiet
    if ($LASTEXITCODE -ne 0) {
        Show-Message ('Update failed. See runtime\errors\update_apply_error.txt') $true
        exit $LASTEXITCODE
    }
    Show-Message 'Update finished. Please start the app again.' $false
    exit 0
} catch {
    Show-Message ('Update failed: ' + $_.Exception.Message) $true
    exit 1
}
"""


# 客户机 cmd（PS 1.1 时代兼容）：仅 ASCII、CRLF，见 make_manual_update_kit.py。
MANUAL_UPDATE_BAT = r"""@echo off
setlocal
cd /d "%~dp0"

set "ARCHIVE="
for %%F in ("jixiebijiaoben_full_*.zip") do set "ARCHIVE=%%~fF"
if not defined ARCHIVE for %%F in ("jixiebijiaoben_update_*.zip") do set "ARCHIVE=%%~fF"
if not defined ARCHIVE goto NOZIP

set "ROOT=%~dp0"
if not exist "%ROOT%package_manifest.json" if not exist "%ROOT%jixiebijiaoben_launcher.exe" goto FINDCHILD
goto APPLY

:FINDCHILD
if not exist "%ROOT%jixiebijiaoben\package_manifest.json" if not exist "%ROOT%jixiebijiaoben\jixiebijiaoben_launcher.exe" goto NOROOT
set "ROOT=%~dp0jixiebijiaoben"
goto APPLY

:APPLY
echo Package: %ARCHIVE%
echo Install root: %ROOT%
echo Closing app processes and applying update...
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0apply_update.ps1" -PackageRoot "%ROOT%" -ArchivePath "%ARCHIVE%" -PidToWait 0 -WaitTimeoutSeconds 5
if errorlevel 1 goto FAILED
echo.
echo Update finished. Please start the app again.
pause
exit /b 0

:NOZIP
echo [ERROR] put jixiebijiaoben_update_*.zip or jixiebijiaoben_full_*.zip in this folder first.
pause
exit /b 1

:NOROOT
echo [ERROR] cannot find the installed app. Put this folder INSIDE the install
echo directory (next to jixiebijiaoben_launcher.exe) and run again.
pause
exit /b 1

:FAILED
echo Update FAILED. See runtime\errors\update_apply_error.txt
pause
exit /b 1
"""


def validate_ps1_text(text: str) -> None:
    """拒绝转义损坏指纹：脚本里任何连续两个反斜杠都意味着有人把
    Python 源码字节当成了 PowerShell 文本（历史上真实发生过）。"""
    if "\\\\" in text:
        raise ValueError("PowerShell orchestration script contains double backslashes (escaping corruption)")
    for token in (
        "TrimEnd('\\')",
        "update_apply.py",
        "engine\\app\\python.exe",
    ):
        if token not in text:
            raise ValueError(f"PowerShell orchestration script is missing required token: {token}")


def write_orchestration_script(directory: Path, *, filename: str = "apply_update.ps1") -> Path:
    validate_ps1_text(WINDOWS_UPDATE_PS1)
    directory.mkdir(parents=True, exist_ok=True)
    script_path = directory / filename
    script_path.write_text(WINDOWS_UPDATE_PS1, encoding="utf-8-sig", newline="\r\n")
    return script_path


def write_manual_update_bat(directory: Path, *, filename: str = "手动更新.bat") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    bat_path = directory / filename
    bat_path.write_bytes(MANUAL_UPDATE_BAT.replace("\n", "\r\n").encode("ascii"))
    return bat_path
