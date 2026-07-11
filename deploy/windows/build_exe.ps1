# Bundle gaze-analytics into a single-file Windows .exe using PyInstaller.
# Output: release/gaze-analytics.exe

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot | Split-Path -Parent
Set-Location $root

if (-not (Test-Path ".\.venv")) {
    Write-Error "No .venv found. Create one with:  py -3.12 -m venv .venv"
}

. .\.venv\Scripts\Activate.ps1

pip install --upgrade pyinstaller
pip install -e ".[dev]"

$release = Join-Path $root "release"
New-Item -ItemType Directory -Force -Path $release | Out-Null

pyinstaller `
    --clean `
    --onefile `
    --name gaze-analytics `
    --paths src `
    --add-data "src/gaze_analytics/storage/schema.sql;gaze_analytics/storage" `
    --collect-all openvino `
    --collect-all cv2 `
    --collect-submodules pydantic_settings `
    --collect-submodules filterpy `
    --collect-submodules imagehash `
    --collect-submodules mss `
    --collect-submodules scipy `
    --hidden-import gaze_analytics `
    --hidden-import gaze_analytics.tracker `
    --hidden-import gaze_analytics.content `
    --hidden-import gaze_analytics.engagement `
    --hidden-import gaze_analytics.storage `
    --hidden-import gaze_analytics.privacy `
    --distpath $release `
    --workpath (Join-Path $env:TEMP "gaze-analytics-build") `
    --specpath (Join-Path $env:TEMP "gaze-analytics-build") `
    src/gaze_analytics/__main__.py

Write-Host "`nBuilt: $release\gaze-analytics.exe" -ForegroundColor Green
Write-Host "Copy this exe + the models/ folder to any Windows machine to run."
