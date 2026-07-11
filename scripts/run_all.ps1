# Runs analytics (headless) + dashboard together for local demo.
# Ctrl+C to stop both.

$ErrorActionPreference = "Stop"

Write-Host "Starting gaze-analytics + dashboard..." -ForegroundColor Cyan

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

# Activate venv if present
if (Test-Path ".\.venv\Scripts\Activate.ps1") {
    . .\.venv\Scripts\Activate.ps1
}

$analytics = Start-Job -Name "gaze-analytics" -ScriptBlock {
    param($r)
    Set-Location $r
    python -m gaze_analytics --headless --device AUTO
} -ArgumentList $root

$dashboard = Start-Job -Name "gaze-dashboard" -ScriptBlock {
    param($r)
    Set-Location $r
    streamlit run dashboard/app.py --server.headless true
} -ArgumentList $root

Write-Host "Analytics job id: $($analytics.Id)  Dashboard job id: $($dashboard.Id)"
Write-Host "Dashboard will be at http://localhost:8501"
Write-Host "Press Ctrl+C to stop."

try {
    while ($true) {
        Receive-Job -Job $analytics
        Receive-Job -Job $dashboard
        Start-Sleep -Seconds 1
    }
} finally {
    Stop-Job -Job $analytics, $dashboard -ErrorAction SilentlyContinue
    Remove-Job -Job $analytics, $dashboard -Force -ErrorAction SilentlyContinue
    Write-Host "Stopped." -ForegroundColor Yellow
}
