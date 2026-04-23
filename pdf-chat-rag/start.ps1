# Document Parser - Start Backend, Frontend, and open Browser
# Run from: Document Parser folder (npm start) or pdf-chat-rag folder (.\start.ps1)

$ErrorActionPreference = "Stop"
$ProjectRoot = $PSScriptRoot
if (-not (Test-Path "$ProjectRoot\backend\main.py")) {
    # If run from "Document Parser", switch to pdf-chat-rag
    $ProjectRoot = Join-Path (Get-Location) "pdf-chat-rag"
    if (-not (Test-Path "$ProjectRoot\backend\main.py")) {
        Write-Host "Error: Run this script from 'Document Parser' or 'pdf-chat-rag' folder." -ForegroundColor Red
        exit 1
    }
}

function Get-PidOnPort($port) {
    $line = netstat -ano | findstr "LISTENING" | findstr ":$port "
    if ($line) {
        $parts = $line -split '\s+'
        return $parts[-1]
    }
    return $null
}

function Stop-ProcessOnPort($port, $name) {
    $procId = Get-PidOnPort $port
    if ($procId) {
        Write-Host "Stopping existing process on port $port (PID $procId)..." -ForegroundColor Yellow
        taskkill /PID $procId /F 2>$null
        Start-Sleep -Seconds 1
    }
}

Write-Host "Document Parser - Starting backend and frontend..." -ForegroundColor Cyan
Stop-ProcessOnPort 8000 "backend"
Stop-ProcessOnPort 3000 "frontend"

$backendPath = Join-Path $ProjectRoot "backend"
$frontendPath = Join-Path $ProjectRoot "frontend"

Write-Host "Starting backend (FastAPI) on http://127.0.0.1:8000 ..." -ForegroundColor Green
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location '$backendPath'; .\.venv\Scripts\Activate.ps1; python main.py"
)

Write-Host "Starting frontend (Next.js) on http://127.0.0.1:3000 ..." -ForegroundColor Green
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "Set-Location '$frontendPath'; npm run dev"
)

Write-Host "Waiting for servers to be ready..." -ForegroundColor Cyan
$maxWait = 45
$waited = 0
$backendOk = $false
$frontendOk = $false

while ($waited -lt $maxWait) {
    if (-not $backendOk) {
        try {
            $r = Invoke-WebRequest -Uri "http://127.0.0.1:8000/health" -UseBasicParsing -TimeoutSec 2 -ErrorAction SilentlyContinue
            if ($r.StatusCode -eq 200) { $backendOk = $true }
        } catch {}
    }
    if (-not $frontendOk) {
        try {
            $r = Invoke-WebRequest -Uri "http://127.0.0.1:3000" -UseBasicParsing -TimeoutSec 2 -ErrorAction SilentlyContinue
            if ($r.StatusCode -eq 200) { $frontendOk = $true }
        } catch {}
    }
    if ($backendOk -and $frontendOk) { break }
    Start-Sleep -Seconds 2
    $waited += 2
}

if (-not $backendOk) { Write-Host "Backend may still be starting. Check the backend window." -ForegroundColor Yellow }
if (-not $frontendOk) { Write-Host "Frontend may still be starting. Check the frontend window." -ForegroundColor Yellow }

Write-Host "Opening browser at http://127.0.0.1:3000" -ForegroundColor Green
Start-Process "http://127.0.0.1:3000"

Write-Host ""
Write-Host "Done. Two terminal windows are open (backend and frontend). Close them when you are done." -ForegroundColor Cyan
Write-Host "App URL: http://127.0.0.1:3000" -ForegroundColor Cyan
