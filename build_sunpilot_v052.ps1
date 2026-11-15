$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$python = '.\.venv\Scripts\python.exe'
if (!(Test-Path $python)) { throw 'Virtual environment not found. Please create .venv first.' }
& $python -m pip install pyinstaller tzdata
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& $python -m PyInstaller `
  --onedir `
  --windowed `
  --clean `
  --noconfirm `
  --name SunPilotAI `
  --add-data 'data\klippan_hourly_2022_2025.csv:data' `
  --add-data 'models\sunpilot_v1.joblib:models' `
  --add-data 'results\scheduler_metrics.json:results' `
  --add-data 'results\metrics.json:results' `
  --collect-data tzdata `
  --hidden-import matplotlib.backends.backend_qtagg `
  --exclude-module gradio `
  --exclude-module gradio_client `
  --exclude-module plotly `
  --exclude-module PySide6.QtWebEngineCore `
  --exclude-module PySide6.QtWebEngineWidgets `
  --exclude-module PySide6.QtPdf `
  --exclude-module PySide6.QtPdfWidgets `
  .\sunpilot_desktop_v052.py
if ($LASTEXITCODE -ne 0) { throw 'PyInstaller build failed.' }
Write-Host 'SunPilot .exe created in dist\SunPilotAI\SunPilotAI.exe' -ForegroundColor Green
