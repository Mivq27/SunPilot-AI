$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$python = '.\.venv\Scripts\python.exe'
if (!(Test-Path $python)) { throw 'Missing .venv. Create the Python environment first.' }
foreach ($required in @('sunpilot_desktop_v1.py', 'src\live_planner.py', 'src\windows_scheduler.py', 'data\klippan_hourly_2022_2025.csv', 'models\sunpilot_v1.joblib', 'assets\sunpilot.ico')) {
  if (!(Test-Path $required)) { throw "Missing required file: $required" }
}
& $python -m pip install pyinstaller tzdata
if ($LASTEXITCODE -ne 0) { throw 'PyInstaller dependency installation failed.' }
& $python -m PyInstaller `
  --onedir `
  --windowed `
  --clean `
  --noconfirm `
  --name SunPilotAI `
  --icon 'assets\sunpilot.ico' `
  --add-data 'assets\sunpilot.ico:assets' `
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
  .\sunpilot_desktop_v1.py
if ($LASTEXITCODE -ne 0) { throw 'PyInstaller build failed.' }
if (!(Test-Path '.\dist\SunPilotAI\SunPilotAI.exe')) { throw 'Executable not found after build.' }
Write-Host 'Built dist\SunPilotAI\SunPilotAI.exe' -ForegroundColor Green
Write-Host 'Test that .exe before creating a GitHub release.' -ForegroundColor Cyan
