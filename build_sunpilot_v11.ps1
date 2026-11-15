$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$python = '.\.venv\Scripts\python.exe'
if (!(Test-Path $python)) { throw 'Missing .venv. Please prepare the environment first.' }
& $python -m pip install -r requirements-dev.txt
if ($LASTEXITCODE -ne 0) { throw 'Required dev dependencies could not be installed.' }
& $python -m pytest -q .\tests
if ($LASTEXITCODE -ne 0) { throw 'The test suite failed; refusing to create a release binary.' }
& $python .\scripts\verify_release.py
if ($LASTEXITCODE -ne 0) { throw 'Scientific release gate failed. Recompute corrected metrics before building.' }
& $python -m pip install pyinstaller tzdata
if ($LASTEXITCODE -ne 0) { throw 'PyInstaller installation failed.' }
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
if ($LASTEXITCODE -ne 0) { throw 'PyInstaller failed.' }
if (!(Test-Path '.\dist\SunPilotAI\SunPilotAI.exe')) { throw 'Missing .exe after build.' }
Write-Host 'v1.1 scientific build succeeded: dist\SunPilotAI\SunPilotAI.exe' -ForegroundColor Green
Write-Host 'Do GUI and Task Scheduler smoke tests BEFORE publishing a new release.' -ForegroundColor Cyan
