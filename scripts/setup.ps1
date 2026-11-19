$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")
py -3 -m venv .venv
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
& ".\.venv\Scripts\python.exe" -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
if (-not (Test-Path ".env")) { Copy-Item ".env.example" ".env" }
