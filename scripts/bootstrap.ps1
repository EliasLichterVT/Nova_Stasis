$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
$NovaHome = if ($env:NOVA_LAUNCHER_HOME) { $env:NOVA_LAUNCHER_HOME } else { Join-Path $HOME ".nova-stasis" }
$Bootstrap = Join-Path $NovaHome "bootstrap"
$Uv = Join-Path $Bootstrap "uv.exe"
New-Item -ItemType Directory -Force -Path $Bootstrap | Out-Null

if (-not (Test-Path $Uv)) {
    $Installer = Join-Path ([System.IO.Path]::GetTempPath()) ("nova-uv-" + [guid]::NewGuid() + ".ps1")
    try {
        [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
        Invoke-WebRequest -UseBasicParsing -Uri "https://astral.sh/uv/0.9.30/install.ps1" -OutFile $Installer
        $env:UV_UNMANAGED_INSTALL = $Bootstrap
        powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Installer
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    } finally {
        Remove-Item $Installer -Force -ErrorAction SilentlyContinue
    }
}

& $Uv run --no-project --no-config --managed-python --python 3.12 -- python (Join-Path $Root "launcher.py") @args
exit $LASTEXITCODE
