# TTSVS installer for Windows (PowerShell). Run from anywhere:  powershell -ExecutionPolicy Bypass -File scripts\install.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$py = $null
foreach ($v in "3.11", "3.12", "3.10") {
    try { & py "-$v" --version *> $null; if ($LASTEXITCODE -eq 0) { $py = @("py", "-$v"); break } } catch {}
}
if (-not $py) { throw "Python 3.10-3.12 not found. Install it from python.org first." }

if (-not (Test-Path .venv)) { & $py[0] $py[1..($py.Length-1)] -m venv .venv }
$venvPy = ".venv\Scripts\python.exe"
& $venvPy -m pip install --upgrade pip

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host "==> NVIDIA GPU detected: installing CUDA PyTorch"
    & $venvPy -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
} else {
    Write-Host "==> No NVIDIA GPU detected: installing CPU PyTorch"
    & $venvPy -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
}

Write-Host "==> Installing TTSVS and Chatterbox"
& $venvPy -m pip install -e ".[chatterbox,book]"
Write-Host "Done. Check with: .venv\Scripts\ttsvs doctor ; start with scripts\run.bat"
