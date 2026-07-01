# DRIFT installer — Windows (NVIDIA / CUDA).
#
# On Windows the default PyPI torch wheel is CPU-only, so this installs the CUDA
# build from PyTorch's index first, then DRIFT. Adjust the cuXXX tag to match
# your CUDA toolkit (cu121 / cu124 / …). Run from a PowerShell in the repo root:
#   powershell -ExecutionPolicy Bypass -File scripts\install.ps1
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
  Write-Error "uv not found — install it first: https://github.com/astral-sh/uv"
  exit 1
}

$cuda = "cu121"   # change to your CUDA toolkit, e.g. cu124

Write-Host "[1/3] creating .venv (Python 3.12) ..."
uv venv --python 3.12 .venv
& .\.venv\Scripts\Activate.ps1

Write-Host "[2/3] installing CUDA torch ($cuda) + DRIFT ..."
uv pip install torch --index-url "https://download.pytorch.org/whl/$cuda"
uv pip install -e .

Write-Host "[3/3] done."
Write-Host ""
Write-Host "  installed OK   next:"
Write-Host ""
Write-Host "    .\.venv\Scripts\Activate.ps1"
Write-Host "    drift doctor            # check environment + device (expect cuda=True)"
Write-Host "    drift node              # this machine joins as a worker"
Write-Host "    drift run               # (on the head machine) auto-discovers workers"
Write-Host ""
Write-Host "  firewall: allow python.exe through Windows Defender Firewall on Private"
Write-Host "  networks, and enable Network Discovery so zeroconf can find nodes."
