$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
if (Get-Command uv -ErrorAction SilentlyContinue) {
    uv sync --extra dev
} else {
    python -m venv .venv
    & .\.venv\Scripts\python.exe -m pip install -e '.[dev]'
}
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed.' }
& .\.venv\Scripts\python.exe -m rleval doctor
Write-Host 'Ready. Run: .\.venv\Scripts\rleval.exe smoke'
