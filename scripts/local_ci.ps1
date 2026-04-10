$ErrorActionPreference = "Stop"

if (-not (Test-Path ".venv")) {
  python -m venv .venv
}

.\.venv\Scripts\python -m pip install -e .[dev]
$env:PYTHONPATH = "src"
.\.venv\Scripts\python -m pytest tests -q
.\.venv\Scripts\python -m turbine_project.cli run-all --config configs/default.yaml --max-input-chunks 1 --max-assets 2
Write-Host "Local CI pipeline completed successfully."
