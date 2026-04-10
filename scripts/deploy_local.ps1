$ErrorActionPreference = "Stop"

docker compose up --build -d
Write-Host "Local deployment started."
Write-Host "Dashboard: http://localhost:8501"
