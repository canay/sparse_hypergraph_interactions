$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

Write-Host "Running review-oriented replication checks."
.\run_smoke.ps1
Write-Host "Review-oriented integrity checks completed. No experiment was executed."
Write-Host "Historical scripts are under code/. Final calibrated source and commands are under replication_package/calibrated_runs/."
