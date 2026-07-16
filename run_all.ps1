$ErrorActionPreference = "Stop"

Set-Location $PSScriptRoot

Write-Host "Running review-oriented replication checks."
.\run_smoke.ps1
Write-Host "Review-oriented integrity checks completed. No experiment was executed."
Write-Host "Full historical reruns are available through code/shil_run_experiments.py, code/shil_q1_extension.py, and code/shil_scale_stress.py."
