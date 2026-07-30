#!/usr/bin/env bash
set -euo pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$RUN_DIR"

if [[ -f partitioned_recovery.pid ]]; then
  existing_pid="$(tr -d '\r\n' < partitioned_recovery.pid)"
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "recovery_already_running pid=$existing_pid" >&2
    exit 2
  fi
fi

if [[ -f outputs/full-real-combined/run_summary.json ]]; then
  echo "combined_output_already_complete_refusing_duplicate" >&2
  exit 3
fi

mkdir -p logs
nohup bash remote_launch_partitioned_full_real.sh > logs/launcher_nohup.log 2>&1 &
launcher_pid=$!
printf '%s\n' "$launcher_pid" > partitioned_recovery.pid
echo "started pid=$launcher_pid"
