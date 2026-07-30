#!/usr/bin/env bash
set -u -o pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$RUN_DIR"

RUN_ID="2026-07-18_codex_vps_stability-calibrated-shil-recovery"
PAPER_ID="sparse_hypergraph_interactions"
PYTHON_BIN="/home/ubuntu/experiments/sparse_hypergraph_interactions/2026-07-17_codex_vps_stability-calibrated-shil/.venv/bin/python"
STATUS_FILE="REMOTE_RECOVERY_STATUS.txt"
PID_FILE="partitioned_recovery.pid"
STARTED_AT="$(date --iso-8601=seconds)"

mkdir -p logs env outputs analysis config/partitioned_full_real
printf '%s\n' "$$" > "$PID_FILE"

notify() {
  if command -v mesaj >/dev/null 2>&1; then
    timeout 15s mesaj "$1" >/dev/null 2>&1 || true
  fi
}

write_status() {
  local state="$1"
  local exit_code="$2"
  local finished_at="$3"
  {
    echo "run_id=$RUN_ID"
    echo "pid=$$"
    echo "started_at=$STARTED_AT"
    echo "finished_at=$finished_at"
    echo "exit_code=$exit_code"
    echo "status=$state"
    echo "execution=10_locked_seeds_partitioned_max_3_workers"
    echo "per_seed_hard_timeout=24h"
  } > "$STATUS_FILE"
}

write_status "preflight" "pending" "pending"
notify "$PAPER_ID: $RUN_ID preflight basladi. Hedef=10 kilitli Dry Bean seed; en fazla 3 CPU; seed basina 24h timeout."

if [[ ! -x "$PYTHON_BIN" ]]; then
  write_status "failed_missing_verified_python" "127" "$(date --iso-8601=seconds)"
  notify "$PAPER_ID: $RUN_ID baslatilamadi; dogrulanmis Python ortami bulunamadi."
  exit 127
fi

"$PYTHON_BIN" -m unittest discover -s tests -p 'test_*.py' -v \
  > logs/unit_tests_cli.log 2>&1
rc=$?
printf '%s\n' "$rc" > logs/unit_tests_exit_code.txt
if [[ "$rc" -ne 0 ]]; then
  write_status "failed_unit_tests" "$rc" "$(date --iso-8601=seconds)"
  notify "$PAPER_ID: $RUN_ID unit testte durdu. Exit=$rc."
  exit "$rc"
fi

if [[ ! -f outputs/smoke-recovery/run_summary.json ]]; then
  timeout --signal=TERM --kill-after=2m 20m \
    taskset -c 0 "$PYTHON_BIN" src/sc_shil_experiment.py \
    --config config/full_config.json \
    --mode smoke \
    --output outputs/smoke-recovery \
    > logs/smoke_recovery_cli.log 2>&1
  rc=$?
  printf '%s\n' "$rc" > logs/smoke_recovery_exit_code.txt
  if [[ "$rc" -ne 0 ]]; then
    write_status "failed_remote_smoke" "$rc" "$(date --iso-8601=seconds)"
    notify "$PAPER_ID: $RUN_ID remote smoke turunda durdu. Exit=$rc."
    exit "$rc"
  fi
fi

write_status "running_partitioned_full_real" "pending" "pending"
notify "$PAPER_ID: $RUN_ID uzun run basladi. 10 seed; en fazla 3 eszamanli is; her seed atomik checkpoint."

nice -n 10 ionice -c2 -n7 \
  "$PYTHON_BIN" src/run_partitioned_full_real.py \
  --config config/full_config.json \
  --experiment-script src/sc_shil_experiment.py \
  --config-dir config/partitioned_full_real \
  --output-root outputs/full-real-partitions \
  --combined-output outputs/full-real-combined \
  --workers 3 \
  --cores 0,1,2 \
  > logs/partitioned_full_real_cli.log 2>&1
rc=$?
printf '%s\n' "$rc" > logs/partitioned_full_real_exit_code.txt
if [[ "$rc" -ne 0 ]]; then
  write_status "failed_partitioned_full_real" "$rc" "$(date --iso-8601=seconds)"
  notify "$PAPER_ID: $RUN_ID partitioned run hata/timeout ile durdu. Exit=$rc; tamamlanan seed checkpointleri korundu."
  exit "$rc"
fi

"$PYTHON_BIN" src/analyze_results.py \
  --input outputs/full-real-combined \
  --output analysis/full-real-combined \
  > logs/full_real_analysis_cli.log 2>&1
rc=$?
printf '%s\n' "$rc" > logs/full_real_analysis_exit_code.txt
if [[ "$rc" -ne 0 ]]; then
  write_status "completed_compute_analysis_failed" "$rc" "$(date --iso-8601=seconds)"
  notify "$PAPER_ID: $RUN_ID hesap tamam, analiz basarisiz. Exit=$rc."
  exit "$rc"
fi

write_status "completed_and_analyzed_remote_unverified_local" "0" "$(date --iso-8601=seconds)"
notify "$PAPER_ID: $RUN_ID tamamlandi ve analiz edildi. Yerel indirme/dogrulama bekliyor."
exit 0
