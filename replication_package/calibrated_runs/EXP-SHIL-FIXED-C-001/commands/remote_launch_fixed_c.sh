#!/usr/bin/env bash
set -u -o pipefail

RUN_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$RUN_DIR"

RUN_ID="2026-07-27_codex_vps_fixed-c-cost-sensitivity"
PAPER_ID="sparse_hypergraph_interactions"
PYTHON_BIN="/home/ubuntu/experiments/sparse_hypergraph_interactions/2026-07-17_codex_vps_stability-calibrated-shil/.venv/bin/python"
STATUS_FILE="REMOTE_FIXED_C_STATUS.txt"
PID_FILE="fixed_c_run.pid"
STARTED_AT="$(date --iso-8601=seconds)"

mkdir -p logs outputs analysis config/partitioned_full_real
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
    echo "fixed_c=1.0"
    echo "per_seed_hard_timeout=24h"
  } > "$STATUS_FILE"
}

write_status "preflight" "pending" "pending"
notify "$PAPER_ID: $RUN_ID preflight basladi. Fixed C=1.0; 10 kilitli Dry Bean seed; en fazla 3 CPU."

if [[ ! -x "$PYTHON_BIN" ]]; then
  write_status "failed_missing_verified_python" "127" "$(date --iso-8601=seconds)"
  notify "$PAPER_ID: $RUN_ID baslatilamadi; dogrulanmis Python ortami bulunamadi."
  exit 127
fi

"$PYTHON_BIN" -m py_compile \
  src/predecessor_sc_shil_experiment.py \
  src/fixed_c_experiment.py \
  src/run_partitioned_fixed_c.py \
  src/analyze_fixed_c.py \
  > logs/py_compile_cli.log 2>&1
rc=$?
printf '%s\n' "$rc" > logs/py_compile_exit_code.txt
if [[ "$rc" -ne 0 ]]; then
  write_status "failed_py_compile" "$rc" "$(date --iso-8601=seconds)"
  notify "$PAPER_ID: $RUN_ID Python derleme kontrolunde durdu. Exit=$rc."
  exit "$rc"
fi

if [[ ! -f outputs/smoke-vps/run_summary.json ]]; then
  timeout --signal=TERM --kill-after=2m 20m \
    taskset -c 0 "$PYTHON_BIN" src/fixed_c_experiment.py \
    --config config/full_config.json \
    --mode smoke \
    --output outputs/smoke-vps \
    > logs/smoke_vps_cli.log 2>&1
  rc=$?
  printf '%s\n' "$rc" > logs/smoke_vps_exit_code.txt
  if [[ "$rc" -ne 0 ]]; then
    write_status "failed_remote_smoke" "$rc" "$(date --iso-8601=seconds)"
    notify "$PAPER_ID: $RUN_ID remote smoke turunda durdu. Exit=$rc."
    exit "$rc"
  fi
fi

"$PYTHON_BIN" -c "import pandas as pd; d=pd.read_csv('outputs/smoke-vps/base_fit_diagnostics.csv'); assert d.groupby('base_method').size().to_dict()=={'l1_fixed':4,'shil':4}; f=d[d.base_method=='l1_fixed']; assert (f.selected_c==1.0).all() and (f.inner_val_rows==0).all()" \
  > logs/smoke_invariant_cli.log 2>&1
rc=$?
printf '%s\n' "$rc" > logs/smoke_invariant_exit_code.txt
if [[ "$rc" -ne 0 ]]; then
  write_status "failed_remote_smoke_invariant" "$rc" "$(date --iso-8601=seconds)"
  notify "$PAPER_ID: $RUN_ID remote smoke invariant kontrolunde durdu. Exit=$rc."
  exit "$rc"
fi

write_status "running_partitioned_full_real" "pending" "pending"
notify "$PAPER_ID: $RUN_ID uzun run basladi. 10 seed; sabit-C L1 ve contemporary SC-SHIL; atomik checkpoint."

nice -n 10 ionice -c2 -n7 \
  "$PYTHON_BIN" src/run_partitioned_fixed_c.py \
  --config config/full_config.json \
  --experiment-script src/fixed_c_experiment.py \
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
  notify "$PAPER_ID: $RUN_ID partitioned run durdu. Exit=$rc; tamamlanan seed checkpointleri korundu."
  exit "$rc"
fi

"$PYTHON_BIN" src/analyze_fixed_c.py \
  --combined outputs/full-real-combined \
  --historical inputs/historical_tuned_metrics.csv \
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
notify "$PAPER_ID: $RUN_ID tamamlandi ve analiz edildi. Yerel indirme ve bagimsiz dogrulama bekliyor."
exit 0
