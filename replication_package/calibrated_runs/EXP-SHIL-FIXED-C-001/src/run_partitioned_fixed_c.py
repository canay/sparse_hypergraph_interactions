from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd


METHODS = ["SC-L1-fixed-C", "SC-SHIL-contemporary"]
COLLECTIONS = [
    "metrics",
    "selected_edges",
    "base_fit_diagnostics",
    "selection_frequencies",
    "calibration_path",
]
KEYS = ["dataset", "scenario", "data_seed", "split_seed"]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def derive_seed_config(base_config: dict[str, Any], seed: int, path: Path) -> None:
    derived = json.loads(json.dumps(base_config))
    derived["real_split_seeds"] = [int(seed)]
    derived["execution_partition"] = {
        "kind": "single_locked_real_split",
        "seed": int(seed),
        "base_seed_order": [
            int(value) for value in base_config["base_seed_order"]
        ],
        "scientific_protocol_changed": True,
        "methodology_change_id": "MCH-SHIL-002",
    }
    write_json(path, derived)


def validate_output_hashes(attempt: Path) -> None:
    recorded = json.loads((attempt / "output_hashes.json").read_text("utf-8"))
    for name, expected in recorded.items():
        observed = sha256_file(attempt / name)
        if observed != str(expected).upper():
            raise AssertionError(
                f"Output hash mismatch for {attempt / name}: {observed}"
            )


def validate_completed_attempt(
    attempt: Path, seed: int, expected_protocol: str
) -> dict[str, Any]:
    required = [
        "metrics.csv",
        "selected_edges.csv",
        "base_fit_diagnostics.csv",
        "selection_frequencies.csv",
        "calibration_path.csv",
        "dataset_manifest.csv",
        "run_summary.json",
        "config_resolved.json",
        "run_identity.json",
        "environment.json",
        "output_hashes.json",
    ]
    missing = [name for name in required if not (attempt / name).is_file()]
    if missing:
        raise AssertionError(f"Seed {seed} attempt missing outputs: {missing}")
    validate_output_hashes(attempt)
    summary = json.loads((attempt / "run_summary.json").read_text("utf-8"))
    config = json.loads((attempt / "config_resolved.json").read_text("utf-8"))
    identity = json.loads((attempt / "run_identity.json").read_text("utf-8"))
    if summary.get("status") != "completed" or int(summary["dataset_runs"]) != 1:
        raise AssertionError(f"Seed {seed} attempt is not completed")
    if [int(value) for value in config["real_split_seeds"]] != [int(seed)]:
        raise AssertionError(f"Seed {seed} derived config mismatch")
    if str(config["protocol_sha256"]).upper() != expected_protocol:
        raise AssertionError(f"Seed {seed} config protocol mismatch")
    if str(identity["protocol_sha256"]).upper() != expected_protocol:
        raise AssertionError(f"Seed {seed} identity protocol mismatch")
    metrics = pd.read_csv(attempt / "metrics.csv")
    if len(metrics) != len(METHODS) or set(metrics["model"]) != set(METHODS):
        raise AssertionError(f"Seed {seed} method rows are incomplete")
    if set(metrics["split_seed"].astype(int)) != {int(seed)}:
        raise AssertionError(f"Seed {seed} metric provenance mismatch")
    if metrics.duplicated(KEYS + ["model"]).any():
        raise AssertionError(f"Seed {seed} has duplicate metric keys")
    if set(metrics["base_fit_count"].astype(int)) != {80}:
        raise AssertionError(f"Seed {seed} fit-count invariant failed")
    diagnostics = pd.read_csv(attempt / "base_fit_diagnostics.csv")
    counts = diagnostics.groupby("base_method").size().to_dict()
    if counts != {"l1_fixed": 80, "shil": 80}:
        raise AssertionError(f"Seed {seed} diagnostic counts failed: {counts}")
    fixed = diagnostics[diagnostics["base_method"] == "l1_fixed"]
    if not (fixed["selected_c"].astype(float) == 1.0).all():
        raise AssertionError(f"Seed {seed} fixed C invariant failed")
    if not (fixed["inner_val_rows"].astype(int) == 0).all():
        raise AssertionError(f"Seed {seed} fixed-C inner validation is nonzero")
    return {
        "seed": int(seed),
        "attempt": str(attempt),
        "metrics_rows": len(metrics),
        "protocol_sha256": expected_protocol,
        "output_hashes_sha256": sha256_file(attempt / "output_hashes.json"),
    }


def completed_attempt(
    seed_root: Path, seed: int, expected_protocol: str
) -> Path | None:
    if not seed_root.exists():
        return None
    for attempt in sorted(seed_root.glob("attempt-*")):
        try:
            validate_completed_attempt(attempt, seed, expected_protocol)
            return attempt
        except (AssertionError, FileNotFoundError, KeyError, ValueError):
            continue
    return None


def next_attempt(seed_root: Path) -> Path:
    seed_root.mkdir(parents=True, exist_ok=True)
    indexes: list[int] = []
    for path in seed_root.glob("attempt-*"):
        try:
            indexes.append(int(path.name.split("-")[-1]))
        except ValueError:
            continue
    return seed_root / f"attempt-{max(indexes, default=0) + 1:02d}"


def run_command_with_heartbeat(
    command: list[str],
    cwd: Path,
    environment: dict[str, str],
    log_path: Path,
    heartbeat_path: Path,
    seed: int,
    core: int,
) -> int:
    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        log.write("command=" + json.dumps(command) + "\n")
        log.write(f"seed={seed}\ncore={core}\nstarted_epoch={started}\n")
        log.flush()
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        last_heartbeat = 0.0
        while process.poll() is None:
            now = time.time()
            if now - last_heartbeat >= 300:
                heartbeat = {
                    "seed": int(seed),
                    "core": int(core),
                    "pid": int(process.pid),
                    "elapsed_seconds": now - started,
                    "time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                }
                write_json(heartbeat_path, heartbeat)
                log.write("heartbeat=" + json.dumps(heartbeat) + "\n")
                log.flush()
                last_heartbeat = now
            time.sleep(30)
        return int(process.returncode)


def run_seed(
    seed: int,
    core: int,
    base_config: dict[str, Any],
    config_dir: Path,
    output_root: Path,
    experiment_script: Path,
    python_executable: Path,
) -> dict[str, Any]:
    expected_protocol = str(base_config["protocol_sha256"]).upper()
    seed_root = output_root / f"seed-{seed}"
    existing = completed_attempt(seed_root, seed, expected_protocol)
    if existing is not None:
        return {"seed": seed, "status": "skipped_completed", "attempt": str(existing)}
    config_path = config_dir / f"full_real_seed_{seed}.json"
    derive_seed_config(base_config, seed, config_path)
    attempt = next_attempt(seed_root)
    attempt.mkdir(parents=True, exist_ok=False)
    log_path = seed_root / f"{attempt.name}.cli.log"
    heartbeat_path = seed_root / f"{attempt.name}.heartbeat.json"
    if os.name == "nt":
        command = [
            str(python_executable),
            str(experiment_script),
            "--config",
            str(config_path),
            "--mode",
            "full-real",
            "--output",
            str(attempt),
        ]
    else:
        command = [
            "timeout",
            "--signal=TERM",
            "--kill-after=10m",
            "24h",
            "taskset",
            "-c",
            str(core),
            str(python_executable),
            str(experiment_script),
            "--config",
            str(config_path),
            "--mode",
            "full-real",
            "--output",
            str(attempt),
        ]
    environment = os.environ.copy()
    environment.update(
        {
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        }
    )
    started = time.time()
    exit_code = run_command_with_heartbeat(
        command,
        experiment_script.parent.parent,
        environment,
        log_path,
        heartbeat_path,
        seed,
        core,
    )
    status = {
        "seed": int(seed),
        "core": int(core),
        "attempt": str(attempt),
        "command": command,
        "exit_code": exit_code,
        "started_epoch": started,
        "finished_epoch": time.time(),
    }
    write_json(attempt / "partition_status.json", status)
    if exit_code != 0:
        raise RuntimeError(f"Seed {seed} failed with exit code {exit_code}")
    validate_completed_attempt(attempt, seed, expected_protocol)
    return {**status, "status": "completed"}


def merge_partitions(
    base_config_path: Path,
    output_root: Path,
    combined_output: Path,
    experiment_script: Path,
) -> dict[str, Any]:
    if combined_output.exists():
        raise FileExistsError(f"Refusing to overwrite combined output: {combined_output}")
    base_config = json.loads(base_config_path.read_text("utf-8"))
    seeds = [int(value) for value in base_config["base_seed_order"]]
    expected_protocol = str(base_config["protocol_sha256"]).upper()
    partitions: list[dict[str, Any]] = []
    attempts: list[Path] = []
    for seed in seeds:
        attempt = completed_attempt(output_root / f"seed-{seed}", seed, expected_protocol)
        if attempt is None:
            raise AssertionError(f"No verified completed partition for seed {seed}")
        partitions.append(validate_completed_attempt(attempt, seed, expected_protocol))
        attempts.append(attempt)
    temporary = combined_output.with_name(
        f".{combined_output.name}.tmp-{os.getpid()}"
    )
    if temporary.exists():
        raise FileExistsError(f"Temporary combined output exists: {temporary}")
    temporary.mkdir(parents=True)
    for collection in COLLECTIONS:
        frames = [pd.read_csv(attempt / f"{collection}.csv") for attempt in attempts]
        pd.concat(frames, ignore_index=True).to_csv(
            temporary / f"{collection}.csv", index=False
        )
    manifests = [pd.read_csv(attempt / "dataset_manifest.csv") for attempt in attempts]
    pd.concat(manifests, ignore_index=True).to_csv(
        temporary / "dataset_manifest.csv", index=False
    )
    metrics = pd.read_csv(temporary / "metrics.csv")
    if set(metrics["split_seed"].astype(int)) != set(seeds):
        raise AssertionError("Combined seed set mismatch")
    if len(metrics) != len(seeds) * len(METHODS):
        raise AssertionError("Combined method-row count mismatch")
    if set(metrics["model"]) != set(METHODS):
        raise AssertionError("Combined method set mismatch")
    if metrics.duplicated(KEYS + ["model"]).any():
        raise AssertionError("Combined output has duplicate metric keys")
    shutil.copy2(base_config_path, temporary / "config_resolved.json")
    environments = [
        json.loads((attempt / "environment.json").read_text("utf-8"))
        for attempt in attempts
    ]
    write_json(
        temporary / "environment.json",
        {
            "partitioned_execution": True,
            "partition_count": len(attempts),
            "environments": environments,
        },
    )
    write_json(
        temporary / "partition_manifest.json",
        {
            "base_config": str(base_config_path.resolve()),
            "base_config_sha256": sha256_file(base_config_path),
            "protocol_sha256": expected_protocol,
            "methodology_change_id": "MCH-SHIL-002",
            "scientific_protocol_changed": True,
            "partitions": partitions,
        },
    )
    write_json(
        temporary / "run_identity.json",
        {
            "experiment_id": base_config["experiment_id"],
            "mode": "full-real",
            "execution_mode": "partitioned_locked_seeds",
            "protocol_sha256": expected_protocol,
            "orchestrator_sha256": sha256_file(Path(__file__)),
            "experiment_source_sha256": sha256_file(experiment_script),
            "predecessor_source_sha256": sha256_file(
                experiment_script.with_name("predecessor_sc_shil_experiment.py")
            ),
        },
    )
    summary = {
        "experiment_id": base_config["experiment_id"],
        "mode": "full-real",
        "status": "completed",
        "dataset_runs": len(seeds),
        "method_rows": len(metrics),
        "execution_mode": "partitioned_locked_seeds",
        "completed_at_local": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    write_json(temporary / "run_summary.json", summary)
    write_json(
        temporary / "output_hashes.json",
        {
            path.name: sha256_file(path)
            for path in sorted(temporary.iterdir())
            if path.is_file() and path.name != "output_hashes.json"
        },
    )
    temporary.replace(combined_output)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment-script", type=Path, required=True)
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--config-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--combined-output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--cores", default="0,1,2")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_config = json.loads(args.config.read_text("utf-8"))
    seeds = [int(value) for value in base_config["base_seed_order"]]
    cores = [int(value) for value in args.cores.split(",") if value.strip()]
    if not 1 <= int(args.workers) <= len(cores):
        raise ValueError("workers must be between 1 and the declared core count")
    args.config_dir.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    expected_protocol = str(base_config["protocol_sha256"]).upper()
    pending = [
        seed
        for seed in seeds
        if completed_attempt(args.output_root / f"seed-{seed}", seed, expected_protocol)
        is None
    ]
    status_path = args.output_root / "orchestrator_status.json"
    for offset in range(0, len(pending), int(args.workers)):
        batch = pending[offset : offset + int(args.workers)]
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(batch)
        ) as executor:
            futures = [
                executor.submit(
                    run_seed,
                    seed,
                    cores[index],
                    base_config,
                    args.config_dir,
                    args.output_root,
                    args.experiment_script,
                    args.python_executable,
                )
                for index, seed in enumerate(batch)
            ]
            for future in concurrent.futures.as_completed(futures):
                result = future.result()
                print(json.dumps(result, sort_keys=True), flush=True)
                completed = [
                    seed
                    for seed in seeds
                    if completed_attempt(
                        args.output_root / f"seed-{seed}",
                        seed,
                        expected_protocol,
                    )
                    is not None
                ]
                write_json(
                    status_path,
                    {
                        "status": "running",
                        "completed_seeds": completed,
                        "completed_count": len(completed),
                        "total": len(seeds),
                        "last_result": result,
                        "time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                    },
                )
    summary = merge_partitions(
        args.config, args.output_root, args.combined_output, args.experiment_script
    )
    write_json(
        status_path,
        {
            "status": "completed",
            "completed_seeds": seeds,
            "completed_count": len(seeds),
            "total": len(seeds),
            "summary": summary,
            "time": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        },
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
