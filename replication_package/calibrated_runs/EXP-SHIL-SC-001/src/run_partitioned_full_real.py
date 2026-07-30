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


METHODS = ["L1-full", "L1-top8", "L1-match", "SHIL-k8", "SC-L1", "SC-SHIL"]
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
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
        "base_seed_order": [int(value) for value in base_config["real_split_seeds"]],
        "scientific_protocol_changed": False,
    }
    write_json(path, derived)


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
        "output_hashes.json",
    ]
    missing = [name for name in required if not (attempt / name).is_file()]
    if missing:
        raise AssertionError(f"Seed {seed} attempt missing outputs: {missing}")

    summary = json.loads((attempt / "run_summary.json").read_text(encoding="utf-8"))
    config = json.loads((attempt / "config_resolved.json").read_text(encoding="utf-8"))
    identity = json.loads((attempt / "run_identity.json").read_text(encoding="utf-8"))
    if summary.get("status") != "completed" or int(summary.get("dataset_runs", 0)) != 1:
        raise AssertionError(f"Seed {seed} attempt is not a completed one-run output")
    if [int(value) for value in config.get("real_split_seeds", [])] != [int(seed)]:
        raise AssertionError(f"Seed {seed} derived config mismatch")
    if str(config.get("protocol_sha256", "")).upper() != expected_protocol:
        raise AssertionError(f"Seed {seed} config protocol mismatch")
    if str(identity.get("protocol_sha256", "")).upper() != expected_protocol:
        raise AssertionError(f"Seed {seed} identity protocol mismatch")

    metrics = pd.read_csv(attempt / "metrics.csv")
    if len(metrics) != len(METHODS) or set(metrics["model"]) != set(METHODS):
        raise AssertionError(f"Seed {seed} method rows are incomplete")
    if set(metrics["split_seed"].astype(int)) != {int(seed)}:
        raise AssertionError(f"Seed {seed} metric provenance mismatch")
    if metrics.duplicated(KEYS + ["model"]).any():
        raise AssertionError(f"Seed {seed} has duplicate metric keys")
    return {
        "seed": int(seed),
        "attempt": str(attempt),
        "metrics_rows": len(metrics),
        "protocol_sha256": expected_protocol,
        "output_hashes_sha256": sha256_file(attempt / "output_hashes.json"),
    }


def completed_attempt(seed_root: Path, seed: int, expected_protocol: str) -> Path | None:
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
    indexes = []
    for path in seed_root.glob("attempt-*"):
        try:
            indexes.append(int(path.name.split("-")[-1]))
        except ValueError:
            continue
    return seed_root / f"attempt-{(max(indexes, default=0) + 1):02d}"


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
    # Keep the command log outside the scientific output directory. The core
    # runner intentionally refuses any non-empty output directory so that a
    # previous attempt can never be overwritten.
    log_path = seed_root / f"{attempt.name}.cli.log"
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
    with log_path.open("w", encoding="utf-8") as log:
        log.write("command=" + json.dumps(command) + "\n")
        log.write(f"seed={seed}\ncore={core}\nstarted_epoch={started}\n")
        log.flush()
        completed = subprocess.run(
            command,
            cwd=experiment_script.parent.parent,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    status = {
        "seed": int(seed),
        "core": int(core),
        "attempt": str(attempt),
        "command": command,
        "exit_code": int(completed.returncode),
        "started_epoch": started,
        "finished_epoch": time.time(),
    }
    write_json(attempt / "partition_status.json", status)
    if completed.returncode != 0:
        raise RuntimeError(f"Seed {seed} failed with exit code {completed.returncode}")
    validate_completed_attempt(attempt, seed, expected_protocol)
    return {**status, "status": "completed"}


def merge_partitions(
    base_config_path: Path, output_root: Path, combined_output: Path
) -> dict[str, Any]:
    if combined_output.exists():
        raise FileExistsError(f"Refusing to overwrite combined output: {combined_output}")
    base_config = json.loads(base_config_path.read_text(encoding="utf-8"))
    seeds = [int(value) for value in base_config["real_split_seeds"]]
    expected_protocol = str(base_config["protocol_sha256"]).upper()
    partitions: list[dict[str, Any]] = []
    attempts: list[Path] = []
    for seed in seeds:
        attempt = completed_attempt(output_root / f"seed-{seed}", seed, expected_protocol)
        if attempt is None:
            raise AssertionError(f"No verified completed partition for seed {seed}")
        partitions.append(validate_completed_attempt(attempt, seed, expected_protocol))
        attempts.append(attempt)

    temporary = combined_output.with_name(f".{combined_output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise FileExistsError(f"Temporary combined output already exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
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
        observed_seeds = set(metrics["split_seed"].astype(int))
        if observed_seeds != set(seeds):
            raise AssertionError(
                f"Combined seed mismatch: expected {seeds}, observed {sorted(observed_seeds)}"
            )
        if len(metrics) != len(seeds) * len(METHODS):
            raise AssertionError(
                f"Combined metrics rows mismatch: {len(metrics)}"
            )
        if set(metrics["model"]) != set(METHODS):
            raise AssertionError("Combined method set mismatch")
        if metrics.duplicated(KEYS + ["model"]).any():
            raise AssertionError("Combined output contains duplicate metric keys")

        shutil.copy2(base_config_path, temporary / "config_resolved.json")
        environments = [
            json.loads((attempt / "environment.json").read_text(encoding="utf-8"))
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
                "scientific_protocol_changed": False,
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
                "experiment_source_sha256": sha256_file(
                    Path(__file__).with_name("sc_shil_experiment.py")
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
    except BaseException:
        raise


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
    base_config = json.loads(args.config.read_text(encoding="utf-8"))
    seeds = [int(value) for value in base_config["real_split_seeds"]]
    cores = [int(value) for value in args.cores.split(",") if value.strip()]
    if not 1 <= int(args.workers) <= len(cores):
        raise ValueError("workers must be between 1 and the number of declared cores")
    args.config_dir.mkdir(parents=True, exist_ok=True)
    args.output_root.mkdir(parents=True, exist_ok=True)

    expected_protocol = str(base_config["protocol_sha256"]).upper()
    pending = [
        seed
        for seed in seeds
        if completed_attempt(args.output_root / f"seed-{seed}", seed, expected_protocol)
        is None
    ]
    for offset in range(0, len(pending), int(args.workers)):
        batch = pending[offset : offset + int(args.workers)]
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(batch)) as executor:
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
                print(json.dumps(future.result(), sort_keys=True), flush=True)

    summary = merge_partitions(args.config, args.output_root, args.combined_output)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
