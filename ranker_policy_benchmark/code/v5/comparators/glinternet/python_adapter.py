"""Fail-closed file-schema bridge for the scoped glinternet comparator.

The bridge validates a path-and-hash-bound execution config before rendering a
container command.  It does not grant global scientific execution authority.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence


class GlinternetAdapterError(ValueError):
    """Raised when comparator identity, input, or output evidence is invalid."""


SCHEMA_VERSION = 1
ADAPTER_ID = "glinternet-r-file-v1"
PACKAGE_VERSION = "1.0.13"
PACKAGE_SOURCE_SHA256 = (
    "316C5973FF55DEA0BA0BEBB6930449B12DDC21EBD77B32714BEC665643DBC076"
)
ROCKER_ARM64_DIGEST = (
    "sha256:f7e09f3032a60a6446328448f458d5f47189d83d7776ac2f0c33073cd781edda"
)
RUNTIME_IMAGE_TAG = "track-a-glinternet-smoke:1.0.13-r4.6.1-arm64"
SCORE_DIRECTION = "higher_is_better"
BUILD_AUTHORIZATION_EVIDENCE_SHA256 = (
    "1629B59CD27D193B70E33701045BA0EA42DC8733C06551C00D6A28E1BB07EC3B"
)
ACTIVE_STRUCTURED_REGISTRY_CANONICAL_SHA256 = (
    "04072E5A3B80BED77A144361AC429D0B1478285CDCD449306787EB5B5F52C584"
)
PILOT_A_CONFIG_SHA256 = (
    "46E3A6CFD886FD6EC6DF6F0180907E6992B1F7D552A1E4FB6645E9C2F4D6A6CB"
)
PILOT_A_FREEZE_PATH = "freeze/SCIENTIFIC_RUN_FROZEN_PILOT_A.json"

REQUEST_FIELDS = {
    "schema_version",
    "adapter_id",
    "fixture_id",
    "task",
    "family",
    "index_base",
    "seed",
    "n_lambda",
    "lambda_min_ratio",
    "tolerance",
    "max_iter",
    "num_cores",
    "feature_names",
    "candidate_pair_ids",
    "files",
    "expected_package",
}
FILE_KEYS = {
    "train_x",
    "train_y",
    "validation_x",
    "validation_y",
    "test_x",
    "candidate_pairs",
}
FILE_SPEC_FIELDS = {"path", "sha256", "rows", "columns"}
RESPONSE_FILES = {
    "adapter_meta.csv",
    "validation_path.csv",
    "pair_scores.csv",
    "predictions.csv",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _exact_fields(record: Mapping[str, Any], expected: set[str], label: str) -> None:
    if not isinstance(record, Mapping):
        raise GlinternetAdapterError(f"{label} must be an object")
    missing = expected - set(record)
    unknown = set(record) - expected
    if missing or unknown:
        raise GlinternetAdapterError(
            f"{label} field mismatch: missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _resolved_child(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative.strip():
        raise GlinternetAdapterError(f"{label} path must be non-empty")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as error:
        raise GlinternetAdapterError(f"{label} path escapes fixture root") from error
    return candidate


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            headers = list(reader.fieldnames or [])
            rows = list(reader)
    except OSError as error:
        raise GlinternetAdapterError(f"Cannot read CSV {path}: {error}") from error
    if not headers:
        raise GlinternetAdapterError(f"CSV has no header: {path}")
    return headers, rows


def _finite_float(value: Any, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise GlinternetAdapterError(f"{label} must be numeric") from error
    if not math.isfinite(number):
        raise GlinternetAdapterError(f"{label} must be finite")
    return number


def _validate_file_spec(
    fixture_root: Path, key: str, spec: Mapping[str, Any]
) -> tuple[Path, list[str], list[dict[str, str]]]:
    _exact_fields(spec, FILE_SPEC_FIELDS, f"files.{key}")
    path = _resolved_child(fixture_root, spec["path"], f"files.{key}")
    if not path.is_file():
        raise GlinternetAdapterError(f"Declared input is missing: {key}")
    expected_hash = str(spec["sha256"]).upper()
    if len(expected_hash) != 64 or sha256_file(path) != expected_hash:
        raise GlinternetAdapterError(f"Input SHA-256 mismatch: {key}")
    headers, rows = _read_csv(path)
    if int(spec["rows"]) != len(rows) or int(spec["columns"]) != len(headers):
        raise GlinternetAdapterError(f"Input shape mismatch: {key}")
    return path, headers, rows


def load_and_validate_request(request_path: str | Path) -> dict[str, Any]:
    path = Path(request_path).resolve()
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GlinternetAdapterError(f"Cannot load request: {error}") from error
    _exact_fields(document, REQUEST_FIELDS, "request")
    if document["schema_version"] != SCHEMA_VERSION:
        raise GlinternetAdapterError("Unsupported request schema_version")
    if document["adapter_id"] != ADAPTER_ID:
        raise GlinternetAdapterError("Unexpected adapter_id")
    if document["task"] != "pairwise_strong_hierarchy_screening":
        raise GlinternetAdapterError("Unexpected comparator task")
    if document["family"] != "binomial" or document["index_base"] != 0:
        raise GlinternetAdapterError("Only zero-based binomial requests are supported")
    controls = {
        "seed": int(document["seed"]),
        "n_lambda": int(document["n_lambda"]),
        "lambda_min_ratio": _finite_float(
            document["lambda_min_ratio"], "lambda_min_ratio"
        ),
        "tolerance": _finite_float(document["tolerance"], "tolerance"),
        "max_iter": int(document["max_iter"]),
        "num_cores": int(document["num_cores"]),
    }
    if controls["n_lambda"] < 3 or not 0 < controls["lambda_min_ratio"] < 1:
        raise GlinternetAdapterError("Invalid lambda path controls")
    if controls["tolerance"] <= 0 or controls["max_iter"] < 1:
        raise GlinternetAdapterError("Invalid optimizer controls")
    if controls["num_cores"] != 1:
        raise GlinternetAdapterError("num_cores must remain one")
    expected_package = document["expected_package"]
    _exact_fields(
        expected_package,
        {"name", "version", "source_sha256", "rocker_arm64_digest"},
        "expected_package",
    )
    if expected_package != {
        "name": "glinternet",
        "version": PACKAGE_VERSION,
        "source_sha256": PACKAGE_SOURCE_SHA256,
        "rocker_arm64_digest": ROCKER_ARM64_DIGEST,
    }:
        raise GlinternetAdapterError("Package or image identity mismatch")
    feature_names = document["feature_names"]
    pair_ids = document["candidate_pair_ids"]
    if (
        not isinstance(feature_names, list)
        or len(feature_names) < 2
        or any(not isinstance(value, str) or not value for value in feature_names)
        or len(set(feature_names)) != len(feature_names)
    ):
        raise GlinternetAdapterError("feature_names must be unique non-empty strings")
    if (
        not isinstance(pair_ids, list)
        or not pair_ids
        or any(not isinstance(value, str) or not value for value in pair_ids)
        or len(set(pair_ids)) != len(pair_ids)
    ):
        raise GlinternetAdapterError("candidate_pair_ids must be unique strings")

    files = document["files"]
    _exact_fields(files, FILE_KEYS, "files")
    fixture_root = path.parent
    validated: dict[str, Path] = {}
    frames: dict[str, tuple[list[str], list[dict[str, str]]]] = {}
    for key in sorted(FILE_KEYS):
        file_path, headers, rows = _validate_file_spec(fixture_root, key, files[key])
        validated[key] = file_path
        frames[key] = (headers, rows)
    for key in ("train_x", "validation_x", "test_x"):
        headers, rows = frames[key]
        if headers != feature_names:
            raise GlinternetAdapterError(f"Feature header mismatch: {key}")
        for row in rows:
            for name in headers:
                _finite_float(row[name], f"{key}.{name}")
    for key in ("train_y", "validation_y"):
        headers, rows = frames[key]
        if headers != ["y"]:
            raise GlinternetAdapterError(f"Response header mismatch: {key}")
        labels = [row["y"] for row in rows]
        if set(labels) != {"0", "1"}:
            raise GlinternetAdapterError(f"Both binary classes are required: {key}")
    if len(frames["train_x"][1]) != len(frames["train_y"][1]):
        raise GlinternetAdapterError("Train feature/response row mismatch")
    if len(frames["validation_x"][1]) != len(frames["validation_y"][1]):
        raise GlinternetAdapterError("Validation feature/response row mismatch")
    pair_headers, pair_rows = frames["candidate_pairs"]
    if pair_headers != ["pair_id", "left_index", "right_index"]:
        raise GlinternetAdapterError("Candidate-pair header mismatch")
    observed_ids: list[str] = []
    observed_pairs: set[tuple[int, int]] = set()
    for row in pair_rows:
        pair_id = row["pair_id"]
        try:
            left, right = int(row["left_index"]), int(row["right_index"])
        except ValueError as error:
            raise GlinternetAdapterError(
                "Candidate indices must be integers"
            ) from error
        if not 0 <= left < right < len(feature_names):
            raise GlinternetAdapterError("Candidate pair is out of bounds or unordered")
        if (left, right) in observed_pairs:
            raise GlinternetAdapterError("Duplicate candidate pair")
        observed_pairs.add((left, right))
        observed_ids.append(pair_id)
    if observed_ids != pair_ids:
        raise GlinternetAdapterError("Candidate pair IDs differ from request order")
    document["_validated_paths"] = {key: str(value) for key, value in validated.items()}
    return document


def render_container_command(
    request_path: str | Path,
    output_dir: str | Path,
    *,
    image_reference: str,
) -> list[str]:
    request = load_and_validate_request(request_path)
    if image_reference != RUNTIME_IMAGE_TAG:
        raise GlinternetAdapterError(
            "Image reference differs from the locked local smoke tag"
        )
    fixture_root = Path(request_path).resolve().parent
    output = Path(output_dir).resolve()
    if not output.is_dir():
        raise GlinternetAdapterError("Response directory must already exist")
    if any(output.iterdir()):
        raise GlinternetAdapterError("Refusing a non-empty response directory")
    paths = request["_validated_paths"]
    relative_args = {
        "train-x": Path(paths["train_x"]).relative_to(fixture_root).as_posix(),
        "train-y": Path(paths["train_y"]).relative_to(fixture_root).as_posix(),
        "validation-x": Path(paths["validation_x"])
        .relative_to(fixture_root)
        .as_posix(),
        "validation-y": Path(paths["validation_y"])
        .relative_to(fixture_root)
        .as_posix(),
        "test-x": Path(paths["test_x"]).relative_to(fixture_root).as_posix(),
        "candidate-pairs": Path(paths["candidate_pairs"])
        .relative_to(fixture_root)
        .as_posix(),
    }
    command = [
        "docker",
        "run",
        "--rm",
        "--network=none",
        "--cpus=1",
        "--memory=2g",
        "--read-only",
        "--tmpfs=/tmp:rw,noexec,nosuid,size=256m",
        f"--mount=type=bind,src={fixture_root},dst=/input,readonly",
        f"--mount=type=bind,src={output},dst=/output",
        image_reference,
    ]
    for name, relative in relative_args.items():
        command.append(f"--{name}=/input/{relative}")
    command.extend(
        [
            "--output-dir=/output",
            f"--seed={int(request['seed'])}",
            f"--n-lambda={int(request['n_lambda'])}",
            f"--lambda-min-ratio={float(request['lambda_min_ratio'])}",
            f"--tolerance={float(request['tolerance'])}",
            f"--max-iter={int(request['max_iter'])}",
            f"--num-cores={int(request['num_cores'])}",
            f"--fixture-id={request['fixture_id']}",
        ]
    )
    return command


def _single_csv_row(path: Path) -> dict[str, str]:
    _headers, rows = _read_csv(path)
    if len(rows) != 1:
        raise GlinternetAdapterError(f"Expected exactly one row: {path.name}")
    return rows[0]


def validate_response(
    request_path: str | Path, response_dir: str | Path
) -> dict[str, Any]:
    request = load_and_validate_request(request_path)
    root = Path(response_dir).resolve()
    if not root.is_dir():
        raise GlinternetAdapterError("Response directory is missing")
    observed_files = {path.name for path in root.iterdir() if path.is_file()}
    if observed_files != RESPONSE_FILES:
        raise GlinternetAdapterError(
            f"Response file set mismatch: {sorted(observed_files)}"
        )
    meta = _single_csv_row(root / "adapter_meta.csv")
    expected_meta = {
        "schema_version": str(SCHEMA_VERSION),
        "adapter_id": ADAPTER_ID,
        "fixture_id": str(request["fixture_id"]),
        "package": "glinternet",
        "package_version": PACKAGE_VERSION,
        "package_source_sha256": PACKAGE_SOURCE_SHA256,
        "rocker_arm64_digest": ROCKER_ARM64_DIGEST,
        "family": "binomial",
        "index_base": "0",
        "num_cores": "1",
        "fit_path_calls": "1",
        "candidate_pair_count": str(len(request["candidate_pair_ids"])),
        "test_row_count": str(request["files"]["test_x"]["rows"]),
    }
    for key, expected in expected_meta.items():
        if meta.get(key) != expected:
            raise GlinternetAdapterError(f"Adapter metadata mismatch: {key}")
    chosen_index = int(meta.get("chosen_lambda_index", "-1"))
    validation_headers, validation_rows = _read_csv(root / "validation_path.csv")
    if validation_headers != [
        "lambda_index",
        "lambda",
        "validation_log_loss",
        "chosen",
    ]:
        raise GlinternetAdapterError("Validation-path schema mismatch")
    if len(validation_rows) < 3:
        raise GlinternetAdapterError("Validation path is incomplete")
    chosen = []
    for row in validation_rows:
        index = int(row["lambda_index"])
        _finite_float(row["lambda"], "lambda")
        _finite_float(row["validation_log_loss"], "validation_log_loss")
        if row["chosen"] not in {"TRUE", "FALSE", "True", "False", "true", "false"}:
            raise GlinternetAdapterError("Invalid chosen flag")
        if row["chosen"].lower() == "true":
            chosen.append(index)
    if chosen != [chosen_index]:
        raise GlinternetAdapterError("Chosen lambda evidence is ambiguous")

    pair_headers, pair_rows = _read_csv(root / "pair_scores.csv")
    if pair_headers != [
        "pair_id",
        "left_index",
        "right_index",
        "score",
        "selected",
        "score_direction",
    ]:
        raise GlinternetAdapterError("Pair-score schema mismatch")
    if [row["pair_id"] for row in pair_rows] != request["candidate_pair_ids"]:
        raise GlinternetAdapterError("Pair-score universe/order mismatch")
    seen_pairs: set[tuple[int, int]] = set()
    selected_ids: list[str] = []
    for row in pair_rows:
        pair = (int(row["left_index"]), int(row["right_index"]))
        if not 0 <= pair[0] < pair[1] < len(request["feature_names"]):
            raise GlinternetAdapterError("Response pair is invalid")
        if pair in seen_pairs:
            raise GlinternetAdapterError("Response pair is duplicated")
        seen_pairs.add(pair)
        score = _finite_float(row["score"], "pair score")
        if score < 0 or row["score_direction"] != SCORE_DIRECTION:
            raise GlinternetAdapterError("Pair score direction/value is invalid")
        selected = row["selected"].lower()
        if selected not in {"true", "false"} or (selected == "true") != (score > 0):
            raise GlinternetAdapterError("Selected flag disagrees with score")
        if selected == "true":
            selected_ids.append(row["pair_id"])

    prediction_headers, prediction_rows = _read_csv(root / "predictions.csv")
    if prediction_headers != ["row_id", "probability_class_1"]:
        raise GlinternetAdapterError("Prediction schema mismatch")
    if len(prediction_rows) != int(request["files"]["test_x"]["rows"]):
        raise GlinternetAdapterError("Prediction row count mismatch")
    if [int(row["row_id"]) for row in prediction_rows] != list(
        range(len(prediction_rows))
    ):
        raise GlinternetAdapterError(
            "Prediction row IDs are not zero-based and complete"
        )
    for row in prediction_rows:
        probability = _finite_float(row["probability_class_1"], "probability")
        if not 0 <= probability <= 1:
            raise GlinternetAdapterError("Probability is outside [0,1]")
    return {
        "status": "SCHEMA_VALIDATED",
        "fixture_id": request["fixture_id"],
        "chosen_lambda_index": chosen_index,
        "candidate_pair_count": len(pair_rows),
        "selected_pair_ids": tuple(selected_ids),
        "prediction_count": len(prediction_rows),
        "package_version": meta["package_version"],
        "package_source_sha256": meta["package_source_sha256"],
        "rocker_arm64_digest": meta["rocker_arm64_digest"],
    }


def validate_inactive_registry(candidate_root: str | Path) -> None:
    root = Path(candidate_root).resolve()
    config = json.loads(
        (root / "config" / "full_config.json").read_text(encoding="utf-8")
    )
    record = config.get("structured_comparator_candidate")
    if not isinstance(record, Mapping):
        raise GlinternetAdapterError("structured_comparator_candidate is missing")
    if record.get("registry_active") is not False:
        raise GlinternetAdapterError("glinternet registry activation is prohibited")
    if (
        record.get("package") != "glinternet"
        or record.get("version") != PACKAGE_VERSION
        or str(record.get("source_sha256")).upper() != PACKAGE_SOURCE_SHA256
    ):
        raise GlinternetAdapterError("Candidate config package identity mismatch")


def validate_build_contract(adapter_root: str | Path) -> dict[str, Any]:
    root = Path(adapter_root).resolve()
    try:
        contract = json.loads(
            (root / "build_contract.json").read_text(encoding="utf-8")
        )
    except (OSError, json.JSONDecodeError) as error:
        raise GlinternetAdapterError(f"Cannot load build contract: {error}") from error
    _exact_fields(
        contract,
        {
            "schema_version",
            "contract_id",
            "status",
            "registry_active",
            "scientific_compute_authorized",
            "scoped_subprocess_authorized",
            "authorization_evidence",
            "authorization_scope",
            "target_platform",
            "base_image",
            "package",
            "container",
            "adapter",
            "source_bindings",
            "gates",
        },
        "build_contract",
    )
    if contract["schema_version"] != 1 or contract["registry_active"] is not True:
        raise GlinternetAdapterError(
            "Build contract must be schema v1 and registry-active"
        )
    if contract["scientific_compute_authorized"] is not False:
        raise GlinternetAdapterError(
            "Build contract cannot authorize global scientific compute"
        )
    if contract["scoped_subprocess_authorized"] is not True:
        raise GlinternetAdapterError("Scoped subprocess authorization is missing")
    if contract["target_platform"] != "linux/arm64":
        raise GlinternetAdapterError("Unexpected comparator target platform")
    if contract["base_image"].get("platform_digest") != ROCKER_ARM64_DIGEST:
        raise GlinternetAdapterError("Build-contract base image digest mismatch")
    if (
        contract["package"].get("version") != PACKAGE_VERSION
        or str(contract["package"].get("source_sha256")).upper()
        != PACKAGE_SOURCE_SHA256
    ):
        raise GlinternetAdapterError("Build-contract package identity mismatch")
    container = contract["container"]
    if container.get("local_image_tag") != RUNTIME_IMAGE_TAG:
        raise GlinternetAdapterError("Build-contract runtime image tag mismatch")
    if (
        container.get("build_executed") is not True
        or container.get("built_image_id")
        != "sha256:f20fc7e80ccaaaf06292d91e5cf94247f3cad56576e9ee6f5dca755194b49164"
    ):
        raise GlinternetAdapterError(
            "Build contract does not bind the verified ARM64 image"
        )
    authorization = contract["authorization_evidence"]
    _exact_fields(
        authorization,
        {"path", "sha256", "required_decision"},
        "authorization_evidence",
    )
    if authorization != {
        "path": "evidence/glinternet_build_authorization_20260826/BUILD_AUTHORIZATION_EVIDENCE.json",
        "sha256": BUILD_AUTHORIZATION_EVIDENCE_SHA256,
        "required_decision": "AUTHORIZE_EVIDENCE_BOUND_SCOPED_GLINTERNET_SUBPROCESS",
    }:
        raise GlinternetAdapterError("Build authorization evidence binding mismatch")
    scope = contract["authorization_scope"]
    _exact_fields(
        scope,
        {"allowed_scopes", "global_scientific_execution_authorized"},
        "authorization_scope",
    )
    if scope != {
        "allowed_scopes": [
            "masked_timing_s00_s11_s13",
            "separately_frozen_pilot_or_full",
        ],
        "global_scientific_execution_authorized": False,
    }:
        raise GlinternetAdapterError("Build authorization scope mismatch")
    bindings = contract["source_bindings"]
    if not isinstance(bindings, Mapping) or not bindings:
        raise GlinternetAdapterError("Build contract has no source bindings")
    for relative, expected in bindings.items():
        path = _resolved_child(root, relative, "source binding")
        if not path.is_file() or sha256_file(path) != str(expected).upper():
            raise GlinternetAdapterError(f"Build-contract source drift: {relative}")
    return contract


def _hash_bound_json(
    root: Path, record: Mapping[str, Any], label: str
) -> Mapping[str, Any]:
    if not isinstance(record, Mapping) or not {"path", "sha256"}.issubset(record):
        raise GlinternetAdapterError(f"{label} lacks path/SHA-256 binding")
    path = _resolved_child(root, record["path"], label)
    if not path.is_file() or sha256_file(path) != str(record["sha256"]).upper():
        raise GlinternetAdapterError(f"{label} SHA-256 mismatch")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise GlinternetAdapterError(f"Cannot read {label}: {error}") from error
    if not isinstance(document, Mapping):
        raise GlinternetAdapterError(f"{label} must contain an object")
    return document


def _validate_authorization_evidence(
    candidate_root: Path, contract: Mapping[str, Any]
) -> Mapping[str, Any]:
    record = contract["authorization_evidence"]
    evidence = _hash_bound_json(candidate_root, record, "authorization_evidence")
    required = {
        "schema_version",
        "evidence_id",
        "metadata",
        "decision",
        "preauthorization_design_freeze",
        "structured_activation",
        "final_design_review",
        "arm64_smoke",
        "prior_build_contract",
        "pinned_identity",
        "authorized_scopes",
        "gates",
    }
    _exact_fields(evidence, required, "build authorization evidence")
    if (
        evidence["schema_version"] != 1
        or evidence["decision"] != record["required_decision"]
    ):
        raise GlinternetAdapterError("Build authorization decision mismatch")
    metadata = evidence["metadata"]
    _exact_fields(
        metadata,
        {"date_time", "tool", "model_if_known", "operation_id"},
        "build authorization metadata",
    )
    if metadata["operation_id"] != "shil-dual-track-freeze-pilot-theory-20260826-1624":
        raise GlinternetAdapterError("Build authorization operation mismatch")

    freeze_record = evidence["preauthorization_design_freeze"]
    _exact_fields(
        freeze_record,
        {"path", "sha256", "required_status", "required_next_gate"},
        "preauthorization_design_freeze",
    )
    freeze = _hash_bound_json(
        candidate_root, freeze_record, "preauthorization_design_freeze"
    )
    if (
        freeze_record["sha256"]
        != "A471EFE3EF3D4C38547D9A9C5EEFD334CB0EC3ECAC2E8D02CA00ACAB02D98D94"
        or freeze.get("status") != freeze_record["required_status"]
        or freeze.get("next_gate", {}).get("allowed")
        != freeze_record["required_next_gate"]
        or freeze.get("results_seen") is not False
        or freeze.get("scientific_compute_executed") is not False
    ):
        raise GlinternetAdapterError("Preauthorization freeze predicate failed")

    activation_record = evidence["structured_activation"]
    _exact_fields(
        activation_record,
        {"path", "sha256", "required_decision"},
        "structured_activation",
    )
    activation = _hash_bound_json(
        candidate_root, activation_record, "structured_activation"
    )
    if (
        activation_record["sha256"]
        != "C3DB98D49C1C79BFD5EE6AD5CF9B5C81268A8FCDB51A900198DE101387D940C9"
        or activation.get("decision") != activation_record["required_decision"]
        or activation.get("gates", {}).get("registry_active") is not True
    ):
        raise GlinternetAdapterError("Structured activation predicate failed")

    review_record = evidence["final_design_review"]
    _exact_fields(
        review_record,
        {"path", "sha256", "required_decision"},
        "final_design_review",
    )
    review = _hash_bound_json(candidate_root, review_record, "final_design_review")
    if (
        review_record["sha256"]
        != "B92C48FC38B14ACF4F178CFFE8CA8BCD59C878EA89DAE978E900C5A94E694C3B"
        or review.get("review", {}).get("decision")
        != review_record["required_decision"]
        or review.get("review", {}).get("remaining_p0_p1_defects") != []
    ):
        raise GlinternetAdapterError("Final design-review predicate failed")

    smoke_record = evidence["arm64_smoke"]
    _exact_fields(
        smoke_record,
        {"path", "sha256", "required_status", "historic_registry_active"},
        "arm64_smoke",
    )
    smoke = _hash_bound_json(candidate_root, smoke_record, "arm64_smoke")
    if (
        smoke_record["sha256"]
        != "D97A49CE35F215F2DD6616FE3D092FF2E568313E713292FF3263F1B9F521FAF7"
        or smoke.get("status") != smoke_record["required_status"]
        or smoke_record["historic_registry_active"] is not False
        or smoke.get("registry_active") is not False
    ):
        raise GlinternetAdapterError("ARM64 smoke predicate failed")

    prior = evidence["prior_build_contract"]
    _exact_fields(
        prior,
        {
            "path",
            "file_sha256",
            "status",
            "registry_active",
            "scientific_compute_authorized",
        },
        "prior_build_contract",
    )
    if prior != {
        "path": "comparators/glinternet/build_contract.json",
        "file_sha256": "0B8701D999493CF4010BE0ABBAB226860033809894F365DAC87F297721F73677",
        "status": "STATIC_SCHEMA_TESTED_NOT_BUILT_NOT_SMOKED",
        "registry_active": False,
        "scientific_compute_authorized": False,
    }:
        raise GlinternetAdapterError("Prior build-contract binding mismatch")
    identity = evidence["pinned_identity"]
    if (
        identity.get("target_platform") != "linux/arm64"
        or identity.get("base_image_digest") != ROCKER_ARM64_DIGEST
        or identity.get("package") != "glinternet"
        or identity.get("package_version") != PACKAGE_VERSION
        or identity.get("package_source_sha256") != PACKAGE_SOURCE_SHA256
        or identity.get("built_image_id")
        != "sha256:f20fc7e80ccaaaf06292d91e5cf94247f3cad56576e9ee6f5dca755194b49164"
    ):
        raise GlinternetAdapterError("Pinned build identity mismatch")
    gates = evidence["gates"]
    if gates != {
        "scoped_subprocess_authorized": True,
        "global_scientific_execution_authorized": False,
        "registry_must_be_evidence_bound_active": True,
        "execution_config_must_be_true_at_subprocess_time": True,
        "execution_config_must_be_path_and_hash_bound": True,
        "exact_s13_applicability_required": True,
        "scientific_compute_performed_by_authorization": False,
    }:
        raise GlinternetAdapterError("Build authorization gates mismatch")
    return evidence


def validate_scoped_subprocess_authorization(
    candidate_root: str | Path, context: Mapping[str, Any]
) -> dict[str, Any]:
    """Validate all predicates required immediately before a subprocess call."""

    root = Path(candidate_root).resolve()
    contract = validate_build_contract(root / "comparators" / "glinternet")
    evidence = _validate_authorization_evidence(root, contract)
    _exact_fields(
        context,
        {
            "scope",
            "execution_config_binding",
            "cell_metadata",
            "scope_binding",
        },
        "authorization context",
    )
    execution_config_binding = context["execution_config_binding"]
    _exact_fields(
        execution_config_binding,
        {"path", "sha256"},
        "execution config binding",
    )
    config = _hash_bound_json(root, execution_config_binding, "execution config")
    if config.get("execution_enabled") is not True:
        raise GlinternetAdapterError("Bound execution config is not enabled")
    if (
        config.get("structured_comparator_candidate", {}).get("registry_active")
        is not True
    ):
        raise GlinternetAdapterError("Structured comparator is not config-active")

    source_root = root / "src"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    try:
        from structured_comparator_registry import load_structured_comparator_registry

        registry = load_structured_comparator_registry(
            root / "config" / "structured_comparator_registry.json"
        )
    except Exception as error:
        raise GlinternetAdapterError(
            f"Structured registry activation validation failed: {error}"
        ) from error
    if (
        registry.canonical_sha256 != ACTIVE_STRUCTURED_REGISTRY_CANONICAL_SHA256
        or not registry.registry_active
    ):
        raise GlinternetAdapterError("Structured registry is not evidence-bound active")

    cell = context["cell_metadata"]
    expected_cell = {
        "scenario_id": "S13",
        "cell_type": "structured_pairwise",
        "candidate_orders": [2],
        "heredity": "respected",
        "hierarchy": "strong",
        "analysis_role": "sensitivity_only",
        "primary_estimand": False,
    }
    if cell != expected_cell or registry.active_comparators_for(cell) != (
        "glinternet.strong_hierarchy.pairwise",
    ):
        raise GlinternetAdapterError("Subprocess cell is not exact eligible S13")

    scope = context["scope"]
    binding = context["scope_binding"]
    authorized_scopes = evidence["authorized_scopes"]
    if scope == "masked_timing_s00_s11_s13":
        expected_scope = {
            "execution_config": {
                "path": "config/timing_config.json",
                "sha256": "E0A644BC56274A2FCB8C27BFDE83D886737E9F4C46F8C8B1319E022D0DF36023",
            },
            "scenario_package": ["S00", "S11", "S13"],
            "comparator_execution_scenarios": ["S13"],
            "reserved_seed": 883,
            "engineering_only": True,
            "sealed_output_required": True,
            "scientific_magnitudes_exported": False,
            "purpose": "timing_and_resource_measurement_only",
        }
        if authorized_scopes.get(scope) != expected_scope:
            raise GlinternetAdapterError("Masked timing evidence binding mismatch")
        if dict(execution_config_binding) != expected_scope["execution_config"]:
            raise GlinternetAdapterError("Masked timing execution config mismatch")
        _exact_fields(
            binding,
            {
                "scenario_package",
                "reserved_seed",
                "sealed_output",
                "scientific_magnitudes_exported",
            },
            "masked timing scope binding",
        )
        if binding != {
            "scenario_package": ["S00", "S11", "S13"],
            "reserved_seed": 883,
            "sealed_output": True,
            "scientific_magnitudes_exported": False,
        }:
            raise GlinternetAdapterError("Masked timing scope predicate failed")
        timing = config.get("masked_timing_authorization")
        if (
            timing
            != {
                "scope": "masked_timing_s00_s11_s13",
                "engineering_only": True,
                "reserved_seed": 883,
                "scenario_package": ["S00", "S11", "S13"],
                "comparator_execution_scenarios": ["S13"],
                "sealed_output": True,
                "scientific_magnitudes_exported": False,
            }
            or config.get("execution_class") != "engineering_only_masked_timing"
        ):
            raise GlinternetAdapterError("Masked timing config predicate failed")
    elif scope == "separately_frozen_pilot_or_full":
        _exact_fields(
            binding,
            {"run_class", "freeze_path", "freeze_sha256"},
            "scientific run scope binding",
        )
        if binding["run_class"] not in authorized_scopes[scope]["allowed_run_classes"]:
            raise GlinternetAdapterError("Scientific run class is not authorized")
        freeze_record = {
            "path": binding["freeze_path"],
            "sha256": binding["freeze_sha256"],
        }
        run_freeze = _hash_bound_json(
            root, freeze_record, "separate scientific run freeze"
        )
        if (
            run_freeze.get("status") != "SCIENTIFIC_RUN_FROZEN"
            or run_freeze.get("execution_enabled") is not True
            or run_freeze.get("run_class") != binding["run_class"]
            or run_freeze.get("execution_config") != dict(execution_config_binding)
            or run_freeze.get("build_authorization_evidence_sha256")
            != BUILD_AUTHORIZATION_EVIDENCE_SHA256
        ):
            raise GlinternetAdapterError(
                "Separate scientific run freeze predicate failed"
            )
        if binding["run_class"] == "pilot":
            if binding["freeze_path"] != PILOT_A_FREEZE_PATH or dict(
                execution_config_binding
            ) != {
                "path": "config/pilot_config.json",
                "sha256": PILOT_A_CONFIG_SHA256,
            }:
                raise GlinternetAdapterError("Pilot A path/config binding mismatch")
            _validate_pilot_a_freeze(root, run_freeze, config)
    else:
        raise GlinternetAdapterError("Unknown subprocess authorization scope")
    return {
        "status": "SCOPED_SUBPROCESS_AUTHORIZED",
        "scope": scope,
        "comparator_id": "glinternet.strong_hierarchy.pairwise",
        "global_scientific_execution_authorized": False,
    }


def _validate_pilot_a_freeze(
    root: Path, freeze: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    """Validate the immutable six-unit structural-only Pilot A authorization."""

    expected_units = [
        {"scenario_id": scenario_id, "seed": seed}
        for scenario_id in ["S00", "S11", "S13"]
        for seed in [901, 907]
    ]
    if (
        freeze.get("freeze_id") != "track-a-scientific-pilot-a2-schema-repair-20260826"
        or freeze.get("authorized_run_id") != "EXP-SHIL-PILOT-A2-SCHEMA-REPAIR-pilot"
        or freeze.get("results_seen") is not False
        or freeze.get("scientific_compute_executed_during_authorization") is not False
        or freeze.get("exact_units") != expected_units
        or freeze.get("workload") != {"n_pairs": 4, "ranker_call_budget_4B_plus_2": 18}
        or freeze.get("blindness")
        != {
            "mode": "structural_only",
            "scientific_metrics_visible_during_pilot": False,
            "scientific_magnitudes_visible_during_pilot": False,
            "allowed_live_fields": [
                "status",
                "unit_count",
                "artifact_count",
                "bytes",
                "sha256",
                "heartbeat_sequence",
                "failure_reason",
            ],
        }
        or freeze.get("durability")
        != {
            "heartbeat_interval_seconds": 2.0,
            "minimum_advancing_heartbeat_samples": 2,
            "per_unit_timeout_seconds": 300,
            "whole_run_watchdog_seconds": 3600,
            "max_attempts_per_unit": 2,
            "reentry_rule": "same_atomic_unit_seed_only_no_replacement_seed",
            "authorized_reentry_seeds": [],
        }
    ):
        raise GlinternetAdapterError("Pilot A frozen design predicate failed")
    bindings = freeze.get("bindings")
    if not isinstance(bindings, Mapping):
        raise GlinternetAdapterError("Pilot A freeze bindings are missing")
    expected_bindings = {
        "protocol": {
            "path": "MD/02_design/track_a_benchmark_protocol_20260826.md",
            "sha256": "2E233357300FC66091CD66D24AA57BFEEC6FD1DE58D6139A18D86599F61836B8",
        },
        "execution_config": {
            "path": "config/pilot_config.json",
            "sha256": PILOT_A_CONFIG_SHA256,
        },
        "primary_registry": {
            "path": "config/method_registry.json",
            "file_sha256": "167AE80669A31FB38E6EC7BB8A6D756CD62365BF7311A3602DA2EEBE6FA8A3D1",
            "canonical_sha256": "37991B636B952023DC230A6B0CD790C127F864874C7C07C5F712267E10C32515",
            "method_count": 14,
        },
        "structured_registry": {
            "path": "config/structured_comparator_registry.json",
            "file_sha256": "4F8F51B9AFDBB615EFBAF6DEE952466E5D1AF6DB18F1089F93DAF8ED94519E60",
            "canonical_sha256": ACTIVE_STRUCTURED_REGISTRY_CANONICAL_SHA256,
        },
        "prospective_freeze_v4": {
            "path": "freeze/PROSPECTIVE_DESIGN_FREEZE_V4.json",
            "sha256": "9B08EECEA064110BC81455C951CFF0FFE4348823E8B2470F51DDB266994AED79",
        },
        "timing_evidence": {
            "path": "evidence/vps_masked_timing_b4_20260826/TIMING_EVIDENCE.json",
            "sha256": "2CDF9C23D56BFDEB2412A3DC1317AA130B25D4AA0A1135D5D051841DD13443AE",
        },
        "timing_resource_preflight": {
            "path": "evidence/vps_masked_timing_b4_20260826/PILOT_TIMING_RESOURCE_PREFLIGHT.json",
            "sha256": "F4B6A56B644843E1A15B7B2B6DA10EEACE0999008F190A27F86BAA33F57BF8EE",
        },
        "timing_run_manifest": {
            "path": "evidence/vps_masked_timing_b4_20260826/RUN_MANIFEST.json",
            "sha256": "48CF2478AE5BDB591953B6F4240CAABD1B398E10026DA6B90892090C7F051B60",
        },
        "build_authorization": {
            "path": "evidence/glinternet_build_authorization_20260826/BUILD_AUTHORIZATION_EVIDENCE.json",
            "sha256": BUILD_AUTHORIZATION_EVIDENCE_SHA256,
        },
        "arm64_smoke": {
            "path": "evidence/vps_arm64_glinternet_smoke_20260826/SMOKE_RESULT.json",
            "sha256": "D97A49CE35F215F2DD6616FE3D092FF2E568313E713292FF3263F1B9F521FAF7",
        },
    }
    for label, expected in expected_bindings.items():
        if bindings.get(label) != expected:
            raise GlinternetAdapterError(f"Pilot A {label} binding mismatch")
        expected_hash = expected.get("sha256", expected.get("file_sha256"))
        if label == "protocol":
            protocol_path = root.parents[1] / expected["path"]
            if (
                not protocol_path.is_file()
                or sha256_file(protocol_path) != expected_hash
            ):
                raise GlinternetAdapterError("Pilot A protocol binding mismatch")
        else:
            record = {"path": expected["path"], "sha256": expected_hash}
            _hash_bound_json(root, record, f"Pilot A {label}")
    predecessor = _hash_bound_json(
        root, bindings.get("pilot_a1_freeze", {}), "Pilot A1 predecessor freeze"
    )
    if (
        predecessor.get("freeze_id") != "track-a-scientific-pilot-a-20260826"
        or predecessor.get("source_bundle", {}).get("sha256")
        != "4F42578E7ADA29DAA6752269AD9CF875517F4BE6423261ED32C01897F09F8372"
    ):
        raise GlinternetAdapterError("Pilot A1 predecessor binding mismatch")
    repair = _hash_bound_json(
        root,
        bindings.get("repair_classification", {}),
        "Pilot A1 repair classification",
    )
    if (
        repair.get("joint_decision") != "CONDITIONAL_SOURCE_ONLY_REPAIR_AUTHORIZED"
        or repair.get("repair_budget", {}).get("consumed_by_this_classification") != 1
        or repair.get("repair_budget", {}).get("further_source_repair_allowed")
        is not False
        or repair.get("scientific_outputs_inspected") is not False
    ):
        raise GlinternetAdapterError("Pilot A1 repair classification mismatch")
    validator_record = bindings.get("blind_validator_v3")
    if not isinstance(validator_record, Mapping):
        raise GlinternetAdapterError("Pilot A2 blind validator binding is missing")
    validator_path = root / str(validator_record.get("path", ""))
    if (
        validator_record.get("path") != "src/validate_blind_pilot.py"
        or not validator_path.is_file()
        or sha256_file(validator_path) != validator_record.get("sha256")
    ):
        raise GlinternetAdapterError("Pilot A2 blind validator binding mismatch")
    build_contract_binding = bindings.get("build_contract")
    live_build_contract = root / "comparators" / "glinternet" / "build_contract.json"
    if build_contract_binding != {
        "path": "comparators/glinternet/build_contract.json",
        "sha256": sha256_file(live_build_contract),
    }:
        raise GlinternetAdapterError("Pilot A build-contract binding mismatch")
    if freeze.get(
        "build_authorization_evidence_sha256"
    ) != BUILD_AUTHORIZATION_EVIDENCE_SHA256 or freeze.get("arm64_image") != {
        "tag": RUNTIME_IMAGE_TAG,
        "image_id": "sha256:f20fc7e80ccaaaf06292d91e5cf94247f3cad56576e9ee6f5dca755194b49164",
        "platform": "linux/arm64",
        "base_image_digest": ROCKER_ARM64_DIGEST,
    }:
        raise GlinternetAdapterError("Pilot A build/image binding mismatch")
    preflight = _hash_bound_json(
        root, bindings["timing_resource_preflight"], "Pilot A timing preflight"
    )
    if preflight.get("overall_status") != "pass" or preflight.get("checks") != {
        "disk": "pass",
        "ram": "pass",
        "timing_sample": "pass",
    }:
        raise GlinternetAdapterError("Pilot A timing/resource preflight did not pass")
    if config.get("pilot_a_contract", {}).get("planned_units") != expected_units:
        raise GlinternetAdapterError("Pilot A config unit plan mismatch")
    try:
        import durability
        import sc_shil_experiment

        live_source = durability.build_source_binding(
            root, sc_shil_experiment.TRACK_A_DURABLE_SOURCE_PATHS
        )
    except Exception as error:
        raise GlinternetAdapterError(
            f"Cannot validate Pilot A durable source: {error}"
        ) from error
    if freeze.get("source_bundle") != {
        "sha256": live_source["source_bundle_sha256"],
        "file_count": len(live_source["files"]),
    }:
        raise GlinternetAdapterError("Pilot A durable source binding mismatch")


def command_text(command: Sequence[str]) -> str:
    """Render for inspection only; no shell execution is performed."""

    return "\n".join(str(value) for value in command)
