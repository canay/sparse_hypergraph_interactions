"""Runtime boundary for the isolated S13 glinternet sensitivity analysis.

The public runner is side-effect free while either execution gate is closed.
Its production invoker is pinned to the existing file adapter, build contract,
and container image; tests inject a response-producing callable instead.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Callable, Mapping, Sequence

import numpy as np

import sc_shil_experiment as legacy
from structured_comparator_registry import StructuredComparatorRegistry
import support_metrics


class StructuredComparatorRuntimeError(ValueError):
    """Raised when sensitivity execution or its isolated outputs are ambiguous."""


@dataclass(frozen=True)
class ComparatorInvocation:
    request_path: Path
    response_dir: Path
    adapter_root: Path
    authorization_context: Mapping[str, Any] | None = None
    timeout_seconds: float = 600.0


@dataclass(frozen=True)
class ComparatorResponse:
    validation: Mapping[str, Any]
    pair_rows: tuple[Mapping[str, Any], ...]
    probabilities_class_1: tuple[float, ...]
    command: tuple[str, ...]


@dataclass(frozen=True)
class StructuredSensitivityResult:
    registry_sha256: str
    status_rows: tuple[Mapping[str, Any], ...]
    metric_rows: tuple[Mapping[str, Any], ...]
    selected_edge_rows: tuple[Mapping[str, Any], ...]
    raw_rows: tuple[Mapping[str, Any], ...]

    def __post_init__(self) -> None:
        if len(self.status_rows) != 1:
            raise StructuredComparatorRuntimeError(
                "Sensitivity result requires exactly one status row"
            )
        status = self.status_rows[0].get("status")
        if status not in {"N/A", "DISABLED", "COMPLETED"}:
            raise StructuredComparatorRuntimeError("Unknown sensitivity status")
        if status != "COMPLETED" and (
            self.metric_rows or self.selected_edge_rows or self.raw_rows
        ):
            raise StructuredComparatorRuntimeError(
                "N/A or disabled sensitivity result cannot carry substitute outputs"
            )
        if status == "COMPLETED" and (
            len(self.metric_rows) != 1 or len(self.raw_rows) != 1
        ):
            raise StructuredComparatorRuntimeError(
                "Completed sensitivity result has an incomplete isolated artifact set"
            )


ComparatorInvoker = Callable[[ComparatorInvocation], ComparatorResponse]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _write_csv(
    path: Path, headers: Sequence[str], rows: Sequence[Sequence[Any]]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(headers)
        writer.writerows(rows)


def _file_spec(path: Path, rows: int, columns: int) -> dict[str, Any]:
    return {
        "path": path.name,
        "sha256": _sha256_file(path),
        "rows": int(rows),
        "columns": int(columns),
    }


def _prepare_request(
    dataset: legacy.Dataset,
    split_seed: int,
    request_root: Path,
    adapter_root: Path,
    authorization_context: Mapping[str, Any] | None,
) -> tuple[ComparatorInvocation, tuple[tuple[int, int], ...], np.ndarray]:
    split = legacy.outer_split_scale(dataset, int(split_seed))
    feature_names = tuple(str(name) for name in dataset.feature_names)
    if len(feature_names) != np.asarray(dataset.X).shape[1]:
        raise StructuredComparatorRuntimeError("Feature-name count mismatch")
    if len(set(feature_names)) != len(feature_names):
        raise StructuredComparatorRuntimeError("Feature names must be unique")
    if set(np.unique(dataset.y)) != {0, 1}:
        raise StructuredComparatorRuntimeError(
            "glinternet sensitivity runtime requires binary labels"
        )
    pairs = tuple(
        (left, right)
        for left in range(len(feature_names))
        for right in range(left + 1, len(feature_names))
    )
    pair_ids = tuple(f"pair-{index:06d}" for index in range(len(pairs)))
    files: dict[str, dict[str, Any]] = {}
    matrices = {
        "train_x": np.asarray(split["X_train"], dtype=float),
        "validation_x": np.asarray(split["X_val"], dtype=float),
        "test_x": np.asarray(split["X_test"], dtype=float),
    }
    for name, matrix in matrices.items():
        path = request_root / f"{name}.csv"
        _write_csv(path, feature_names, matrix.tolist())
        files[name] = _file_spec(path, matrix.shape[0], matrix.shape[1])
    responses = {
        "train_y": np.asarray(split["y_train"], dtype=int),
        "validation_y": np.asarray(split["y_val"], dtype=int),
    }
    for name, values in responses.items():
        path = request_root / f"{name}.csv"
        _write_csv(path, ("y",), ((int(value),) for value in values))
        files[name] = _file_spec(path, len(values), 1)
    pair_path = request_root / "candidate_pairs.csv"
    _write_csv(
        pair_path,
        ("pair_id", "left_index", "right_index"),
        (
            (pair_id, left, right)
            for pair_id, (left, right) in zip(pair_ids, pairs, strict=True)
        ),
    )
    files["candidate_pairs"] = _file_spec(pair_path, len(pairs), 3)
    request = {
        "schema_version": 1,
        "adapter_id": "glinternet-r-file-v1",
        "fixture_id": f"S13-seed-{int(split_seed)}",
        "task": "pairwise_strong_hierarchy_screening",
        "family": "binomial",
        "index_base": 0,
        "seed": int(split_seed),
        "n_lambda": 100,
        "lambda_min_ratio": 0.01,
        "tolerance": 1e-5,
        "max_iter": 2000,
        "num_cores": 1,
        "feature_names": list(feature_names),
        "candidate_pair_ids": list(pair_ids),
        "files": files,
        "expected_package": {
            "name": "glinternet",
            "version": "1.0.13",
            "source_sha256": "316C5973FF55DEA0BA0BEBB6930449B12DDC21EBD77B32714BEC665643DBC076",
            "rocker_arm64_digest": "sha256:f7e09f3032a60a6446328448f458d5f47189d83d7776ac2f0c33073cd781edda",
        },
    }
    request_path = request_root / "request.json"
    request_path.write_text(
        json.dumps(request, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    response_dir = request_root / "response"
    response_dir.mkdir()
    return (
        ComparatorInvocation(
            request_path, response_dir, adapter_root, authorization_context
        ),
        pairs,
        np.asarray(split["y_test"], dtype=int),
    )


def _load_pinned_adapter(adapter_root: Path) -> Any:
    adapter_path = adapter_root / "python_adapter.py"
    spec = importlib.util.spec_from_file_location(
        "track_a_pinned_glinternet_adapter", adapter_path
    )
    if spec is None or spec.loader is None:
        raise StructuredComparatorRuntimeError("Cannot load pinned glinternet adapter")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def production_glinternet_invoker(
    invocation: ComparatorInvocation,
) -> ComparatorResponse:
    """Invoke only the hash-bound existing adapter/container contract."""

    adapter = _load_pinned_adapter(invocation.adapter_root)
    if invocation.authorization_context is None:
        raise StructuredComparatorRuntimeError(
            "Scoped subprocess authorization context is required"
        )
    candidate_root = invocation.adapter_root.resolve().parents[1]
    adapter.validate_scoped_subprocess_authorization(
        candidate_root, invocation.authorization_context
    )
    contract = adapter.validate_build_contract(invocation.adapter_root)
    if (
        contract.get("scoped_subprocess_authorized") is not True
        or contract.get("scientific_compute_authorized") is not False
    ):
        raise StructuredComparatorRuntimeError(
            "Pinned build contract lacks scoped-only subprocess authorization"
        )
    command = adapter.render_container_command(
        invocation.request_path,
        invocation.response_dir,
        image_reference=adapter.RUNTIME_IMAGE_TAG,
    )
    try:
        subprocess.run(command, check=True, timeout=invocation.timeout_seconds)
    except subprocess.TimeoutExpired as error:
        raise StructuredComparatorRuntimeError(
            f"Pinned comparator exceeded {invocation.timeout_seconds:.1f} seconds"
        ) from error
    validation = adapter.validate_response(
        invocation.request_path, invocation.response_dir
    )
    with (invocation.response_dir / "pair_scores.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        pair_rows = tuple(dict(row) for row in csv.DictReader(handle))
    with (invocation.response_dir / "predictions.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        prediction_rows = tuple(dict(row) for row in csv.DictReader(handle))
    probabilities = tuple(float(row["probability_class_1"]) for row in prediction_rows)
    return ComparatorResponse(
        validation=dict(validation),
        pair_rows=pair_rows,
        probabilities_class_1=probabilities,
        command=tuple(command),
    )


def _status_result(
    registry: StructuredComparatorRegistry,
    comparator_id: str,
    status: str,
    reason: str,
) -> StructuredSensitivityResult:
    return StructuredSensitivityResult(
        registry_sha256=registry.canonical_sha256,
        status_rows=(
            {
                "comparator_id": comparator_id,
                "analysis_role": "sensitivity_only",
                "primary_estimand": False,
                "status": status,
                "reason": reason,
                "substitution_applied": False,
            },
        ),
        metric_rows=(),
        selected_edge_rows=(),
        raw_rows=(),
    )


def run_structured_sensitivity_cell(
    dataset: legacy.Dataset,
    *,
    registry: StructuredComparatorRegistry,
    cell_metadata: Mapping[str, Any],
    execution_enabled: bool,
    split_seed: int,
    candidate_root: str | Path,
    invoker: ComparatorInvoker = production_glinternet_invoker,
    authorization_context: Mapping[str, Any] | None = None,
) -> StructuredSensitivityResult:
    """Run the isolated comparator, or emit explicit N/A/disabled status only."""

    eligible = registry.eligible_comparators_for(cell_metadata)
    comparator_id = (
        registry.comparators[0].comparator_id
        if registry.comparators
        else "glinternet.strong_hierarchy.pairwise"
    )
    if not eligible:
        return _status_result(
            registry, comparator_id, "N/A", "cell_outside_registered_applicability"
        )
    if eligible != (comparator_id,):
        raise StructuredComparatorRuntimeError(
            "Structured registry returned an ambiguous comparator set"
        )
    if not registry.registry_active:
        return _status_result(
            registry, comparator_id, "DISABLED", "registry_active_false"
        )
    if not isinstance(execution_enabled, bool) or not execution_enabled:
        return _status_result(
            registry, comparator_id, "DISABLED", "execution_enabled_false"
        )

    root = Path(candidate_root).resolve()
    adapter_root = root / "comparators" / "glinternet"
    if not adapter_root.is_dir():
        raise StructuredComparatorRuntimeError("Pinned glinternet adapter is missing")
    with tempfile.TemporaryDirectory(prefix="track-a-glinternet-") as directory:
        invocation, pairs, y_test = _prepare_request(
            dataset,
            int(split_seed),
            Path(directory),
            adapter_root,
            authorization_context,
        )
        response = invoker(invocation)
        if not isinstance(response, ComparatorResponse):
            raise StructuredComparatorRuntimeError(
                "Comparator invoker returned an unknown response type"
            )
        if len(response.pair_rows) != len(pairs):
            raise StructuredComparatorRuntimeError(
                "Comparator response does not cover the pair universe"
            )
        scores: list[float] = []
        selected: list[tuple[int, int]] = []
        for expected_pair, row in zip(pairs, response.pair_rows, strict=True):
            observed_pair = (int(row["left_index"]), int(row["right_index"]))
            if observed_pair != expected_pair:
                raise StructuredComparatorRuntimeError(
                    "Comparator pair response order differs from request"
                )
            score = float(row["score"])
            if not math.isfinite(score) or score < 0:
                raise StructuredComparatorRuntimeError(
                    "Comparator score must be finite and non-negative"
                )
            if str(row["score_direction"]) != "higher_is_better":
                raise StructuredComparatorRuntimeError(
                    "Comparator score direction mismatch"
                )
            selected_flag = str(row["selected"]).lower()
            if selected_flag not in {"true", "false"}:
                raise StructuredComparatorRuntimeError("Invalid selected flag")
            if (selected_flag == "true") != (score > 0):
                raise StructuredComparatorRuntimeError(
                    "Selected flag and comparator score disagree"
                )
            scores.append(score)
            if selected_flag == "true":
                selected.append(observed_pair)
        p1 = np.asarray(response.probabilities_class_1, dtype=float)
        if (
            p1.ndim != 1
            or len(p1) != len(y_test)
            or not np.isfinite(p1).all()
            or np.any((p1 < 0) | (p1 > 1))
        ):
            raise StructuredComparatorRuntimeError(
                "Comparator probabilities are invalid or incomplete"
            )
        probability = np.c_[1.0 - p1, p1]
        metric = {
            "comparator_id": comparator_id,
            "analysis_role": "sensitivity_only",
            "primary_estimand": False,
            "status": "COMPLETED",
            "candidate_pair_count": len(pairs),
            "selected_edge_count": len(selected),
            "candidate_average_precision": support_metrics.candidate_wide_average_precision(
                pairs,
                scores,
                dataset.true_edges,
                score_direction="higher_is_better",
            ),
            **legacy.predictive_metrics(y_test, probability, np.asarray([0, 1])),
            **support_metrics.support_metrics(
                selected, dataset.true_edges, dataset.equivalence_groups
            ),
        }
        selected_rows = tuple(
            {
                "comparator_id": comparator_id,
                "analysis_role": "sensitivity_only",
                "primary_estimand": False,
                "rank": rank,
                "edge": f"{edge[0]}-{edge[1]}",
                "order": 2,
            }
            for rank, edge in enumerate(selected, start=1)
        )
        raw = {
            "comparator_id": comparator_id,
            "analysis_role": "sensitivity_only",
            "primary_estimand": False,
            "request_sha256": _sha256_file(invocation.request_path),
            "adapter_status": response.validation.get("status"),
            "package_version": response.validation.get("package_version"),
            "package_source_sha256": response.validation.get("package_source_sha256"),
            "pair_scores_json": json.dumps(scores, separators=(",", ":")),
            "probabilities_class_1_json": json.dumps(
                p1.tolist(), separators=(",", ":")
            ),
            "container_command_json": json.dumps(
                list(response.command), separators=(",", ":")
            ),
        }
    return StructuredSensitivityResult(
        registry_sha256=registry.canonical_sha256,
        status_rows=(
            {
                "comparator_id": comparator_id,
                "analysis_role": "sensitivity_only",
                "primary_estimand": False,
                "status": "COMPLETED",
                "reason": "applicable_and_executed",
                "substitution_applied": False,
            },
        ),
        metric_rows=(metric,),
        selected_edge_rows=selected_rows,
        raw_rows=(raw,),
    )
