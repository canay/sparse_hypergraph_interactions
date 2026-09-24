"""Narrow V2 adapter for the Track-A blind pilot validator.

V1 incorrectly required ``config_resolved.json`` to preserve the source
configuration's byte serialization.  The runner deliberately writes the same
JSON document in canonical form, so content identity—not byte identity—is the
correct invariant for that one resolved snapshot.  Every frozen/live binding
and every other V1 predicate remains unchanged.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import durability
import validate_blind_pilot as v1


SCHEMA_VERSION = 1
VALIDATOR_ID = "track-a-blind-pilot-structural-validator-v2"
CORRECTION_ID = "V1_RESOLVED_CONFIG_SERIALIZATION_FALSE_FAILURE"


def validate_blind_pilot_v2(
    *,
    pilot_root: str | Path,
    freeze_path: str | Path,
    expected_freeze_sha256: str,
    candidate_root: str | Path,
    project_root: str | Path,
    preflight_path: str | Path,
) -> dict[str, Any]:
    """Run V1 with one narrowly scoped semantic-equivalence correction."""

    candidate = Path(candidate_root).resolve()
    source_config = candidate / "config" / "pilot_config.json"
    original_require_file_hash = v1._require_file_hash

    def require_file_hash_v2(path: Path, expected: Any, code: str) -> str:
        if code != "RESOLVED_CONFIG_HASH_MISMATCH":
            return original_require_file_hash(path, expected, code)

        # The frozen source file itself has already passed the original byte
        # hash check earlier in BP03.  Only its runner-emitted snapshot may
        # differ in whitespace/key ordering.
        source_document = v1._read_json(source_config, "CONFIG_UNREADABLE")
        resolved_document = v1._read_json(path, code)
        if resolved_document != source_document:
            v1._fail(code)
        return durability.sha256_file(path)

    v1._require_file_hash = require_file_hash_v2
    try:
        result = v1.validate_blind_pilot(
            pilot_root=pilot_root,
            freeze_path=freeze_path,
            expected_freeze_sha256=expected_freeze_sha256,
            candidate_root=candidate,
            project_root=project_root,
            preflight_path=preflight_path,
        )
    finally:
        v1._require_file_hash = original_require_file_hash

    result["validator_id"] = VALIDATOR_ID
    result["correction_scope"] = CORRECTION_ID
    return result


def write_v2_artifacts(
    result: Mapping[str, Any],
    *,
    output_dir: str | Path,
    sealed_pilot_root: str | Path,
) -> tuple[Path, Path]:
    """Write allowlisted V2 predicates and a self-hashed audit receipt."""

    output = Path(output_dir).resolve()
    sealed = Path(sealed_pilot_root).resolve()
    try:
        output.relative_to(sealed)
    except ValueError:
        pass
    else:
        raise ValueError("Validator output must stay outside the sealed pilot output")

    output.mkdir(parents=True, exist_ok=True)
    predicate_path = output / "BLIND_PILOT_PREDICATES_V2.json"
    receipt_path = output / "BLIND_PILOT_VALIDATOR_RECEIPT_V2.json"
    if predicate_path.exists() or receipt_path.exists():
        raise FileExistsError("Refusing to overwrite blind-pilot V2 artifacts")

    durability.atomic_write_json(predicate_path, dict(result))
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "validator_id": VALIDATOR_ID,
        "validation_mode": "READ_ONLY_BLIND_STRUCTURAL_NO_SCIENTIFIC_INFERENCE",
        "correction_scope": CORRECTION_ID,
        "overall_result": result["overall_result"],
        "predicate_artifact_sha256": durability.sha256_file(predicate_path),
        "base_validator_source_sha256": durability.sha256_file(Path(v1.__file__)),
        "correction_adapter_sha256": durability.sha256_file(Path(__file__)),
        "scientific_inference": "PROHIBITED",
    }
    durability.atomic_write_json(receipt_path, receipt)
    return predicate_path, receipt_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pilot-output", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--expected-freeze-sha256", required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = validate_blind_pilot_v2(
        pilot_root=args.pilot_output,
        freeze_path=args.freeze,
        expected_freeze_sha256=args.expected_freeze_sha256,
        candidate_root=args.candidate_root,
        project_root=args.project_root,
        preflight_path=args.preflight,
    )
    write_v2_artifacts(
        result,
        output_dir=args.output_dir,
        sealed_pilot_root=args.pilot_output,
    )
    return 0 if result["overall_result"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
