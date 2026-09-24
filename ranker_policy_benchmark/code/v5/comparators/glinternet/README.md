# Inactive glinternet comparator adapter

Date/time: 2026-08-26 15:38 +03:00  
Tool: Codex  
Model, if known: GPT-5.6 Sol Extra High  
Operation ID: `shil-pr3-inactive-adapter-package-20260826-1531`

Status: `STATIC_SCHEMA_TESTED; CONTAINER_NOT_BUILT; REAL_R_SMOKE_PENDING;
REGISTRY_INACTIVE; NO_SCIENTIFIC_COMPUTE`.

This package turns the PR-3 install plan into an inspectable but unexecuted
adapter smoke contract. It is restricted to pairwise, strong-hierarchy cells
and is not a third member of the full pair/triple ranker registry.

## Locked identities

- Base: `rocker/r-ver:4.6.1` for `linux/arm64`, platform digest
  `sha256:f7e09f3032a60a6446328448f458d5f47189d83d7776ac2f0c33073cd781edda`.
- Package: CRAN `glinternet 1.0.13`, GPL-2, source tarball SHA-256
  `316C5973FF55DEA0BA0BEBB6930449B12DDC21EBD77B32714BEC665643DBC076`.
- Adapter: `glinternet-r-file-v1`, zero-based pair IDs at the Python boundary
  and explicit one-based conversion only inside the R process.

The Docker registry manifest and CRAN tarball bytes were queried read-only on
2026-08-26. No image was pulled, no package was installed, and no remote host
was modified.

## Boundary

The R adapter fits one `glinternet()` regularization path on the training split,
selects the lambda with minimum validation log loss, maps every declared
`interactionPairs` candidate to a nonnegative coefficient-magnitude score, and
predicts only after selection on the held-out test split. It sets
`numCores=1`, does not call `glinternet.cv()`, and rejects undeclared pairs.

Python validates exact file hashes, shapes, feature order, pair identity,
package/image metadata, validation-path uniqueness, complete candidate scores,
and held-out probability shape. It can render the locked Docker command but has
no execution function.

## Current evidence and remaining gate

The deterministic fixture and mock response pass without requiring Docker, R,
or glinternet. These tests establish only schema and boundary behavior. They do
not establish that the ARM64 image builds or that the real package produces the
mock numerical values.

Verification on 2026-08-26: comparator-specific tests `9/9 PASS`, complete
candidate suite `106/106 PASS`, and Black `25/25 PASS`.

The following commands are a future human-authorized gate and were **not run**:

```text
docker build --platform=linux/arm64 --pull=false --file=Dockerfile --tag=track-a-glinternet-smoke:1.0.13-r4.6.1-arm64 .
docker image inspect track-a-glinternet-smoke:1.0.13-r4.6.1-arm64
```

After a successful build, the built image ID must be recorded, the response
directory must start empty, runtime networking must remain disabled, and the
real response must pass `python_adapter.validate_response()`. Until that live
container smoke and the protocol applicability gate pass,
`structured_comparator_candidate.registry_active=false` remains mandatory.

From this directory on the documented ARM64 VPS, the exact pending smoke is:

```text
python -B -c 'import python_adapter as a; a.validate_build_contract(".")'
docker build --platform=linux/arm64 --pull=false --file=Dockerfile --tag=track-a-glinternet-smoke:1.0.13-r4.6.1-arm64 .
docker image inspect --format '{{.Id}} {{index .Config.Labels "org.opencontainers.image.base.digest"}} {{index .Config.Labels "org.opencontainers.image.glinternet.source-sha256"}}' track-a-glinternet-smoke:1.0.13-r4.6.1-arm64
mkdir _smoke_response_a _smoke_response_b
docker run --rm --network=none --cpus=1 --memory=2g --read-only --tmpfs=/tmp:rw,noexec,nosuid,size=256m --mount=type=bind,src="$(pwd)/fixture",dst=/input,readonly --mount=type=bind,src="$(pwd)/_smoke_response_a",dst=/output track-a-glinternet-smoke:1.0.13-r4.6.1-arm64 --train-x=/input/train_x.csv --train-y=/input/train_y.csv --validation-x=/input/validation_x.csv --validation-y=/input/validation_y.csv --test-x=/input/test_x.csv --candidate-pairs=/input/candidate_pairs.csv --output-dir=/output --seed=260826 --n-lambda=12 --lambda-min-ratio=0.05 --tolerance=1e-05 --max-iter=2000 --num-cores=1 --fixture-id=glinternet-tiny-binomial-v1
docker run --rm --network=none --cpus=1 --memory=2g --read-only --tmpfs=/tmp:rw,noexec,nosuid,size=256m --mount=type=bind,src="$(pwd)/fixture",dst=/input,readonly --mount=type=bind,src="$(pwd)/_smoke_response_b",dst=/output track-a-glinternet-smoke:1.0.13-r4.6.1-arm64 --train-x=/input/train_x.csv --train-y=/input/train_y.csv --validation-x=/input/validation_x.csv --validation-y=/input/validation_y.csv --test-x=/input/test_x.csv --candidate-pairs=/input/candidate_pairs.csv --output-dir=/output --seed=260826 --n-lambda=12 --lambda-min-ratio=0.05 --tolerance=1e-05 --max-iter=2000 --num-cores=1 --fixture-id=glinternet-tiny-binomial-v1
python -B -c 'import python_adapter as a; print(a.validate_response("fixture/request.json", "_smoke_response_a")); print(a.validate_response("fixture/request.json", "_smoke_response_b"))'
(cd _smoke_response_a && sha256sum *.csv | sort) > /tmp/glinternet-smoke-a.sha256
(cd _smoke_response_b && sha256sum *.csv | sort) > /tmp/glinternet-smoke-b.sha256
diff -u /tmp/glinternet-smoke-a.sha256 /tmp/glinternet-smoke-b.sha256
```

Prerequisites are Linux/ARM64, Docker with BuildKit checksum support, Docker
daemon access, at least one CPU, 2 GiB free RAM, sufficient image-build disk,
and outbound access to Docker Hub and CRAN during build only. The runtime phase
must have no network. Build output, image ID/labels, both response hashes, R and
compiler identity, and validator results must be captured before any activation.

## 2026-08-26 real ARM64 double-smoke result

Date/time: 2026-08-26 15:44 +03:00  
Tool: Codex  
Model, if known: GPT-5.6 Sol Extra High (user-attested; runtime exposed GPT-5 family)  
Operation ID: `shil-pr3-arm64-double-smoke-20260826-1544`

The user had explicitly authorized using the idle MTA/VPS resources for the
parallel method track. The pending non-scientific gate above was therefore run
on the documented ARM64 VPS in a new, previously absent remote directory.

- The pinned Rocker image resolved at the declared digest and `glinternet
  1.0.13` compiled from the hash-verified CRAN tarball under R 4.6.1/gcc 13.3.
- Built image ID:
  `sha256:f20fc7e80ccaaaf06292d91e5cf94247f3cad56576e9ee6f5dca755194b49164`.
- Both fixture runs used `--network=none`, one CPU, 2 GiB memory, a read-only
  root filesystem, the same seed, and separate initially empty output folders.
- Both real responses passed `validate_response()`: six declared pairs, lambda
  index 11, selected support `f0__f1`, and eight held-out predictions.
- All four response CSV hashes were identical across the two runs. The pulled
  evidence manifest verifies `20/20` files.

Evidence is bound by
`evidence/vps_arm64_glinternet_smoke_20260826/SMOKE_RESULT.json`. The original
`build_contract.json` remains an immutable pre-execution contract; it was not
rewritten after observing the smoke. Current comparator status is
`REAL_ARM64_DOUBLE_SMOKE_PASS; REGISTRY_INACTIVE; NO_SCIENTIFIC_COMPUTE`.
Registry activation and the scientific pilot remain prohibited until the
applicability-restricted registry amendment and experiment freeze/launch gates
pass.
