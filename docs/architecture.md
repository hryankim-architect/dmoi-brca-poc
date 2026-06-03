# Architecture

This repo keeps a small, flat architecture. One Python process drives everything. The goal is reproducibility: a reviewer should be able to read the source, run `make run`, and trace every output back to a logged audit event.

## Control flow

`make run` (or `scripts/run_lab.sh` on a compute node) calls `dmoi_brca.pipeline.run_pipeline`. That function opens three concurrent concerns: it emits a structured audit event via `dmoi_brca.audit`, it opens an MLflow run context via `dmoi_brca.tracking`, and it executes the project body (VCF → HRD scoring, Cellpose segmentation, or classifier training depending on the stage). All three finish before the pipeline returns the artifact JSON and metrics.

## Pipeline stages, audit events, and outputs

| Stage | Audit event emitted | Primary output |
|---|---|---|
| Data fetch | `fetch.start` / `fetch.done` | checksum-verified files under `data/` |
| Pipeline run | `pipeline.start` / `pipeline.done` | artifact JSON under `artifacts/` |
| Canary probe | `canary.start` / `canary.pass` (or `canary.fail`) | exit code 0 or non-zero |
| Audit verify | — (read-only) | `(ok, n_entries, first_bad_ts)` tuple |

## Substrate integration points

The scaffold connects to the substrate through three loosely-coupled channels. Each one is optional: remove the environment variable and the channel silently becomes a no-op. The local NDJSON ledger on disk stays authoritative for audit regardless of whether the remote post succeeds.

| Channel | Module | Env var | Endpoint |
|---|---|---|---|
| Audit (immutable record) | `dmoi_brca.audit` | `AUDIT_HOST` | `http://${AUDIT_HOST}/events` |
| MLflow (experiment tracking) | `dmoi_brca.tracking` | `MLFLOW_TRACKING_URI` | configurable |
| Canary (daily probe) | `dmoi_brca.canary` | `BIOSCAFFOLD_CANARY_FIXTURE` | invoked by `lab_semantic_check.py` |

## How the audit ledger works

Each NDJSON entry carries a `prev_hash` field set to the SHA-256 of the canonical JSON (keys sorted, no extra whitespace) of the entry before it. Any insertion or modification invalidates the hash of every subsequent entry. The `audit.verify()` function walks the chain in one pass and returns `(ok, n_entries, first_bad_ts)`.

On the substrate this runs at roughly 6.19 µs per entry up to 10k entries, with a tamper-detect time of about 6 ms for a full chain re-verify. This repo does not exercise that scale, but it uses the same format so the substrate's `gatk_audit.py` verifier works against it without modification.

## Why MLflow

Three reasons:

1. Parameters and metrics are version-controlled alongside the run, so the demo output is reproducible without retaining a separate notes file.
2. The no-op wrapper means `make run` succeeds on a laptop with no MLflow server. A reviewer cloning this repo gets a working run regardless of their environment.
3. When the substrate is available, other repos in this portfolio post to the same MLflow server, so runs can be compared across projects in one UI.

## Why a deterministic canary

The canary is what `lab_semantic_check.py` probes on a daily schedule. It must be:

- Fixture-driven, so the result is deterministic across machines.
- Under 30 seconds end-to-end.
- Exit 0 on success, non-zero on any deviation from the expected output.
- Self-contained, no external services.

A consistently green canary means substrate-level monitoring detects regressions in this repo without any repo-specific alerting infrastructure.

## What this architecture intentionally avoids

- No microservices.
- No async runtime.
- No process supervisor.
- No container per pipeline stage (single Python process throughout).
- Input validation is ad hoc (Pydantic where it helps), not a framework.
- No DAG engine (Nextflow, Airflow, etc.). Those belong inside the pipeline body when a project needs them, not in the scaffold itself.

The scaffold defines the contract. The body implements the science.
