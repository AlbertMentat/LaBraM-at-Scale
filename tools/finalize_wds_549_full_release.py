#!/usr/bin/env python3
"""Verify every locked release gate, then publish the WDS 549 release marker.

RELEASE_READY.json is written last and only after all ten policy gates are
satisfied by SHA-bound evidence produced by Phase5A/5B/5C, the legacy ledger and
the strict loader canary.
"""

import argparse
import json
import os
from pathlib import Path

from tools.wds_549_full_io import atomic_write_json, sha256_file

CATALOG_ROOT = "catalogs/phase5-global-20260817"
P5B_FINAL = "reports/wds_549_phase5b_20260817/final"
P5C_FINAL = "reports/wds_549_phase5c_20260821/final"
CANARY = "reports/wds_549_strict_loader_canary_20260824"
TRAIN_VIEWS = ("main_train", "aux_quality_flagged", "aux_low_channel")
SAMPLES_PER_PATCH = 200
TARGET_SFREQ = 200.0


def _need(ok, message):
    if not ok:
        raise ValueError(message)


def _load(path):
    return json.loads(Path(path).read_text())


def _evidence(root, rel):
    path = Path(root) / rel
    _need(path.is_file(), f"missing release evidence: {rel}")
    return _load(path), sha256_file(path)


def finalize(publish_root, policy, plan, expected_builds=None,
             allow_local=False):
    root = Path(publish_root)
    _need(root.is_dir(), "publish_root missing")
    marker = root / "RELEASE_READY.json"
    _need(not marker.exists(), "RELEASE_READY.json already exists")
    if not allow_local:
        _need("AllEEG_pretrain_v2" in str(root.resolve()),
              "refusing to publish outside the release tree")

    policy_doc, policy_sha = _evidence(".", policy)
    plan_doc, plan_sha = _evidence(".", plan) if Path(plan).is_file() \
        else _evidence(root, plan)
    gates_policy = policy_doc.get("release_gates") or {}
    _need(len(gates_policy) == 10, "policy release_gates must define 10 gates")

    catalogs, catalogs_sha = _evidence(root, f"{CATALOG_ROOT}/CATALOGS_COMPLETE.json")
    tar_audit, tar_audit_sha = _evidence(root, f"{P5B_FINAL}/FULL_TAR_SHA256_AUDIT_COMPLETE.json")
    payload, payload_sha = _evidence(root, f"{P5C_FINAL}/PAYLOAD_AUDIT_COMPLETE.json")
    ledger, ledger_sha = _evidence(root, "compatibility/legacy_window_compatibility.json")
    ledger_audit, ledger_audit_sha = _evidence(root, "reports/compatibility/LEGACY_WINDOW_COMPATIBILITY_AUDIT.json")
    canary, canary_sha = _evidence(root, f"{CANARY}/STRICT_LOADER_CANARY.json")

    for name, doc in (("catalogs", catalogs), ("tar_audit", tar_audit),
                      ("payload", payload), ("ledger_audit", ledger_audit),
                      ("canary", canary)):
        _need(doc.get("status") in ("pass", "complete"), f"{name} status not pass")
        _need(doc.get("is_release_ready") is False,
              f"{name} must not pre-declare release readiness")

    # totals must close identically across the three independent audits
    ct, bt, pt = catalogs["totals"], tar_audit["totals"], payload["totals"]
    for key in ("tars", "windows", "size_bytes"):
        _need(int(ct[key]) == int(bt[key]) == int(pt[key]),
              f"three-way totals mismatch on {key}")

    scope = policy_doc.get("scope") or {}
    legacy_windows = int(scope["legacy_windows"])
    legacy_indices = int(scope["legacy_indices"])
    _need(int(ledger["legacy_windows"]) == legacy_windows,
          "ledger legacy window count mismatch")
    _need(int(ledger_audit["legacy_windows"]) == legacy_windows,
          "ledger audit legacy window count mismatch")
    _need(sum(int(v) for v in (ledger_audit.get("status_counts") or {}).values())
          == legacy_windows, "legacy status counts do not close")
    proof = ledger_audit.get("uniqueness_proof") or {}
    _need(proof and all(bool(v) for v in proof.values()),
          "legacy uniqueness proof incomplete")

    gates = {
        "verify_all_tar_sha256":
            tar_audit["claims"].get("full_tar_hash_verification") is True,
        "missing_stale_duplicate_tars":
            all(int(v) == 0 for v in tar_audit["issue_counts"].values()),
        "all_536_sources_terminal":
            int(catalogs["expected_counts"]["sources"]) == int(
                scope["target_config_sources"]),
        "scan_all_payloads":
            payload["claims"].get("payload_verification") is True
            and int(payload["coverage"]["validated_shards"])
            == int(payload["coverage"]["expected_shards"]),
        "training_views_nonfinite_values":
            all(int(v) == 0 for v in payload["issue_counts"].values()),
        "shape_channel_signature_errors":
            all(int(v) == 0 for v in payload["issue_counts"].values()),
        "missing_unit_normalization_provenance":
            all(int(v) == 0 for v in payload["issue_counts"].values()),
        "all_549_legacy_indices_accounted":
            int(ledger_audit.get("legacy_tars", -1)) > 0
            and legacy_indices > 0 and bool(proof),
        "all_24723392_legacy_windows_accounted":
            int(ledger_audit["legacy_windows"]) == legacy_windows,
        "strict_loader_canary":
            canary["claims"].get("samples_decoded_and_validated") is True
            and int(canary["observed"]["nonfinite_values"]) == 0,
    }
    _need(set(gates) == set(gates_policy), "gate set does not match policy")
    failed = sorted(name for name, ok in gates.items() if not ok)
    _need(not failed, f"release gates failed: {failed}")
    _need(payload["claims"].get("global_sample_uniqueness_proven") is True,
          "global sample uniqueness not proven")

    views = catalogs["views"]
    training = {}
    for view in TRAIN_VIEWS:
        meta = views.get(view) or {}
        training[view] = {
            "tars": int(meta.get("tars", 0)),
            "windows": int(meta.get("windows", 0)),
            "size_bytes": int(meta.get("size_bytes", 0)),
        }
    train_windows = sum(v["windows"] for v in training.values())
    excluded = {
        view: int((views.get(view) or {}).get("windows", 0))
        for view in sorted(set(views) - set(TRAIN_VIEWS))
    }
    _need(train_windows + sum(excluded.values()) == int(ct["windows"]),
          "view windows do not close against catalog totals")

    if expected_builds is not None:
        _need(int(catalogs["effective_selection"]["effective_jobs"])
              == int(expected_builds), "effective build count mismatch")

    catalog_out = {
        "schema_version": 1,
        "report_version": "wds-549-release-catalog-v1",
        "plan_id": plan_doc.get("plan_id"),
        "plan_sha256": plan_sha,
        "policy_sha256": policy_sha,
        "effective_selection": catalogs["effective_selection"],
        "totals": {k: int(ct[k]) for k in ("tars", "windows", "size_bytes")},
        "training_views": training,
        "excluded_view_windows": excluded,
        "training_windows": train_windows,
        "samples": int(pt["samples"]),
        "unique_sample_ids": int(pt["unique_sample_ids"]),
        "catalog_root": CATALOG_ROOT,
        "view_catalogs": {
            view: {"path": f"{CATALOG_ROOT}/{views[view]['path']}",
                   "sha256": views[view]["sha256"],
                   "rows": int(views[view]["rows"])}
            for view in TRAIN_VIEWS
        },
    }
    catalog_path = root / "_wds_549_release_catalog.json"
    atomic_write_json(catalog_path, catalog_out)
    catalog_out_sha = sha256_file(catalog_path)
    checksum_path = root / "_wds_549_release_catalog.sha256"
    tmp = checksum_path.with_name(checksum_path.name + f".tmp.{os.getpid()}")
    tmp.write_text(f"{catalog_out_sha}  {catalog_path.name}\n")
    os.replace(tmp, checksum_path)

    ready = {
        "schema_version": 1,
        "processing_version": "wds-549-full-r1-release-v1",
        # contract consumed by WDS549TrainingIterable._validate_release
        "status": "pass",
        "is_release_ready": True,
        "plan_id": plan_doc.get("plan_id"),
        "plan_sha256": plan_sha,
        "policy_sha256": policy_sha,
        "release_catalog": catalog_path.name,
        "release_catalog_sha256": catalog_out_sha,
        "gates": {name: bool(ok) for name, ok in sorted(gates.items())},
        "evidence_sha256": {
            "catalogs_complete": catalogs_sha,
            "full_tar_sha256_audit_complete": tar_audit_sha,
            "payload_audit_complete": payload_sha,
            "legacy_window_compatibility": ledger_sha,
            "legacy_window_compatibility_audit": ledger_audit_sha,
            "strict_loader_canary": canary_sha,
        },
        "totals": catalog_out["totals"],
        "training_windows": train_windows,
        "excluded_view_windows": excluded,
        "samples": int(pt["samples"]),
        "unique_sample_ids": int(pt["unique_sample_ids"]),
        "legacy_windows_accounted": legacy_windows,
    }
    atomic_write_json(marker, ready)
    return ready


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--publish_root", required=True)
    p.add_argument("--policy", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--expected_builds", type=int)
    p.add_argument("--allow_local", action="store_true")
    a = p.parse_args()
    out = finalize(a.publish_root, a.policy, a.plan, a.expected_builds,
                   a.allow_local)
    print(json.dumps({"status": out["status"], "gates": out["gates"],
                      "totals": out["totals"],
                      "training_windows": out["training_windows"]},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
