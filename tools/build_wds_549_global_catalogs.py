#!/usr/bin/env python3
"""Phase 5A gate for global source/build terminal checks and per-view tar catalogs."""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.audit_wds_549_build_states import audit as audit_build_states  # noqa: E402
from tools.wds_549_full_io import atomic_write_json, sha256_file  # noqa: E402

DEFAULT_COUNTS = {
    "sources": 536,
    "planned_sources": 529,
    "archive_only_sources": 7,
    "jobs": 767,
}
READY_STATUSES = {"ready_with_main", "ready_aux_only", "ready_archive_only"}
TUH_LOGICAL_SOURCE = "TUH_full"
TUH_REPLACEMENT_SOURCE_KIND = "processed_fallback"
PAIRED_SHORT_SOURCES = {"Alljoined-processed-v3", "Things-EEG-processed"}
PAIRED_SHORT_SOURCE_KIND = "paired_image_eeg_cache"
NATURAL_SHORT_SOURCES = {
    "EEGWorkload", "ON_ds004771", "ON_ds004785",
    "ZEN_29601", "ZEN_7149276", "ZEN_7891964",
}
SOURCE_REPAIR_SOURCES = {"ON_ds003620", "ON_ds004554"}
ON004395_CHANNEL_REPAIR_SOURCES = {"ON_ds004395"}
FACED_NEMAR_SOURCES = {"FACED"}
ACCEPTED_TERMINAL_SELECTION_VERSIONS = {
    "wds-549-accepted-terminals-20260818.1",
    "wds-549-accepted-terminals-20260821.1",
}
SHA64 = re.compile(r"^[0-9a-f]{64}$")


def _j(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def atomic_write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    h = hashlib.sha256()
    total = 0
    n = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for row in rows:
            line = _j(row)
            payload = (line + "\n").encode("utf-8")
            f.write(line)
            f.write("\n")
            h.update(payload)
            total += len(payload)
            n += 1
    os.replace(tmp, path)
    return {"sha256": h.hexdigest(), "bytes": total, "rows": n}


def _need(ok, msg):
    if not ok:
        raise ValueError(msg)


def _load(path):
    return json.loads(Path(path).read_text())


def _normalize_views(d):
    out = {}
    for k, v in d.items():
        iv = int(v)
        _need(iv >= 0, f"negative view count: {k}")
        if iv > 0:
            out[k] = iv
    return dict(sorted(out.items()))


def _infer_status(views):
    if int(views.get("main_train", 0)) > 0:
        return "ready_with_main"
    if int(views.get("aux_low_channel", 0)) + int(views.get("aux_quality_flagged", 0)) > 0:
        return "ready_aux_only"
    return "ready_archive_only"


def _validate_counts(expected_counts):
    c = dict(DEFAULT_COUNTS if expected_counts is None else expected_counts)
    _need(set(c.keys()) == {"sources", "planned_sources", "archive_only_sources", "jobs"}, "expected_counts keys mismatch")
    for k, v in c.items():
        _need(isinstance(v, int) and v > 0, f"expected_counts[{k}] invalid")
    _need(c["planned_sources"] + c["archive_only_sources"] == c["sources"], "expected_counts partition mismatch")
    return c


def _validate_registry(plan_doc, registry_doc, expected):
    summary = registry_doc.get("summary")
    _need(summary == {
        "ready": expected["planned_sources"],
        "ready_archive_only": expected["archive_only_sources"],
        "blocked": 0,
        "sources": expected["sources"],
        "blockers": {},
    }, "registry summary mismatch")

    entries = registry_doc.get("entries")
    _need(isinstance(entries, list) and len(entries) == expected["sources"], "registry entries mismatch")

    reg = {}
    st = Counter()
    for e in entries:
        s = e.get("logical_source")
        k = e.get("source_kind")
        t = e.get("status")
        b = e.get("blockers")
        _need(isinstance(s, str) and s, "registry logical_source missing")
        _need(isinstance(k, str) and k, f"registry source_kind missing: {s}")
        _need(t in {"ready", "ready_archive_only"}, f"registry status invalid: {s}")
        _need(isinstance(b, list) and not b, f"registry blockers nonempty: {s}")
        _need(s not in reg, f"registry duplicate source: {s}")
        reg[s] = {"source_kind": k, "status": t}
        st[t] += 1
    _need(st["ready"] == expected["planned_sources"], "registry ready count mismatch")
    _need(st["ready_archive_only"] == expected["archive_only_sources"], "registry ready_archive_only count mismatch")

    buildable = {}
    for j in plan_doc.get("jobs", []):
        s = j["logical_source"]
        k = j["source_kind"]
        if s in buildable and buildable[s] != k:
            raise ValueError(f"plan source_kind drift in jobs: {s}")
        buildable[s] = k
    archive = {}
    for a in plan_doc.get("archive_only_sources", []):
        s = a["logical_source"]
        k = a["source_kind"]
        if s in archive and archive[s] != k:
            raise ValueError(f"plan source_kind drift in archive_only_sources: {s}")
        archive[s] = k
    _need(len(buildable) == expected["planned_sources"], "plan buildable source count mismatch")
    _need(len(archive) == expected["archive_only_sources"], "plan archive-only source count mismatch")
    _need(not (set(buildable) & set(archive)), "plan buildable/archive source overlap")
    full = {**buildable, **archive}
    _need(len(full) == expected["sources"], "plan total source count mismatch")
    _need(set(full.keys()) == set(reg.keys()), "source set mismatch between plan and registry")
    for s, k in full.items():
        _need(reg[s]["source_kind"] == k, f"registry source_kind mismatch: {s}")
        want = "ready" if s in buildable else "ready_archive_only"
        _need(reg[s]["status"] == want, f"registry status mismatch: {s}")
    return reg, buildable, archive

def _validate_record(rec, policy_views, build_id):
    req = ["path", "sha256", "size_bytes", "windows", "view", "n_channels", "n_patches", "channel_signature"]
    for k in req:
        _need(k in rec, f"catalog tar field missing {k}: {build_id}")
    rel = rec["path"]
    _need(isinstance(rel, str) and rel, f"catalog path invalid: {build_id}")
    _need(not Path(rel).is_absolute(), f"catalog path absolute: {build_id}")
    _need(rel.startswith("views/"), f"catalog path outside views/: {build_id}")
    _need(".." not in Path(rel).parts, f"catalog path traversal: {build_id}")
    _need(rec["view"] in policy_views, f"catalog view invalid: {build_id}")
    _need(bool(SHA64.fullmatch(str(rec["sha256"]))), f"catalog sha invalid: {build_id}")
    for k in ["size_bytes", "windows", "n_channels", "n_patches"]:
        _need(int(rec[k]) > 0, f"catalog {k} must be positive: {build_id}")
    _need(bool(str(rec["channel_signature"])), f"catalog channel_signature empty: {build_id}")


def _validate_build(
        job, identity, publish_root, policy_views, global_paths,
        build_root="corpus_builds", catalog_role="base"):
    bid = job["build_id"]
    _need(
        build_root in {"corpus_builds", "replacement_builds"},
        f"invalid build_root: {build_root}")
    _need(catalog_role in {"base", "replacement"}, "invalid catalog_role")
    target = Path(publish_root) / build_root / bid
    ready_path = target / "READY.json"
    manifest_path = target / "build_manifest.json"
    catalog_path = target / "build_catalog.json"
    terminal_path = target / "TERMINAL.json"

    _need(ready_path.is_file(), f"missing READY.json: {bid}")
    _need(manifest_path.is_file(), f"missing build_manifest.json: {bid}")
    _need(catalog_path.is_file(), f"missing build_catalog.json: {bid}")
    _need(not terminal_path.exists(), f"TERMINAL.json must be absent: {bid}")

    ready = _load(ready_path)
    manifest = _load(manifest_path)
    catalog = _load(catalog_path)

    for k, v in identity.items():
        _need(ready.get(k) == v, f"READY identity mismatch {k}: {bid}")
        _need(manifest.get(k) == v, f"manifest identity mismatch {k}: {bid}")
    _need(catalog.get("schema_version") == 1 and catalog.get("build_id") == bid, f"catalog identity mismatch: {bid}")

    for f in ["logical_source", "source_kind"]:
        if f in ready:
            _need(ready[f] == job[f], f"READY {f} mismatch: {bid}")
        if f in manifest:
            _need(manifest[f] == job[f], f"manifest {f} mismatch: {bid}")

    _need(ready.get("status") in READY_STATUSES, f"READY status invalid: {bid}")
    _need(manifest.get("status") in READY_STATUSES, f"manifest status invalid: {bid}")

    msha = sha256_file(manifest_path)
    csha = sha256_file(catalog_path)
    _need(ready.get("build_manifest_sha256") == msha, f"READY manifest SHA mismatch: {bid}")
    _need(ready.get("catalog_sha256") == csha, f"READY catalog SHA mismatch: {bid}")
    _need(manifest.get("catalog_sha256") == csha, f"manifest catalog SHA mismatch: {bid}")

    tar_files = catalog.get("tar_files")
    # ready_archive_only builds legitimately carry zero tars; the tar-set and
    # status cross-checks below still reject a wrongly emptied catalog.
    _need(isinstance(tar_files, list), f"catalog tar_files invalid: {bid}")

    seen = set()
    vwins = defaultdict(int)
    total_wins = 0
    total_bytes = 0
    tar_rows = []
    for rec in tar_files:
        _validate_record(rec, policy_views, bid)
        rel = rec["path"]
        _need(rel not in seen, f"catalog duplicate tar path: {bid}:{rel}")
        seen.add(rel)
        rel_release = f"{build_root}/{bid}/{rel}"
        _need(rel_release not in global_paths, f"duplicate release-relative tar path: {rel_release}")
        global_paths.add(rel_release)

        tar_path = target / rel
        _need(tar_path.is_file(), f"catalog tar missing: {bid}:{rel}")
        _need(tar_path.stat().st_size == int(rec["size_bytes"]), f"catalog tar size mismatch: {bid}:{rel}")

        wins = int(rec["windows"])
        vwins[rec["view"]] += wins
        total_wins += wins
        total_bytes += int(rec["size_bytes"])
        tar_rows.append({
            "build_id": bid,
            "build_root": build_root,
            "catalog_role": catalog_role,
            "logical_source": job["logical_source"],
            "source_kind": job["source_kind"],
            "path": rel,
            "release_relative_tar": rel_release,
            "sha256": rec["sha256"],
            "size_bytes": int(rec["size_bytes"]),
            "windows": wins,
            "view": rec["view"],
            "n_channels": int(rec["n_channels"]),
            "n_patches": int(rec["n_patches"]),
            "channel_signature": rec["channel_signature"],
        })

    actual = {p.relative_to(target).as_posix() for p in (target / "views").rglob("*.tar")} if (target / "views").exists() else set()
    _need(actual == seen, f"catalog tar set mismatch: {bid}")

    views = _normalize_views(vwins)
    status = _infer_status(views)
    _need(ready["status"] == status, f"READY status vs views mismatch: {bid}")
    _need(manifest["status"] == status, f"manifest status vs views mismatch: {bid}")

    _need(int(catalog.get("tars", -1)) == len(tar_files), f"catalog tars mismatch: {bid}")
    _need(int(catalog.get("windows", -1)) == total_wins, f"catalog windows mismatch: {bid}")
    _need(int(ready.get("tars", -1)) == len(tar_files), f"READY tars mismatch: {bid}")
    _need(int(ready.get("windows", -1)) == total_wins, f"READY windows mismatch: {bid}")
    manifest_tars = manifest.get("tars")
    if manifest_tars is not None:
        _need(int(manifest_tars) == len(tar_files), f"manifest tars mismatch: {bid}")
    _need(int(manifest.get("windows", -1)) == total_wins, f"manifest windows mismatch: {bid}")
    _need(_normalize_views(ready.get("views", {})) == views, f"READY views mismatch: {bid}")
    _need(_normalize_views(manifest.get("views", {})) == views, f"manifest views mismatch: {bid}")
    _need(isinstance(manifest.get("accounting"), dict), f"manifest accounting missing: {bid}")

    return {
        "build_id": bid,
        "build_root": build_root,
        "catalog_role": catalog_role,
        "logical_source": job["logical_source"],
        "source_kind": job["source_kind"],
        "status": status,
        "windows": total_wins,
        "tars": len(tar_files),
        "size_bytes": total_bytes,
        "views": views,
        "ready_sha256": sha256_file(ready_path),
        "build_manifest_sha256": msha,
        "catalog_sha256": csha,
        "ready_relpath": f"{build_root}/{bid}/READY.json",
        "manifest_relpath": f"{build_root}/{bid}/build_manifest.json",
        "catalog_relpath": f"{build_root}/{bid}/build_catalog.json",
        "tar_rows": tar_rows,
    }

def _validate_tuh_overlay(
        replacement_plan, replacement_contract, replacement_finalization,
        base_plan, base_plan_sha, policy_sha, publish_root):
    supplied = [replacement_plan, replacement_contract, replacement_finalization]
    if not any(supplied):
        return None
    _need(all(supplied), "TUH replacement overlay inputs must be all-or-none")

    replacement_plan = Path(replacement_plan)
    replacement_contract = Path(replacement_contract)
    replacement_finalization = Path(replacement_finalization)
    plan = _load(replacement_plan)
    contract = _load(replacement_contract)
    finalization = _load(replacement_finalization)
    plan_sha = sha256_file(replacement_plan)
    contract_sha = sha256_file(replacement_contract)
    finalization_sha = sha256_file(replacement_finalization)

    _need(plan.get("schema_version") == 1, "replacement plan schema mismatch")
    _need(contract.get("schema_version") == 1, "replacement contract schema mismatch")
    _need(contract.get("status") == "pass", "replacement contract status mismatch")
    _need(contract.get("logical_source") == TUH_LOGICAL_SOURCE,
          "replacement contract logical_source mismatch")
    _need(contract.get("source_kind") == TUH_REPLACEMENT_SOURCE_KIND,
          "replacement contract source_kind mismatch")
    _need(contract.get("mode") == "full_selected_pkl_replacement",
          "replacement contract mode mismatch")
    _need(contract.get("replacement_effective_catalog_role") ==
          "replacement_builds_only_for_tuh", "replacement catalog role mismatch")

    base_binding = contract.get("base_binding") or {}
    _need(base_binding.get("base_plan_id") == base_plan.get("plan_id"),
          "replacement contract base plan ID mismatch")
    _need(base_binding.get("base_plan_sha256") == base_plan_sha,
          "replacement contract base plan SHA mismatch")
    _need(base_binding.get("base_policy_sha256") == policy_sha,
          "replacement contract base policy SHA mismatch")
    _need(base_binding.get("effective_catalog_role") ==
          "audit_only_superseded_for_tuh", "base TUH catalog role mismatch")

    _need(plan.get("base_plan_id") == base_plan.get("plan_id"),
          "replacement plan base plan ID mismatch")
    _need(plan.get("base_plan_sha256") == base_plan_sha,
          "replacement plan base plan SHA mismatch")
    _need(plan.get("base_policy_sha256") == policy_sha,
          "replacement plan base policy SHA mismatch")
    _need(plan.get("replacement_contract_sha256") == contract_sha,
          "replacement plan contract SHA mismatch")
    inventory_sha = plan.get("inventory_sha256")
    _need(bool(SHA64.fullmatch(str(inventory_sha))),
          "replacement plan inventory SHA invalid")

    excluded = base_binding.get("base_tuh_raw_build_ids")
    _need(isinstance(excluded, list) and excluded,
          "base TUH replacement build IDs missing")
    _need(excluded == sorted(set(excluded)),
          "base TUH replacement build IDs must be sorted unique")
    summary = plan.get("summary") or {}
    _need(summary.get("base_tuh_raw_builds_excluded_from_effective_catalog") ==
          excluded, "replacement plan excluded base build IDs mismatch")
    base_tuh_jobs = [
        job for job in base_plan.get("jobs", [])
        if job.get("logical_source") == TUH_LOGICAL_SOURCE]
    _need({job["build_id"] for job in base_tuh_jobs} == set(excluded),
          "base TUH plan jobs mismatch replacement exclusion")

    jobs = plan.get("jobs")
    _need(isinstance(jobs, list) and jobs, "replacement jobs missing")
    build_ids = [job.get("build_id") for job in jobs]
    _need(len(set(build_ids)) == len(build_ids),
          "duplicate replacement build IDs")
    _need(len(jobs) == int(summary.get("jobs", -1)),
          "replacement job count mismatch")
    for job in jobs:
        _need(job.get("logical_source") == TUH_LOGICAL_SOURCE,
              "replacement job logical_source mismatch")
        _need(job.get("source_kind") == TUH_REPLACEMENT_SOURCE_KIND,
              "replacement job source_kind mismatch")
        _need(bool(SHA64.fullmatch(str(job.get("job_manifest_sha256")))),
              "replacement job manifest SHA invalid")

    _need(finalization.get("status") == "pass",
          "replacement finalization status mismatch")
    _need(finalization.get("is_release_ready") is False,
          "replacement finalization release flag mismatch")
    _need((finalization.get("input_hashes") or {}) == {
        "replacement_plan_sha256": plan_sha,
        "replacement_contract_sha256": contract_sha,
        "inventory_sha256": inventory_sha,
    }, "replacement finalization input hashes mismatch")
    report_ids = [
        row.get("build_id") for row in finalization.get("replacement_builds", [])]
    _need(set(report_ids) == set(build_ids) and len(report_ids) == len(build_ids),
          "replacement finalization build set mismatch")
    _need(int((finalization.get("summary") or {}).get("builds", -1)) == len(jobs),
          "replacement finalization summary build count mismatch")

    base_ready_hashes = {}
    for build_id in excluded:
        ready_path = Path(publish_root) / "corpus_builds" / build_id / "READY.json"
        _need(ready_path.is_file(), f"missing superseded base READY: {build_id}")
        base_ready_hashes[build_id] = sha256_file(ready_path)
    _need(finalization.get("base_raw_ready_hashes_audit_only") == base_ready_hashes,
          "replacement finalization base READY hashes mismatch")

    return {
        "jobs": jobs,
        "excluded_base_build_ids": excluded,
        "replacement_plan_id": plan["plan_id"],
        "replacement_plan_sha256": plan_sha,
        "replacement_contract_sha256": contract_sha,
        "replacement_finalization_sha256": finalization_sha,
        "inventory_sha256": inventory_sha,
    }


def _validate_paired_short_overlay(
        replacement_plan, replacement_contract, replacement_finalization,
        base_plan, base_plan_sha, policy_sha, publish_root):
    supplied = [replacement_plan, replacement_contract, replacement_finalization]
    if not any(supplied):
        return None
    _need(all(supplied),
          "paired short replacement overlay inputs must be all-or-none")
    replacement_plan = Path(replacement_plan)
    replacement_contract = Path(replacement_contract)
    replacement_finalization = Path(replacement_finalization)
    plan = _load(replacement_plan)
    contract = _load(replacement_contract)
    finalization = _load(replacement_finalization)
    plan_sha = sha256_file(replacement_plan)
    contract_sha = sha256_file(replacement_contract)
    finalization_sha = sha256_file(replacement_finalization)

    _need(plan.get("schema_version") == 1,
          "paired short replacement plan schema mismatch")
    _need(contract.get("schema_version") == 1
          and contract.get("status") == "pass",
          "paired short replacement contract mismatch")
    _need(contract.get("replacement_effective_catalog_role") ==
          "replacement_builds_only_for_paired_short",
          "paired short replacement catalog role mismatch")
    base_binding = contract.get("base_binding") or {}
    _need(base_binding.get("base_plan_id") == base_plan.get("plan_id"),
          "paired short base plan ID mismatch")
    _need(base_binding.get("base_plan_sha256") == base_plan_sha,
          "paired short base plan SHA mismatch")
    _need(base_binding.get("base_policy_sha256") == policy_sha,
          "paired short base policy SHA mismatch")
    _need(base_binding.get("effective_catalog_role") ==
          "audit_only_superseded_for_paired_short",
          "paired short base catalog role mismatch")
    transform = contract.get("transform") or {}
    _need(transform.get("output_view") == "main_train"
          and int(transform.get("output_n_patches", -1)) == 1
          and int(transform.get("output_window_seconds", -1)) == 1
          and int(transform.get("output_padding_samples", -1)) == 0,
          "paired short transform contract mismatch")

    _need(plan.get("base_plan_id") == base_plan.get("plan_id")
          and plan.get("base_plan_sha256") == base_plan_sha,
          "paired short replacement plan base binding mismatch")
    _need(plan.get("base_policy_sha256") == policy_sha,
          "paired short replacement plan policy mismatch")
    _need(plan.get("replacement_contract_sha256") == contract_sha,
          "paired short replacement plan contract SHA mismatch")
    inventory_sha = plan.get("inventory_sha256")
    _need(bool(SHA64.fullmatch(str(inventory_sha))),
          "paired short replacement inventory SHA invalid")

    excluded = base_binding.get("base_paired_build_ids")
    _need(isinstance(excluded, list) and excluded == sorted(set(excluded))
          and len(excluded) == 30,
          "paired short base build IDs mismatch")
    summary = plan.get("summary") or {}
    _need(summary.get("base_paired_builds_excluded_from_effective_catalog")
          == excluded, "paired short plan exclusion mismatch")
    base_jobs = [job for job in base_plan.get("jobs", [])
                 if job.get("logical_source") in PAIRED_SHORT_SOURCES]
    _need({job["build_id"] for job in base_jobs} == set(excluded),
          "paired short base plan jobs mismatch")

    jobs = plan.get("jobs")
    _need(isinstance(jobs, list) and len(jobs) == len(excluded),
          "paired short replacement jobs mismatch")
    build_ids = [job.get("build_id") for job in jobs]
    _need(len(set(build_ids)) == len(build_ids)
          and not (set(build_ids) & set(excluded)),
          "paired short replacement build IDs invalid")
    _need({job.get("source_build_id") for job in jobs} == set(excluded),
          "paired short source build set mismatch")
    for job in jobs:
        _need(job.get("logical_source") in PAIRED_SHORT_SOURCES,
              "paired short logical source mismatch")
        _need(job.get("source_kind") == PAIRED_SHORT_SOURCE_KIND,
              "paired short source kind mismatch")
        _need(bool(SHA64.fullmatch(str(job.get("job_manifest_sha256")))),
              "paired short job manifest SHA invalid")

    _need(finalization.get("status") == "pass"
          and finalization.get("is_release_ready") is False,
          "paired short finalization status mismatch")
    _need((finalization.get("input_hashes") or {}) == {
        "replacement_plan_sha256": plan_sha,
        "replacement_contract_sha256": contract_sha,
        "inventory_sha256": inventory_sha,
    }, "paired short finalization input hashes mismatch")
    report_ids = [row.get("build_id") for row in
                  finalization.get("replacement_builds", [])]
    _need(set(report_ids) == set(build_ids)
          and len(report_ids) == len(build_ids),
          "paired short finalization build set mismatch")
    _need(int((finalization.get("summary") or {}).get("builds", -1))
          == len(jobs), "paired short finalization build count mismatch")

    base_ready_hashes = {}
    for build_id in excluded:
        ready_path = Path(publish_root) / "corpus_builds" / build_id / "READY.json"
        _need(ready_path.is_file(),
              f"missing paired short superseded READY: {build_id}")
        base_ready_hashes[build_id] = sha256_file(ready_path)
    _need(finalization.get("base_ready_hashes_audit_only") == base_ready_hashes,
          "paired short finalization base READY hashes mismatch")
    return {
        "jobs": jobs,
        "excluded_base_build_ids": excluded,
        "replacement_plan_id": plan["plan_id"],
        "replacement_plan_sha256": plan_sha,
        "replacement_contract_sha256": contract_sha,
        "replacement_finalization_sha256": finalization_sha,
        "inventory_sha256": inventory_sha,
    }


def _validate_remediation_overlay(
        label, replacement_plan, replacement_contract,
        replacement_finalization, base_plan, base_plan_sha, policy_sha,
        publish_root, expected_sources, expected_catalog_role,
        expected_base_role, base_artifact_name="READY.json",
        base_hash_field="base_ready_hashes_audit_only",
        require_same_source_kind=True):
    supplied = [
        replacement_plan, replacement_contract, replacement_finalization]
    if not any(supplied):
        return None
    _need(all(supplied),
          f"{label} replacement overlay inputs must be all-or-none")
    replacement_plan = Path(replacement_plan)
    replacement_contract = Path(replacement_contract)
    replacement_finalization = Path(replacement_finalization)
    plan = _load(replacement_plan)
    contract = _load(replacement_contract)
    finalization = _load(replacement_finalization)
    plan_sha = sha256_file(replacement_plan)
    contract_sha = sha256_file(replacement_contract)
    finalization_sha = sha256_file(replacement_finalization)

    _need(plan.get("schema_version") == 1,
          f"{label} replacement plan schema mismatch")
    _need(contract.get("schema_version") == 1
          and contract.get("status") == "pass"
          and contract.get("is_release_ready") is False,
          f"{label} replacement contract mismatch")
    _need(contract.get("replacement_effective_catalog_role")
          == expected_catalog_role,
          f"{label} replacement catalog role mismatch")
    base_binding = contract.get("base_binding") or {}
    _need(base_binding.get("base_plan_id") == base_plan.get("plan_id")
          and base_binding.get("base_plan_sha256") == base_plan_sha
          and base_binding.get("base_policy_sha256") == policy_sha,
          f"{label} base binding mismatch")
    _need(base_binding.get("effective_catalog_role") == expected_base_role,
          f"{label} base catalog role mismatch")
    excluded = base_binding.get("base_build_ids")
    _need(isinstance(excluded, list) and excluded
          and excluded == sorted(set(excluded)),
          f"{label} excluded base build IDs mismatch")
    base_jobs = [
        job for job in base_plan.get("jobs", [])
        if job.get("logical_source") in expected_sources]
    base_by_id = {job["build_id"]: job for job in base_jobs}
    _need(set(base_by_id) == set(excluded)
          and {job["logical_source"] for job in base_jobs}
          == set(expected_sources),
          f"{label} base source coverage mismatch")

    _need(plan.get("base_plan_id") == base_plan.get("plan_id")
          and plan.get("base_plan_sha256") == base_plan_sha
          and plan.get("base_policy_sha256") == policy_sha,
          f"{label} replacement plan base binding mismatch")
    _need(plan.get("replacement_contract_sha256") == contract_sha,
          f"{label} replacement plan contract SHA mismatch")
    inventory_sha = plan.get("inventory_sha256")
    _need(bool(SHA64.fullmatch(str(inventory_sha))),
          f"{label} replacement inventory SHA invalid")
    summary = plan.get("summary") or {}
    _need(summary.get("base_builds_excluded_from_effective_catalog")
          == excluded, f"{label} plan exclusion mismatch")
    jobs = plan.get("jobs")
    _need(isinstance(jobs, list) and len(jobs) == len(excluded)
          and int(summary.get("jobs", -1)) == len(jobs),
          f"{label} replacement job count mismatch")
    build_ids = [job.get("build_id") for job in jobs]
    _need(len(set(build_ids)) == len(build_ids)
          and not (set(build_ids) & set(excluded)),
          f"{label} replacement build IDs invalid")
    _need({job.get("source_build_id") for job in jobs} == set(excluded)
          and {job.get("logical_source") for job in jobs}
          == set(expected_sources),
          f"{label} replacement source coverage mismatch")
    for job in jobs:
        base_job = base_by_id.get(job.get("source_build_id"))
        _need(base_job is not None
              and job.get("logical_source")
              == base_job.get("logical_source")
              and (not require_same_source_kind
                   or job.get("source_kind") == base_job.get("source_kind")),
              f"{label} replacement source identity mismatch")
        _need(bool(SHA64.fullmatch(str(job.get("job_manifest_sha256")))),
              f"{label} replacement job manifest SHA invalid")

    _need(finalization.get("status") == "pass"
          and finalization.get("is_release_ready") is False
          and finalization.get("effective_catalog_approved") is True,
          f"{label} finalization is not approved")
    finalization_inputs = finalization.get("input_hashes") or {}
    required_inputs = {
        "replacement_plan_sha256": plan_sha,
        "replacement_contract_sha256": contract_sha,
        "inventory_sha256": inventory_sha,
    }
    _need(all(finalization_inputs.get(key) == value
              for key, value in required_inputs.items())
          and all(bool(SHA64.fullmatch(str(value)))
                  for value in finalization_inputs.values()),
          f"{label} finalization input hashes mismatch")
    report_ids = [row.get("build_id") for row in
                  finalization.get("replacement_builds", [])]
    _need(set(report_ids) == set(build_ids)
          and len(report_ids) == len(build_ids)
          and int((finalization.get("summary") or {}).get("builds", -1))
          == len(jobs),
          f"{label} finalization build coverage mismatch")
    base_artifact_hashes = {}
    for build_id in excluded:
        artifact_path = (Path(publish_root) / "corpus_builds"
                         / build_id / base_artifact_name)
        _need(artifact_path.is_file(),
              f"missing {label} superseded {base_artifact_name}: {build_id}")
        base_artifact_hashes[build_id] = sha256_file(artifact_path)
    _need(finalization.get(base_hash_field) == base_artifact_hashes,
          f"{label} finalization base artifact hashes mismatch")
    return {
        "label": label,
        "jobs": jobs,
        "excluded_base_build_ids": excluded,
        "replacement_plan_id": plan["plan_id"],
        "replacement_plan_sha256": plan_sha,
        "replacement_contract_sha256": contract_sha,
        "replacement_finalization_sha256": finalization_sha,
        "inventory_sha256": inventory_sha,
    }


def _validate_current_catalog_acceptance(
        acceptance_path, plan_sha, policy_sha, accepted_terminal_sha):
    if acceptance_path is None:
        return None
    path = Path(acceptance_path)
    document = _load(path)
    _need(document.get("schema_version") == 1
          and document.get("decision_version")
          == "wds-549-current-catalog-training-acceptance-20260821.1"
          and document.get("status") == "pass"
          and document.get("is_release_ready") is False
          and document.get("decision")
          == "train_on_current_effective_catalog_without_remaining_quarantine_repairs",
          "current catalog acceptance identity mismatch")
    _need(document.get("plan_sha256") == plan_sha
          and document.get("policy_sha256") == policy_sha
          and document.get("accepted_terminals_sha256")
          == accepted_terminal_sha,
          "current catalog acceptance input mismatch")
    selection = document.get("training_selection") or {}
    _need(selection.get("included_views") == [
              "main_train", "aux_quality_flagged", "aux_low_channel"]
          and selection.get("excluded_views") == [
              "quarantine_quality_severe",
              "quarantine_nonfinite_or_decode", "excluded_leakage"]
          and selection.get("preserve_frozen_source_and_view_weights") is True
          and selection.get("promote_severe_windows") is False,
          "current catalog acceptance view mismatch")
    accounting = document.get("on004395_accounting") or {}
    _need(int(accounting.get("eligible_windows", -1)) == 4_864_633
          and int(accounting.get("severe_windows", -1)) == 1_716_381
          and abs(float(accounting.get("eligible_valid_hours", -1))
                  - 6825.936944444445) < 1e-9,
          "current catalog ON004395 accounting mismatch")
    actions = document.get("deferred_remediation_actions") or []
    _need(len(actions) == 5
          and all(row.get("status") == "pass"
                  and row.get("resolution")
                  == "accept_current_catalog_training_views_keep_severe_excluded"
                  for row in actions),
          "current catalog deferred actions mismatch")
    return {"path": path, "sha256": sha256_file(path)}


def _validate_accepted_terminals(
        accepted_terminals, plan, plan_sha, policy_sha, leakage_sha,
        publish_root):
    if accepted_terminals is None:
        return {
            "by_build": {}, "by_source": {}, "rows": [], "sha256": None}

    path = Path(accepted_terminals)
    document = _load(path)
    _need(document.get("schema_version") == 1,
          "accepted terminal schema mismatch")
    _need(document.get("selection_version")
          in ACCEPTED_TERMINAL_SELECTION_VERSIONS,
          "accepted terminal selection version mismatch")
    _need(document.get("plan_id") == plan.get("plan_id"),
          "accepted terminal plan ID mismatch")
    _need(document.get("plan_sha256") == plan_sha,
          "accepted terminal plan SHA mismatch")
    _need(document.get("policy_sha256") == policy_sha,
          "accepted terminal policy SHA mismatch")
    _need(document.get("leakage_allowlist_sha256") == leakage_sha,
          "accepted terminal leakage SHA mismatch")
    _need(document.get("is_release_ready") is False,
          "accepted terminal release flag mismatch")

    rows = document.get("terminals")
    _need(isinstance(rows, list) and rows,
          "accepted terminal rows missing")
    build_ids = [row.get("build_id") for row in rows]
    _need(build_ids == sorted(set(build_ids)),
          "accepted terminal build IDs must be sorted unique")
    jobs = {job["build_id"]: job for job in plan.get("jobs", [])}
    by_build = {}
    by_source = defaultdict(list)
    for row in rows:
        required = {
            "build_id", "logical_source", "source_kind", "status", "reason",
            "terminal_version", "terminal_sha256", "job_manifest_sha256"}
        _need(set(row) == required,
              "accepted terminal row fields mismatch")
        _need(bool(SHA64.fullmatch(str(row["terminal_sha256"]))),
              "accepted terminal SHA invalid")
        job = jobs.get(row["build_id"])
        _need(job is not None, "accepted terminal build is not planned")
        for field in (
                "build_id", "logical_source", "source_kind",
                "job_manifest_sha256"):
            _need(row[field] == job[field],
                  f"accepted terminal job field mismatch: {field}")

        target = Path(publish_root) / "corpus_builds" / row["build_id"]
        terminal_path = target / "TERMINAL.json"
        _need(terminal_path.is_file(),
              f"accepted TERMINAL missing: {row['build_id']}")
        _need(not (target / "READY.json").exists(),
              f"accepted terminal unexpectedly READY: {row['build_id']}")
        _need(sha256_file(terminal_path) == row["terminal_sha256"],
              f"accepted TERMINAL SHA mismatch: {row['build_id']}")
        terminal = _load(terminal_path)
        expected_identity = {
            "schema_version": 1,
            "build_id": row["build_id"],
            "logical_source": row["logical_source"],
            "source_kind": row["source_kind"],
            "plan_id": plan["plan_id"],
            "policy_sha256": policy_sha,
            "job_manifest_sha256": row["job_manifest_sha256"],
            "status": row["status"],
            "reason": row["reason"],
            "terminal_version": row["terminal_version"],
        }
        for field, value in expected_identity.items():
            _need(terminal.get(field) == value,
                  f"accepted TERMINAL identity mismatch {field}: "
                  f"{row['build_id']}")
        if "leakage_allowlist_sha256" in terminal:
            _need(terminal["leakage_allowlist_sha256"] == leakage_sha,
                  f"accepted TERMINAL leakage SHA mismatch: {row['build_id']}")
        by_build[row["build_id"]] = dict(row)
        by_source[row["logical_source"]].append(dict(row))

    for source, terminal_rows in by_source.items():
        planned = {
            job["build_id"] for job in plan.get("jobs", [])
            if job["logical_source"] == source}
        accepted = {row["build_id"] for row in terminal_rows}
        _need(planned == accepted,
              f"accepted terminal source is only partially covered: {source}")

    return {
        "by_build": by_build,
        "by_source": dict(by_source),
        "rows": [dict(row) for row in rows],
        "sha256": sha256_file(path),
    }


def build_global_catalogs(
        plan, policy, source_registry, leakage_allowlist, publish_root,
        output_root, expected_counts=None, tuh_replacement_plan=None,
        tuh_replacement_contract=None, tuh_replacement_finalization=None,
        paired_short_replacement_plan=None,
        paired_short_replacement_contract=None,
        paired_short_replacement_finalization=None,
        natural_short_replacement_plan=None,
        natural_short_replacement_contract=None,
        natural_short_replacement_finalization=None,
        source_repair_replacement_plan=None,
        source_repair_replacement_contract=None,
        source_repair_replacement_finalization=None,
        on004395_replacement_plan=None,
        on004395_replacement_contract=None,
        on004395_replacement_finalization=None,
        faced_nemar_replacement_plan=None,
        faced_nemar_replacement_contract=None,
        faced_nemar_replacement_finalization=None,
        current_catalog_acceptance=None,
        accepted_terminals=None):
    expected = _validate_counts(expected_counts)
    plan_path = Path(plan)
    policy_path = Path(policy)
    registry_path = Path(source_registry)
    leakage_path = Path(leakage_allowlist)
    publish_root = Path(publish_root)
    output_root = Path(output_root)
    if output_root.exists():
        raise FileExistsError(f"output_root already exists: {output_root}")
    tmp_root = output_root.with_name(output_root.name + f".tmp.{os.getpid()}")
    if tmp_root.exists():
        shutil.rmtree(tmp_root)

    try:
        plan_doc = _load(plan_path)
        policy_doc = _load(policy_path)
        registry_doc = _load(registry_path)
        _ = _load(leakage_path)

        psha = sha256_file(policy_path)
        lsha = sha256_file(leakage_path)
        rsha = sha256_file(registry_path)
        plan_sha = sha256_file(plan_path)
        accepted = _validate_accepted_terminals(
            accepted_terminals, plan_doc, plan_sha, psha, lsha,
            publish_root)
        current_acceptance = _validate_current_catalog_acceptance(
            current_catalog_acceptance, plan_sha, psha, accepted["sha256"])
        if current_acceptance is not None:
            _need(not any((
                natural_short_replacement_plan,
                natural_short_replacement_contract,
                natural_short_replacement_finalization,
                source_repair_replacement_plan,
                source_repair_replacement_contract,
                source_repair_replacement_finalization,
                on004395_replacement_plan,
                on004395_replacement_contract,
                on004395_replacement_finalization)),
                "current catalog acceptance conflicts with deferred replacement inputs")
        tuh_overlay = _validate_tuh_overlay(
            tuh_replacement_plan, tuh_replacement_contract,
            tuh_replacement_finalization, plan_doc, plan_sha, psha,
            publish_root)
        paired_short_overlay = _validate_paired_short_overlay(
            paired_short_replacement_plan, paired_short_replacement_contract,
            paired_short_replacement_finalization, plan_doc, plan_sha, psha,
            publish_root)
        natural_short_overlay = _validate_remediation_overlay(
            "natural_short", natural_short_replacement_plan,
            natural_short_replacement_contract,
            natural_short_replacement_finalization, plan_doc, plan_sha, psha,
            publish_root, NATURAL_SHORT_SOURCES,
            "replacement_builds_only_for_natural_short",
            "audit_only_superseded_for_natural_short")
        source_repair_overlay = _validate_remediation_overlay(
            "source_repair", source_repair_replacement_plan,
            source_repair_replacement_contract,
            source_repair_replacement_finalization, plan_doc, plan_sha, psha,
            publish_root, SOURCE_REPAIR_SOURCES,
            "replacement_builds_only_for_source_repair",
            "audit_only_superseded_for_source_repair")
        on004395_overlay = _validate_remediation_overlay(
            "on004395_channel_repair", on004395_replacement_plan,
            on004395_replacement_contract,
            on004395_replacement_finalization, plan_doc, plan_sha, psha,
            publish_root, ON004395_CHANNEL_REPAIR_SOURCES,
            "replacement_builds_only_for_on004395_channel_repair",
            "audit_only_superseded_for_on004395_channel_repair")
        faced_nemar_overlay = _validate_remediation_overlay(
            "faced_nemar", faced_nemar_replacement_plan,
            faced_nemar_replacement_contract,
            faced_nemar_replacement_finalization, plan_doc, plan_sha, psha,
            publish_root, FACED_NEMAR_SOURCES,
            "replacement_builds_only_for_faced_nemar",
            "audit_only_superseded_for_faced_nemar",
            base_artifact_name="TERMINAL.json",
            base_hash_field="base_terminal_hashes_audit_only",
            require_same_source_kind=False)
        overlays = [overlay for overlay in (
            tuh_overlay, paired_short_overlay, natural_short_overlay,
            source_repair_overlay, on004395_overlay, faced_nemar_overlay)
                    if overlay is not None]
        input_sha256 = {
            "plan": plan_sha,
            "policy": psha,
            "source_registry": rsha,
            "leakage_allowlist": lsha,
        }
        if tuh_overlay is not None:
            input_sha256.update({
                "tuh_replacement_plan": tuh_overlay["replacement_plan_sha256"],
                "tuh_replacement_contract": tuh_overlay[
                    "replacement_contract_sha256"],
                "tuh_replacement_finalization": tuh_overlay[
                    "replacement_finalization_sha256"],
            })
        if paired_short_overlay is not None:
            input_sha256.update({
                "paired_short_replacement_plan": paired_short_overlay[
                    "replacement_plan_sha256"],
                "paired_short_replacement_contract": paired_short_overlay[
                    "replacement_contract_sha256"],
                "paired_short_replacement_finalization": paired_short_overlay[
                    "replacement_finalization_sha256"],
            })
        for overlay in (
                natural_short_overlay, source_repair_overlay,
                on004395_overlay, faced_nemar_overlay):
            if overlay is not None:
                label = overlay["label"]
                input_sha256.update({
                    f"{label}_replacement_plan": overlay[
                        "replacement_plan_sha256"],
                    f"{label}_replacement_contract": overlay[
                        "replacement_contract_sha256"],
                    f"{label}_replacement_finalization": overlay[
                        "replacement_finalization_sha256"],
                })
        if accepted["sha256"] is not None:
            input_sha256["accepted_terminals"] = accepted["sha256"]
        if current_acceptance is not None:
            input_sha256["current_catalog_acceptance"] = (
                current_acceptance["sha256"])

        _need(plan_doc.get("policy_sha256") == psha, "plan policy SHA mismatch")
        _need(plan_doc.get("source_registry_sha256") == rsha, "plan source_registry SHA mismatch")
        _need(plan_doc.get("leakage_allowlist_sha256") == lsha, "plan leakage_allowlist SHA mismatch")
        _need(registry_doc.get("policy_sha256") == psha, "registry policy SHA mismatch")
        _need(registry_doc.get("leakage_allowlist_sha256") == lsha, "registry leakage_allowlist SHA mismatch")

        _need(plan_doc.get("summary", {}).get("sources") == expected["sources"], "plan sources mismatch")
        _need(plan_doc.get("summary", {}).get("planned_sources") == expected["planned_sources"], "plan planned_sources mismatch")
        _need(plan_doc.get("summary", {}).get("archive_only_sources") == expected["archive_only_sources"], "plan archive_only_sources mismatch")
        _need(plan_doc.get("summary", {}).get("jobs") == expected["jobs"], "plan jobs mismatch")
        _need(int(policy_doc.get("scope", {}).get("target_config_sources", -1)) == expected["sources"], "policy target_config_sources mismatch")

        reg_by_source, buildable, archive = _validate_registry(plan_doc, registry_doc, expected)

        _need(not (publish_root / "RELEASE_READY.json").exists(), "RELEASE_READY.json must be absent at publish root")

        inv = audit_build_states(plan_path, policy_path, publish_root)
        s = inv.get("summary", {})
        _need(s.get("builds") == expected["jobs"], "build-state builds mismatch")
        _need(s.get("terminal_builds") == expected["jobs"], "build-state terminal_builds mismatch")
        inventory_rows = inv.get("builds")
        _need(isinstance(inventory_rows, list)
              and len(inventory_rows) == expected["jobs"],
              "build-state row count mismatch")
        inventory_by_build = {row.get("build_id"): row for row in inventory_rows}
        planned_ids = {job["build_id"] for job in plan_doc.get("jobs", [])}
        _need(len(inventory_by_build) == expected["jobs"]
              and set(inventory_by_build) == planned_ids,
              "build-state build set mismatch")
        accepted_by_build = accepted["by_build"]
        for build_id, row in inventory_by_build.items():
            terminal = accepted_by_build.get(build_id)
            if terminal is None:
                _need(row.get("state") == "ready",
                      f"unaccepted non-ready build: {build_id}")
                _need(row.get("status") in READY_STATUSES,
                      f"ready build status mismatch: {build_id}")
            else:
                _need(row.get("state") == "terminal",
                      f"accepted build is not terminal: {build_id}")
                _need(row.get("status") == terminal["status"],
                      f"accepted build status mismatch: {build_id}")
        expected_states = {"ready": expected["jobs"] - len(accepted_by_build)}
        if accepted_by_build:
            expected_states["terminal"] = len(accepted_by_build)
        _need(s.get("states") == expected_states,
              "build-state states mismatch")
        statuses = s.get("statuses", {})
        _need(isinstance(statuses, dict)
              and sum(int(value) for value in statuses.values()) == expected["jobs"],
              "build-state statuses mismatch")
        _need(s.get("logical_sources") == expected["planned_sources"],
              "build-state logical_sources mismatch")
        _need(s.get("logical_source_states") ==
              {"terminal": expected["planned_sources"]},
              "build-state logical_source_states mismatch")
        _need(inv.get("release_ready_present") is False,
              "build-state release_ready_present must be false")

        policy_views = set((policy_doc.get("views") or {}).keys())
        _need(policy_views, "policy views missing")

        global_paths = set()
        build_rows = []
        per_view = {v: [] for v in sorted(policy_views)}
        excluded_lists = [set(overlay["excluded_base_build_ids"])
                          for overlay in overlays]
        excluded_base = set().union(*excluded_lists) if excluded_lists else set()
        _need(sum(len(values) for values in excluded_lists) == len(excluded_base),
              "replacement overlays exclude overlapping base builds")
        effective_accepted_rows = [
            row for row in accepted["rows"]
            if row["build_id"] not in excluded_base]
        effective_accepted_by_source = defaultdict(list)
        for row in effective_accepted_rows:
            effective_accepted_by_source[row["logical_source"]].append(row)
        effective_specs = []
        for job in plan_doc.get("jobs", []):
            if (job["build_id"] in excluded_base
                    or job["build_id"] in accepted["by_build"]):
                continue
            identity = {
                "schema_version": 1,
                "build_id": job["build_id"],
                "plan_id": plan_doc["plan_id"],
                "policy_sha256": psha,
                "leakage_allowlist_sha256": lsha,
                "job_manifest_sha256": job["job_manifest_sha256"],
            }
            effective_specs.append(
                (job, identity, "corpus_builds", "base"))
        replacement_ids = set()
        for overlay in overlays:
            for job in overlay["jobs"]:
                _need(job["build_id"] not in replacement_ids,
                      "duplicate build ID across replacement overlays")
                replacement_ids.add(job["build_id"])
                identity = {
                    "schema_version": 1,
                    "build_id": job["build_id"],
                    "replacement_plan_id": overlay["replacement_plan_id"],
                    "replacement_plan_sha256": overlay[
                        "replacement_plan_sha256"],
                    "replacement_contract_sha256": overlay[
                        "replacement_contract_sha256"],
                    "inventory_sha256": overlay["inventory_sha256"],
                    "job_manifest_sha256": job["job_manifest_sha256"],
                }
                effective_specs.append(
                    (job, identity, "replacement_builds", "replacement"))
        effective_specs.sort(key=lambda item: item[0]["build_id"])
        effective_selection = {
            "base_plan_jobs": len(plan_doc.get("jobs", [])),
            "accepted_terminal_jobs": len(effective_accepted_rows),
            "superseded_base_jobs": len(excluded_base),
            "replacement_jobs": sum(len(overlay["jobs"]) for overlay in overlays),
            "effective_jobs": len(effective_specs),
        }
        _need(
            effective_selection["effective_jobs"] ==
            expected["jobs"] - len(excluded_base) -
            effective_selection["accepted_terminal_jobs"] +
            effective_selection["replacement_jobs"],
            "effective build count mismatch")

        for job, identity, build_root, catalog_role in effective_specs:
            row = _validate_build(
                job, identity, publish_root, policy_views, global_paths,
                build_root=build_root, catalog_role=catalog_role)
            build_rows.append(row)
            for tar_row in row["tar_rows"]:
                per_view[tar_row["view"]].append(tar_row)

        tmp_root.mkdir(parents=True, exist_ok=False)
        out_meta = {}

        index_rows = [{
            "build_id": r["build_id"], "build_root": r["build_root"],
            "catalog_role": r["catalog_role"], "logical_source": r["logical_source"],
            "source_kind": r["source_kind"],
            "status": r["status"], "windows": r["windows"], "tars": r["tars"], "size_bytes": r["size_bytes"],
            "views": r["views"], "ready_sha256": r["ready_sha256"], "build_manifest_sha256": r["build_manifest_sha256"],
            "catalog_sha256": r["catalog_sha256"], "ready_relpath": r["ready_relpath"],
            "manifest_relpath": r["manifest_relpath"], "catalog_relpath": r["catalog_relpath"],
        } for r in build_rows]
        out_meta["BUILD_CATALOG_INDEX.jsonl"] = atomic_write_jsonl(tmp_root / "BUILD_CATALOG_INDEX.jsonl", sorted(index_rows, key=lambda x: x["build_id"]))
        terminal_rows = [{
            **row,
            "catalog_role": (
                "audit_only_superseded_terminal"
                if row["build_id"] in excluded_base else "terminal"),
            "terminal_relpath": (
                f"corpus_builds/{row['build_id']}/TERMINAL.json"),
        } for row in accepted["rows"]]
        out_meta["ACCEPTED_TERMINALS.jsonl"] = atomic_write_jsonl(
            tmp_root / "ACCEPTED_TERMINALS.jsonl", terminal_rows)
        build_count = Counter(r["logical_source"] for r in build_rows)
        size_count = defaultdict(int)
        source_kinds = defaultdict(set)
        source_roles = defaultdict(set)
        for row in build_rows:
            source = row["logical_source"]
            size_count[source] += row["size_bytes"]
            source_kinds[source].add(row["source_kind"])
            source_roles[source].add(row["catalog_role"])
        source_rows = []
        for source in sorted(set(buildable) | set(archive)):
            is_archive_only = source in archive
            terminal_for_source = effective_accepted_by_source.get(source, [])
            terminal_build_ids = sorted(
                row["build_id"] for row in terminal_for_source)
            terminal_reasons = sorted(
                {row["reason"] for row in terminal_for_source})
            if is_archive_only:
                effective_kind = reg_by_source[source]["source_kind"]
                effective_role = "archive_only"
                effective_status = reg_by_source[source]["status"]
            elif terminal_for_source:
                _need(build_count[source] == 0,
                      f"terminal source unexpectedly has catalog builds: {source}")
                _need(len({row["status"] for row in terminal_for_source}) == 1,
                      f"terminal source status is ambiguous: {source}")
                effective_kind = reg_by_source[source]["source_kind"]
                effective_role = "terminal"
                effective_status = terminal_for_source[0]["status"]
            else:
                _need(build_count[source] > 0,
                      f"effective source has no builds: {source}")
                _need(len(source_kinds[source]) == 1,
                      f"effective source_kind is ambiguous: {source}")
                _need(len(source_roles[source]) == 1,
                      f"effective catalog role is ambiguous: {source}")
                effective_kind = next(iter(source_kinds[source]))
                effective_role = next(iter(source_roles[source]))
                effective_status = reg_by_source[source]["status"]
            source_rows.append({
                "logical_source": source,
                "source_kind": effective_kind,
                "registry_source_kind": reg_by_source[source]["source_kind"],
                "registry_status": reg_by_source[source]["status"],
                "effective_catalog_role": effective_role,
                "status": effective_status,
                "is_archive_only": is_archive_only,
                "planned_builds": int(build_count.get(source, 0)),
                "catalog_size_bytes": int(size_count.get(source, 0)),
                "terminal_build_ids": terminal_build_ids,
                "terminal_reasons": terminal_reasons,
            })
        _need(len(source_rows) == expected["sources"],
              "SOURCE_TERMINAL_AUDIT row count mismatch")
        out_meta["SOURCE_TERMINAL_AUDIT.jsonl"] = atomic_write_jsonl(tmp_root / "SOURCE_TERMINAL_AUDIT.jsonl", source_rows)

        view_summary = {}
        for view in sorted(policy_views):
            rows = sorted(per_view[view], key=lambda x: x["release_relative_tar"])
            rel = f"views/{view}.jsonl"
            m = atomic_write_jsonl(tmp_root / rel, rows)
            out_meta[rel] = m
            view_summary[view] = {
                "path": rel, "sha256": m["sha256"], "rows": m["rows"], "bytes": m["bytes"],
                "windows": sum(int(x["windows"]) for x in rows), "tars": len(rows),
                "size_bytes": sum(int(x["size_bytes"]) for x in rows),
            }

        inv_path = tmp_root / "BUILD_STATE_INVENTORY.json"
        inv_sha = atomic_write_json(inv_path, inv)
        out_meta["BUILD_STATE_INVENTORY.json"] = {"sha256": inv_sha, "bytes": inv_path.stat().st_size, "rows": 1}

        tw = sum(r["windows"] for r in build_rows)
        tt = sum(r["tars"] for r in build_rows)
        _need(tw == sum(v["windows"] for v in view_summary.values()), "global windows mismatch")
        _need(tt == sum(v["tars"] for v in view_summary.values()), "global tars mismatch")

        complete = {
            "schema_version": 1,
            "report_version": "wds-549-phase5-global-catalogs-v4",
            "status": "pass",
            "is_release_ready": False,
            "plan_id": plan_doc["plan_id"],
            "input_sha256": input_sha256,
            "expected_counts": expected,
            "effective_selection": effective_selection,
            "plan_summary": {"sources": plan_doc["summary"]["sources"], "planned_sources": plan_doc["summary"]["planned_sources"], "archive_only_sources": plan_doc["summary"]["archive_only_sources"], "jobs": plan_doc["summary"]["jobs"]},
            "registry_summary": registry_doc["summary"],
            "build_state_summary": {"builds": s["builds"], "terminal_builds": s["terminal_builds"], "states": s["states"], "statuses": s["statuses"], "logical_sources": s["logical_sources"], "logical_source_states": s["logical_source_states"], "release_ready_present": False},
            "outputs": out_meta,
            "views": view_summary,
            "totals": {"windows": tw, "tars": tt, "size_bytes": sum(r["size_bytes"] for r in build_rows)},
            "claims": {
                "payload_verification": False,
                "full_tar_hash_verification": False,
                "accepted_terminal_accounting": True,
            },
        }
        atomic_write_json(tmp_root / "CATALOGS_COMPLETE.json", complete)
        os.replace(tmp_root, output_root)
        return complete
    except Exception:
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        raise


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--plan", required=True)
    p.add_argument("--policy", required=True)
    p.add_argument("--source_registry", required=True)
    p.add_argument("--leakage_allowlist", required=True)
    p.add_argument("--publish_root", required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--tuh_replacement_plan")
    p.add_argument("--tuh_replacement_contract")
    p.add_argument("--tuh_replacement_finalization")
    p.add_argument("--paired_short_replacement_plan")
    p.add_argument("--paired_short_replacement_contract")
    p.add_argument("--paired_short_replacement_finalization")
    p.add_argument("--natural_short_replacement_plan")
    p.add_argument("--natural_short_replacement_contract")
    p.add_argument("--natural_short_replacement_finalization")
    p.add_argument("--source_repair_replacement_plan")
    p.add_argument("--source_repair_replacement_contract")
    p.add_argument("--source_repair_replacement_finalization")
    p.add_argument("--on004395_replacement_plan")
    p.add_argument("--on004395_replacement_contract")
    p.add_argument("--on004395_replacement_finalization")
    p.add_argument("--faced_nemar_replacement_plan")
    p.add_argument("--faced_nemar_replacement_contract")
    p.add_argument("--faced_nemar_replacement_finalization")
    p.add_argument("--current_catalog_acceptance")
    p.add_argument("--accepted_terminals")
    a = p.parse_args()

    rep = build_global_catalogs(
        plan=a.plan,
        policy=a.policy,
        source_registry=a.source_registry,
        leakage_allowlist=a.leakage_allowlist,
        publish_root=a.publish_root,
        output_root=a.output_root,
        expected_counts=None,
        tuh_replacement_plan=a.tuh_replacement_plan,
        tuh_replacement_contract=a.tuh_replacement_contract,
        tuh_replacement_finalization=a.tuh_replacement_finalization,
        paired_short_replacement_plan=a.paired_short_replacement_plan,
        paired_short_replacement_contract=a.paired_short_replacement_contract,
        paired_short_replacement_finalization=(
            a.paired_short_replacement_finalization),
        natural_short_replacement_plan=a.natural_short_replacement_plan,
        natural_short_replacement_contract=(
            a.natural_short_replacement_contract),
        natural_short_replacement_finalization=(
            a.natural_short_replacement_finalization),
        source_repair_replacement_plan=a.source_repair_replacement_plan,
        source_repair_replacement_contract=(
            a.source_repair_replacement_contract),
        source_repair_replacement_finalization=(
            a.source_repair_replacement_finalization),
        on004395_replacement_plan=a.on004395_replacement_plan,
        on004395_replacement_contract=a.on004395_replacement_contract,
        on004395_replacement_finalization=(
            a.on004395_replacement_finalization),
        faced_nemar_replacement_plan=a.faced_nemar_replacement_plan,
        faced_nemar_replacement_contract=a.faced_nemar_replacement_contract,
        faced_nemar_replacement_finalization=(
            a.faced_nemar_replacement_finalization),
        current_catalog_acceptance=a.current_catalog_acceptance,
        accepted_terminals=a.accepted_terminals,
    )
    print(json.dumps({"status": rep["status"], "is_release_ready": rep["is_release_ready"], "totals": rep["totals"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
