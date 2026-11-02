#!/usr/bin/env python3
"""Inventory terminal state for every locked WDS 549 build job."""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.wds_549_full_io import atomic_write_json, sha256_file  # noqa: E402


REPORT_VERSION = "wds-549-build-state-inventory-v1"


def common_identity(job, plan, policy_sha):
    return {
        "build_id": job["build_id"],
        "plan_id": plan["plan_id"],
        "policy_sha256": policy_sha,
        "job_manifest_sha256": job["job_manifest_sha256"],
    }


def validate_terminal_document(document, expected, path, kind):
    if any(document.get(key) != value for key, value in expected.items()):
        raise ValueError(f"{kind} identity mismatch: {path}")
    if kind == "READY":
        if (document.get("schema_version") != 1
                or not isinstance(document.get("processing_version"), str)
                or not document["processing_version"]
                or document.get("status") not in {
                    "ready_with_main", "ready_aux_only", "ready_archive_only"}
                or int(document.get("windows", -1)) < 0
                or int(document.get("tars", -1)) < 0):
            raise ValueError(f"READY contract mismatch: {path}")
    else:
        if (document.get("schema_version") != 1
                or not isinstance(document.get("terminal_version"), str)
                or not document["terminal_version"]
                or not str(document.get("status", "")).startswith("blocked_")):
            raise ValueError(f"TERMINAL contract mismatch: {path}")


def inspect_job(job, plan, policy_sha, publish_root):
    target = Path(publish_root) / "corpus_builds" / job["build_id"]
    ready_path = target / "READY.json"
    terminal_path = target / "TERMINAL.json"
    if ready_path.is_file() and terminal_path.is_file():
        raise ValueError(f"build has READY and TERMINAL: {job['build_id']}")
    expected = common_identity(job, plan, policy_sha)
    base = {
        "build_id": job["build_id"],
        "logical_source": job["logical_source"],
        "source_kind": job["source_kind"],
        "source_bytes": int(job["source_bytes"]),
    }
    if ready_path.is_file():
        document = json.loads(ready_path.read_text())
        validate_terminal_document(document, expected, ready_path, "READY")
        return {
            **base,
            "state": "ready",
            "status": document["status"],
            "processing_version": document["processing_version"],
            "windows": int(document["windows"]),
            "tars": int(document["tars"]),
            "terminal_sha256": sha256_file(ready_path),
        }
    if terminal_path.is_file():
        document = json.loads(terminal_path.read_text())
        validate_terminal_document(document, expected, terminal_path, "TERMINAL")
        return {
            **base,
            "state": "terminal",
            "status": document["status"],
            "processing_version": document.get("processing_version"),
            "windows": 0,
            "tars": 0,
            "terminal_version": document["terminal_version"],
            "terminal_sha256": sha256_file(terminal_path),
        }
    if target.is_dir():
        files = sorted(
            path.relative_to(target).as_posix()
            for path in target.rglob("*") if path.is_file())
        return {
            **base,
            "state": "incomplete",
            "status": "incomplete",
            "windows": 0,
            "tars": 0,
            "files": len(files),
            "has_build_lock": (target / "BUILD_LOCK.json").is_file(),
            "progress_files": sum(
                path.startswith("progress/") and path.endswith(".json")
                for path in files),
            "tar_files": sum(
                path.startswith("views/") and path.endswith(".tar")
                for path in files),
        }
    return {
        **base,
        "state": "missing",
        "status": "missing",
        "windows": 0,
        "tars": 0,
    }


def aggregate(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row[key]].append(row)
    output = {}
    for name, records in sorted(groups.items()):
        states = Counter(record["state"] for record in records)
        statuses = Counter(record["status"] for record in records)
        output[name] = {
            "builds": len(records),
            "states": dict(sorted(states.items())),
            "statuses": dict(sorted(statuses.items())),
            "terminal_builds": states["ready"] + states["terminal"],
            "windows": sum(record["windows"] for record in records),
            "tars": sum(record["tars"] for record in records),
            "source_bytes": sum(record["source_bytes"] for record in records),
        }
    return output


def audit(plan_path, policy_path, publish_root):
    plan_path = Path(plan_path)
    policy_path = Path(policy_path)
    plan = json.loads(plan_path.read_text())
    policy_sha = sha256_file(policy_path)
    if plan.get("policy_sha256") != policy_sha:
        raise ValueError("build-state inventory policy SHA mismatch")
    jobs = plan.get("jobs", [])
    if not jobs or len({job["build_id"] for job in jobs}) != len(jobs):
        raise ValueError("build-state inventory invalid plan jobs")
    rows = [
        inspect_job(job, plan, policy_sha, publish_root)
        for job in sorted(jobs, key=lambda item: item["build_id"])]
    states = Counter(row["state"] for row in rows)
    statuses = Counter(row["status"] for row in rows)
    source_summary = aggregate(rows, "logical_source")
    source_states = Counter()
    for summary in source_summary.values():
        if summary["terminal_builds"] == summary["builds"]:
            source_states["terminal"] += 1
        elif summary["terminal_builds"]:
            source_states["partial"] += 1
        else:
            source_states["unstarted"] += 1
    return {
        "schema_version": 1,
        "report_version": REPORT_VERSION,
        "plan_id": plan["plan_id"],
        "plan_sha256": sha256_file(plan_path),
        "policy_sha256": policy_sha,
        "release_ready_present": (
            Path(publish_root) / "RELEASE_READY.json").exists(),
        "summary": {
            "builds": len(rows),
            "states": dict(sorted(states.items())),
            "statuses": dict(sorted(statuses.items())),
            "terminal_builds": states["ready"] + states["terminal"],
            "windows": sum(row["windows"] for row in rows),
            "tars": sum(row["tars"] for row in rows),
            "logical_sources": len(source_summary),
            "logical_source_states": dict(sorted(source_states.items())),
        },
        "by_source_kind": aggregate(rows, "source_kind"),
        "by_logical_source": source_summary,
        "builds": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--publish_root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = audit(args.plan, args.policy, args.publish_root)
    output = Path(args.output)
    atomic_write_json(output, report)
    print(json.dumps({
        "report_version": report["report_version"],
        "release_ready_present": report["release_ready_present"],
        "summary": report["summary"],
        "by_source_kind": report["by_source_kind"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
