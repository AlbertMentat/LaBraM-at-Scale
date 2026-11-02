#!/usr/bin/env python3
"""Policy core for canonicalizing one decoded 200 Hz EEG recording or trial."""
import ast
import hashlib
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_channel_tools():
    path = REPO_ROOT / "eeg_data" / "ch_name_mappings.py"
    spec = importlib.util.spec_from_file_location("wds_549_channel_mappings", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.resolve_channels


def load_standard_1020():
    path = REPO_ROOT / "models" / "labram.py"
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if getattr(target, "id", None) == "standard_1020":
                    return ast.literal_eval(node.value)
    raise RuntimeError("standard_1020 not found")


RESOLVE_CHANNELS = load_channel_tools()
STANDARD_1020 = load_standard_1020()
STANDARD_SET = {name.upper() for name in STANDARD_1020}


def sha256_text(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def ordered_channel_signature(channel_names):
    return sha256_text("\x1f".join(channel_names))[:12]


def make_recording_uid(source_object_identity, recording_locator, raw_channel_order):
    value = json.dumps([
        source_object_identity, recording_locator, list(raw_channel_order)],
        separators=(",", ":"), ensure_ascii=True)
    return sha256_text(value)


def finite_runs(mask):
    runs = []
    start = None
    for index, value in enumerate(mask):
        if value and start is None:
            start = index
        elif not value and start is not None:
            runs.append((start, index))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def dynamic_window_seconds(valid_channels, policy):
    signal = policy["signal_contract"]
    return max(
        int(signal["min_window_seconds"]),
        min(int(signal["max_window_seconds"]),
            int(signal["token_budget"]) // int(valid_channels)))


def classify_quality(
        values, removed_bad_channels, clip_fraction, padding_fraction,
        reader_fallback):
    channel_std = values.std(axis=1)
    channel_mean = values.mean(axis=1)
    low_std_count = int(np.count_nonzero(channel_std < 0.005))
    severe_reasons = []
    if low_std_count:
        severe_reasons.append("channel_std_lt_0.005")
    if np.any(channel_std > 10.0):
        severe_reasons.append("channel_std_gt_10")
    if np.any(np.abs(channel_mean) > 10.0):
        severe_reasons.append("channel_mean_abs_gt_10")
    if clip_fraction > 0.05:
        severe_reasons.append("clip_fraction_gt_0.05")
    if padding_fraction >= 0.5:
        severe_reasons.append("padding_fraction_gte_0.5")

    mild_reasons = []
    if removed_bad_channels:
        mild_reasons.append("removed_bad_channel")
    if np.any(channel_std < 0.05):
        mild_reasons.append("channel_std_lt_0.05")
    if np.any(channel_std > 5.0):
        mild_reasons.append("channel_std_gt_5")
    if np.any(np.abs(channel_mean) > 5.0):
        mild_reasons.append("channel_mean_abs_gt_5")
    if clip_fraction > 0:
        mild_reasons.append("clip_fraction_gt_0")
    if padding_fraction > 0:
        mild_reasons.append("padding_fraction_gt_0")
    if reader_fallback:
        mild_reasons.append("reader_fallback")
    return sorted(set(mild_reasons)), sorted(set(severe_reasons)), {
        "channel_std_min": float(channel_std.min()),
        "channel_std_max": float(channel_std.max()),
        "channel_mean_abs_max": float(np.abs(channel_mean).max()),
        "channel_std_lt_0_005_count": low_std_count,
        "channel_std_lt_0_005_fraction": low_std_count / len(channel_std),
    }


def choose_view(leakage, valid_channels, mild_reasons, severe_reasons):
    if leakage:
        return "excluded_leakage"
    if severe_reasons:
        return "quarantine_quality_severe"
    if valid_channels < 8:
        return "aux_low_channel"
    if mild_reasons:
        return "aux_quality_flagged"
    return "main_train"


def view_weights(view, mild, source_weight, provenance_weight):
    if view == "main_train":
        channel_weight, quality_weight = 1.0, 1.0
    elif view == "aux_low_channel":
        channel_weight, quality_weight = 0.25, 0.5 if mild else 1.0
    elif view == "aux_quality_flagged":
        channel_weight, quality_weight = 1.0, 0.5
    else:
        channel_weight, quality_weight = 0.0, 0.0
    effective = (
        float(source_weight) * channel_weight * quality_weight
        * float(provenance_weight))
    return {
        "source_weight": float(source_weight),
        "channel_weight": channel_weight,
        "quality_weight": quality_weight,
        "provenance_weight": float(provenance_weight),
        "effective_weight": effective,
    }


def patch_valid_fractions(valid_samples, samples_per_patch, n_patches):
    fractions = []
    for patch in range(n_patches):
        start = patch * samples_per_patch
        valid = max(0, min(samples_per_patch, valid_samples - start))
        fractions.append(valid / samples_per_patch)
    return fractions


def process_recording(
        signal, raw_channel_names, dataset_id, source_object_identity,
        recording_locator, policy, raw_unit, raw_unit_evidence,
        subject=None, session=None, trial=None, split=None,
        provenance_tier="raw_header_or_continuous_hdf", source_weight=1.0,
        leakage=False, reader="unknown", reader_version=None,
        reader_fallback=False, sampling_rate=200.0, filter_parameters=None,
        window_sink=None, retain_windows=True):
    signal = np.asarray(signal)
    if signal.ndim != 2:
        raise ValueError(f"signal must have shape (channels, samples), got {signal.shape}")
    if signal.shape[0] != len(raw_channel_names):
        raise ValueError("raw channel count differs from signal")
    target_rate = float(policy["signal_contract"]["target_sfreq_hz"])
    if abs(float(sampling_rate) - target_rate) > 1e-6:
        raise ValueError(f"core requires {target_rate} Hz input, got {sampling_rate}")
    signal = signal.astype(np.float32, copy=False)
    recording_uid = make_recording_uid(
        source_object_identity, recording_locator, raw_channel_names)
    keep_indices, canonical_names = RESOLVE_CHANNELS(
        raw_channel_names, STANDARD_SET, signal.shape[0], dataset_id=dataset_id)
    mapped = signal[keep_indices] if keep_indices else np.empty(
        (0, signal.shape[1]), dtype=np.float32)
    unmapped_channels = [
        name for index, name in enumerate(raw_channel_names)
        if index not in set(keep_indices)]

    removed = []
    retained_indices = []
    for index, name in enumerate(canonical_names):
        values = mapped[index]
        finite = np.isfinite(values)
        nonfinite_fraction = 1.0 - float(finite.mean())
        finite_values = values[finite]
        exact_constant = (
            finite_values.size == 0
            or float(finite_values.min()) == float(finite_values.max()))
        reasons = []
        if nonfinite_fraction >= 0.01:
            reasons.append("nonfinite_fraction_gte_0.01")
        if exact_constant:
            reasons.append("exact_constant")
        if reasons:
            removed.append({
                "canonical_name": name,
                "raw_index": int(keep_indices[index]),
                "nonfinite_fraction": nonfinite_fraction,
                "reasons": reasons,
            })
        else:
            retained_indices.append(index)

    retained_names = [canonical_names[index] for index in retained_indices]
    retained = mapped[retained_indices] if retained_indices else np.empty(
        (0, signal.shape[1]), dtype=np.float32)
    base_manifest = {
        "schema_version": 1,
        "policy_version": policy["policy_version"],
        "recording_uid": recording_uid,
        "dataset_id": dataset_id,
        "source_object_identity": source_object_identity,
        "recording_locator": recording_locator,
        "subject": subject,
        "session": session,
        "trial": trial,
        "split": split,
        "raw_channel_names": list(raw_channel_names),
        "canonical_channel_names": retained_names,
        "unmapped_channel_names": unmapped_channels,
        "removed_channels": removed,
        "raw_unit": raw_unit,
        "raw_unit_evidence": raw_unit_evidence,
        "reader": reader,
        "reader_version": reader_version,
        "reader_fallback": bool(reader_fallback),
        "sampling_rate": target_rate,
        "filter_parameters": filter_parameters or {},
        "normalization": "recording_channel_zscore",
        "normalization_epsilon": float(
            policy["signal_contract"]["normalization_epsilon"]),
        "unit": "dimensionless_zscore",
        "provenance_tier": provenance_tier,
        "leakage": bool(leakage),
    }
    source_samples = signal.shape[1]
    accounting = Counter({"source_samples": source_samples})
    windows = []
    if retained.shape[0] == 0:
        accounting["quarantine_samples"] = source_samples
        base_manifest["terminal_view"] = "quarantine_nonfinite_or_decode"
        return {"manifest": base_manifest, "windows": windows,
                "accounting": dict(accounting)}

    simultaneous_finite = np.all(np.isfinite(retained), axis=0)
    runs = finite_runs(simultaneous_finite)
    finite_samples = int(simultaneous_finite.sum())
    accounting["finite_samples"] = finite_samples
    accounting["nonfinite_gap_samples"] = source_samples - finite_samples
    normalization_values = retained[:, simultaneous_finite]
    epsilon = float(policy["signal_contract"]["normalization_epsilon"])
    mean = normalization_values.mean(axis=1, keepdims=True)
    std = normalization_values.std(axis=1, keepdims=True)
    base_manifest["normalization_mean"] = mean[:, 0].astype(float).tolist()
    base_manifest["normalization_std"] = std[:, 0].astype(float).tolist()
    channel_signature = ordered_channel_signature(retained_names)
    base_manifest["channel_signature"] = channel_signature
    valid_channels = retained.shape[0]
    window_seconds = dynamic_window_seconds(valid_channels, policy)
    samples_per_patch = int(policy["signal_contract"]["samples_per_patch"])
    target_samples = int(round(window_seconds * target_rate))
    n_patches = target_samples // samples_per_patch
    provenance_weight = policy["provenance_tiers"][provenance_tier]
    clip_limit = float(policy["signal_contract"]["clip_std"])
    minimum_run = int(policy["nonfinite_contract"]["minimum_finite_run_samples"])
    view_counts = Counter()

    for run_index, (run_start, run_end) in enumerate(runs):
        run_length = run_end - run_start
        if run_length < minimum_run:
            accounting["short_finite_run_samples"] += run_length
            accounting["quarantine_samples"] += run_length
            continue
        offset = 0
        while offset < run_length:
            remaining = run_length - offset
            if remaining < minimum_run:
                accounting["short_tail_samples"] += remaining
                accounting["quarantine_samples"] += remaining
                break
            valid_samples = min(target_samples, remaining)
            start = run_start + offset
            stop = start + valid_samples
            raw_z = (retained[:, start:stop] - mean) / (std + epsilon)
            clip_fraction = float(np.mean(np.abs(raw_z) > clip_limit))
            clipped = np.clip(raw_z, -clip_limit, clip_limit).astype(np.float32)
            padding_samples = target_samples - valid_samples
            padding_fraction = padding_samples / target_samples
            mild_reasons, severe_reasons, quality_stats = classify_quality(
                raw_z, bool(removed), clip_fraction, padding_fraction,
                reader_fallback)
            view = choose_view(
                leakage, valid_channels, mild_reasons, severe_reasons)
            if padding_samples:
                data = np.zeros((valid_channels, target_samples), dtype=np.float32)
                data[:, :valid_samples] = clipped
            else:
                data = clipped
            padding_mask = np.zeros(target_samples, dtype=bool)
            if padding_samples:
                padding_mask[valid_samples:] = True
            sample_identity = json.dumps([
                recording_uid, channel_signature, start, valid_samples],
                separators=(",", ":"))
            sample_id = sha256_text(sample_identity)[:24]
            weights = view_weights(
                view, bool(mild_reasons), source_weight, provenance_weight)
            metadata = {
                "schema_version": 1,
                "policy_version": policy["policy_version"],
                "recording_uid": recording_uid,
                "sample_id": sample_id,
                "dataset_id": dataset_id,
                "view": view,
                "channel_names": retained_names,
                "channel_signature": channel_signature,
                "n_channels": valid_channels,
                "n_patches": n_patches,
                "window_seconds": window_seconds,
                "start_sample": start,
                "valid_samples": valid_samples,
                "padding_samples": padding_samples,
                "padding_fraction": padding_fraction,
                "valid_patch_fractions": patch_valid_fractions(
                    valid_samples, samples_per_patch, n_patches),
                "clip_fraction": clip_fraction,
                "mild_quality_reasons": mild_reasons,
                "severe_quality_reasons": severe_reasons,
                "quality_stats": quality_stats,
                "weights": weights,
                "unit": "dimensionless_zscore",
                "normalization": "recording_channel_zscore",
                "provenance_tier": provenance_tier,
            }
            window = {
                "data": data,
                "padding_mask": padding_mask,
                "metadata": metadata,
            }
            if window_sink is not None:
                window_sink(window)
            if retain_windows:
                windows.append(window)
            view_counts[view] += 1
            accounting[f"view_{view}_valid_samples"] += valid_samples
            accounting["padding_samples"] += padding_samples
            offset += valid_samples

    base_manifest["view_window_counts"] = dict(view_counts)
    base_manifest["window_count"] = sum(view_counts.values())
    base_manifest["valid_channels"] = valid_channels
    base_manifest["window_seconds"] = window_seconds
    accounted = (
        accounting["nonfinite_gap_samples"]
        + accounting["short_finite_run_samples"]
        + accounting["short_tail_samples"]
        + sum(value for key, value in accounting.items()
              if key.startswith("view_") and key.endswith("_valid_samples")))
    accounting["unique_accounted_source_samples"] = accounted
    if accounted != source_samples:
        raise ValueError(
            f"duration accounting mismatch: {accounted} != {source_samples}")
    return {"manifest": base_manifest, "windows": windows,
            "accounting": dict(accounting)}
