#!/usr/bin/env python3
"""Compute objective speech enhancement metrics for benchmark outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median
from typing import Dict, Iterable, List, Tuple

import numpy as np
import soundfile as sf
from pystoi import stoi as stoi_fn


EPS = 1e-10


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compute objective metrics for DFNet vs Krisp")
    p.add_argument(
        "--run-dir",
        type=Path,
        required=True,
        help="Benchmark directory, e.g. .../nc_launch_v1_2026-02-26",
    )
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument(
        "--align-systems",
        type=str,
        default="",
        help="Comma-separated systems to latency-align before scoring (e.g. krisp).",
    )
    p.add_argument(
        "--align-max-shift-ms",
        type=float,
        default=60.0,
        help="Max absolute shift for latency alignment search.",
    )
    return p.parse_args()


def load_mono(path: Path) -> Tuple[np.ndarray, int]:
    x, sr = sf.read(str(path), always_2d=True, dtype="float32")
    return x.mean(axis=1).astype(np.float64), int(sr)


def align_lengths(*signals: np.ndarray) -> List[np.ndarray]:
    n = min(len(s) for s in signals)
    return [s[:n] for s in signals]


def snr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    noise = reference - estimate
    num = float(np.sum(reference * reference))
    den = float(np.sum(noise * noise))
    return 10.0 * math.log10((num + EPS) / (den + EPS))


def si_sdr_db(reference: np.ndarray, estimate: np.ndarray) -> float:
    ref_energy = float(np.sum(reference * reference)) + EPS
    alpha = float(np.dot(estimate, reference)) / ref_energy
    target = alpha * reference
    noise = estimate - target
    num = float(np.sum(target * target))
    den = float(np.sum(noise * noise))
    return 10.0 * math.log10((num + EPS) / (den + EPS))


def percentile(values: List[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize(values: Iterable[float]) -> Dict[str, float]:
    vals = [float(v) for v in values]
    return {
        "mean": float(mean(vals)),
        "median": float(median(vals)),
        "p5": percentile(vals, 5),
        "p95": percentile(vals, 95),
    }


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def load_metadata(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def safe_stoi(clean: np.ndarray, deg: np.ndarray, sr: int, extended: bool) -> float:
    try:
        return float(stoi_fn(clean, deg, sr, extended=extended))
    except Exception:
        return float("nan")


def estimate_best_shift(reference: np.ndarray, degraded: np.ndarray, max_shift_samples: int) -> int:
    """Return shift (samples) to apply to degraded for best alignment to reference.

    Positive shift means degraded should be shifted left.
    """
    max_shift = int(max(0, max_shift_samples))
    ref = reference.astype(np.float64, copy=False)
    deg = degraded.astype(np.float64, copy=False)
    n = min(len(ref), len(deg))
    ref = ref[:n]
    deg = deg[:n]

    best_shift = 0
    best_score = -1e30
    for shift in range(-max_shift, max_shift + 1):
        if shift > 0:
            a = ref[: n - shift]
            b = deg[shift:n]
        elif shift < 0:
            s = -shift
            a = ref[s:n]
            b = deg[: n - s]
        else:
            a = ref
            b = deg
        if len(a) < 1024:
            continue
        score = float(np.dot(a, b))
        if score > best_score:
            best_score = score
            best_shift = shift
    return int(best_shift)


def apply_shift(signal: np.ndarray, shift: int) -> np.ndarray:
    """Apply sample shift while preserving length."""
    n = len(signal)
    if shift == 0:
        return signal
    if shift > 0:
        if shift >= n:
            return np.zeros_like(signal)
        return np.concatenate([signal[shift:], np.zeros(shift, dtype=signal.dtype)], axis=0)
    s = -shift
    if s >= n:
        return np.zeros_like(signal)
    return np.concatenate([np.zeros(s, dtype=signal.dtype), signal[: n - s]], axis=0)


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    align_systems = {s.strip().lower() for s in args.align_systems.split(",") if s.strip()}
    max_shift_samples = int(round(args.align_max_shift_ms * args.sample_rate / 1000.0))

    dataset_dir = run_dir / "dataset"
    outputs_dfnet = run_dir / "outputs" / "dfnet"
    outputs_krisp = run_dir / "outputs" / "krisp"
    objective_dir = run_dir / "metrics" / "objective"
    objective_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows = load_metadata(dataset_dir / "metadata.csv")
    per_sample_rows: List[Dict[str, object]] = []

    for idx, meta in enumerate(metadata_rows):
        sid = meta["id"]
        clean, sr_clean = load_mono(dataset_dir / "clean" / f"{sid}.wav")
        noisy, sr_noisy = load_mono(dataset_dir / "noisy" / f"{sid}.wav")
        dfnet, sr_df = load_mono(outputs_dfnet / f"{sid}.wav")
        krisp, sr_kr = load_mono(outputs_krisp / f"{sid}.wav")
        if len({sr_clean, sr_noisy, sr_df, sr_kr}) != 1:
            raise ValueError(f"Sample-rate mismatch for {sid}: {sr_clean}, {sr_noisy}, {sr_df}, {sr_kr}")

        clean, noisy, dfnet, krisp = align_lengths(clean, noisy, dfnet, krisp)

        noisy_sisdr = si_sdr_db(clean, noisy)
        noisy_snr = snr_db(clean, noisy)
        noisy_stoi = safe_stoi(clean, noisy, sr_clean, extended=False)
        noisy_estoi = safe_stoi(clean, noisy, sr_clean, extended=True)

        for system_name, enh in [("dfnet", dfnet), ("krisp", krisp)]:
            applied_shift = 0
            if system_name in align_systems and max_shift_samples > 0:
                applied_shift = estimate_best_shift(noisy, enh, max_shift_samples)
                enh = apply_shift(enh, applied_shift)

            enh_sisdr = si_sdr_db(clean, enh)
            enh_snr = snr_db(clean, enh)
            enh_stoi = safe_stoi(clean, enh, sr_clean, extended=False)
            enh_estoi = safe_stoi(clean, enh, sr_clean, extended=True)

            per_sample_rows.append(
                {
                    "id": sid,
                    "system": system_name,
                    "condition": meta["condition"],
                    "sir_db": float(meta["sir_db"]),
                    "snr_db": float(meta["snr_db"]),
                    "si_sdr_noisy": noisy_sisdr,
                    "si_sdr_enh": enh_sisdr,
                    "si_sdri": enh_sisdr - noisy_sisdr,
                    "snr_noisy": noisy_snr,
                    "snr_enh": enh_snr,
                    "snri": enh_snr - noisy_snr,
                    "stoi_noisy": noisy_stoi,
                    "stoi_enh": enh_stoi,
                    "stoi_gain": enh_stoi - noisy_stoi,
                    "estoi_noisy": noisy_estoi,
                    "estoi_enh": enh_estoi,
                    "estoi_gain": enh_estoi - noisy_estoi,
                    "alignment_shift_samples": applied_shift,
                }
            )
        if (idx + 1) % 100 == 0:
            print(f"processed {idx + 1}/{len(metadata_rows)} samples")

    write_csv(objective_dir / "per_sample.csv", per_sample_rows)

    # Aggregate summaries
    summary: Dict[str, object] = {"systems": {}, "delta_dfnet_minus_krisp": {}}
    by_condition_rows: List[Dict[str, object]] = []
    metrics = ["si_sdri", "snri", "stoi_gain", "estoi_gain", "si_sdr_enh", "snr_enh", "stoi_enh", "estoi_enh"]

    grouped: Dict[str, List[Dict[str, object]]] = {"dfnet": [], "krisp": []}
    for row in per_sample_rows:
        grouped[str(row["system"])].append(row)

    for system, rows in grouped.items():
        sys_summary: Dict[str, object] = {"n": len(rows), "metrics": {}, "negative_gain_rate": {}}
        for m in metrics:
            vals = [float(r[m]) for r in rows if not math.isnan(float(r[m]))]
            sys_summary["metrics"][m] = summarize(vals)
        sys_summary["negative_gain_rate"]["si_sdri_lt_0"] = float(np.mean([float(r["si_sdri"]) < 0 for r in rows]))
        sys_summary["negative_gain_rate"]["snri_lt_0"] = float(np.mean([float(r["snri"]) < 0 for r in rows]))
        sys_summary["negative_gain_rate"]["stoi_gain_lt_0"] = float(np.mean([float(r["stoi_gain"]) < 0 for r in rows]))
        sys_summary["negative_gain_rate"]["estoi_gain_lt_0"] = float(np.mean([float(r["estoi_gain"]) < 0 for r in rows]))
        summary["systems"][system] = sys_summary

        for cond in ["easy", "medium", "hard"]:
            crows = [r for r in rows if str(r["condition"]) == cond]
            if not crows:
                continue
            out_row: Dict[str, object] = {"system": system, "condition": cond, "n": len(crows)}
            for m in ["si_sdri", "snri", "stoi_gain", "estoi_gain"]:
                vals = [float(r[m]) for r in crows if not math.isnan(float(r[m]))]
                out_row[f"{m}_mean"] = float(mean(vals))
                out_row[f"{m}_median"] = float(median(vals))
            by_condition_rows.append(out_row)

    # Paired deltas DFNet - Krisp
    df_by_id = {str(r["id"]): r for r in grouped["dfnet"]}
    kr_by_id = {str(r["id"]): r for r in grouped["krisp"]}
    common_ids = sorted(set(df_by_id.keys()) & set(kr_by_id.keys()))
    for m in ["si_sdri", "snri", "stoi_gain", "estoi_gain", "si_sdr_enh", "snr_enh", "stoi_enh", "estoi_enh"]:
        deltas = [float(df_by_id[i][m]) - float(kr_by_id[i][m]) for i in common_ids]
        summary["delta_dfnet_minus_krisp"][m] = summarize(deltas)

    with (objective_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_csv(objective_dir / "by_condition.csv", by_condition_rows)

    print(f"Wrote: {objective_dir / 'per_sample.csv'}")
    print(f"Wrote: {objective_dir / 'summary.json'}")
    print(f"Wrote: {objective_dir / 'by_condition.csv'}")


if __name__ == "__main__":
    main()
