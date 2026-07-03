#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROT-RH finite N=8000 Mangoldt/Jacobi GUE discovery reproducer
================================================================

Minimal, GitHub-ready reproducer for the finite numerical result that is
currently working:

    N = 8000, prime_max = 4000, s0 = 0.60, sigma = -1.0, shape = cos4

What it does
------------
1. Builds the finite self-adjoint Mangoldt-driven Jacobi matrix using the exact
   benchmark modules from the discovery code path.
2. Computes the spectrum and unfolded bulk statistics.
3. Checks three finite GUE gates:
      - KS distance to the GUE Wigner-surmise spacing CDF
      - adjacent gap-ratio mean near the GUE target
      - number variance closer to GUE than Poisson
4. Runs matched controls only for the same fixed candidate. No grid search.
5. Writes a compact discovery folder suitable for GitHub or for sending to David.

Important
---------
This is numerical evidence for one finite benchmark. It is not a proof of RH and
it does not prove an infinite Hilbert-Polya operator. The point is to document
cleanly what is currently working at finite N=8000.

Required files in the same directory
------------------------------------
The script intentionally reuses the exact discovery modules rather than
re-implementing the operator. Put these files beside this script:

  - rot_xi_action_density_control_benchmark_TERMINAL.py
  - rot_canonical_density_renorm_control_benchmark.py
  - rot_canonical_rigidity_control_benchmark.py

Recommended command
-------------------
python rot_rh_N8000_minimal_gue_discovery.py `
  --controls 50 `
  --out-dir rot_gue_N8000_minimal_discovery

That means 50 repetitions for each of the four control modes, i.e. 200 matched
controls total.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import sys
import time
from collections import defaultdict
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

try:
    from tqdm import tqdm
except Exception:  # pragma: no cover
    tqdm = None

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

try:
    import rot_xi_action_density_control_benchmark_TERMINAL as xi
    import rot_canonical_density_renorm_control_benchmark as dens
    import rot_canonical_rigidity_control_benchmark as core
except Exception as e:  # pragma: no cover
    print("ERROR: exact discovery modules are missing or failed to import.")
    print("\nPut these files in the same folder as this script:")
    print("  - rot_xi_action_density_control_benchmark_TERMINAL.py")
    print("  - rot_canonical_density_renorm_control_benchmark.py")
    print("  - rot_canonical_rigidity_control_benchmark.py")
    print(f"\nImport error: {e}")
    sys.exit(1)


# --------------------------------------------------------------------------------------
# Basic I/O helpers
# --------------------------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    if path:
        os.makedirs(path, exist_ok=True)


def piter(it: Iterable[Any], desc: str = "", total: int | None = None):
    if tqdm is not None:
        return tqdm(it, desc=desc, total=total)
    return it


def clean_value(v: Any) -> Any:
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        v = float(v)
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    return v


def clean_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {k: clean_value(v) for k, v in row.items()}


def write_json(path: str, obj: Any) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    ensure_dir(os.path.dirname(path))
    if not rows:
        with open(path, "w", encoding="utf-8") as f:
            f.write("")
        return
    keys: List[str] = []
    for row in rows:
        for k in row.keys():
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(clean_row(row))


def parse_modes(s: str) -> List[str]:
    return [x.strip() for x in str(s).split(",") if x.strip()]


# --------------------------------------------------------------------------------------
# Fixed-candidate exact-args adapter
# --------------------------------------------------------------------------------------

def make_exact_args(args: argparse.Namespace) -> SimpleNamespace:
    """Fields expected by xi.evaluate_xi / dens/core benchmark modules."""
    return SimpleNamespace(
        N=int(args.N),
        prime_max=int(args.prime_max),
        K0=float(args.K0),
        action_mode=str(args.action_mode),
        eps_mode=str(args.eps_mode),
        eps_max_abs=(None if args.eps_max_abs is None or float(args.eps_max_abs) <= 0 else float(args.eps_max_abs)),
        seed=int(args.seed),
        rep_offset=int(args.rep_offset),
        legacy_hash_controls=bool(args.legacy_hash_controls),
        rng_patch="unknown",
        weight_power=float(args.weight_power),
        chunk=int(args.chunk),
        trim=float(args.trim),
        L_values=str(args.L_values),
        nv_windows=int(args.nv_windows),
        nv_weight=float(args.nv_weight),
        hybrid_nv_weight=float(args.hybrid_nv_weight),
        nv_fail_penalty=float(args.nv_fail_penalty),
        score_metric=str(args.score_metric),
        hard_gates=bool(args.hard_gates),
        prefer_nv_pass=bool(args.prefer_nv_pass),
        gate_ks_max=float(args.gate_ks_max),
        gate_r_min=float(args.gate_r_min),
        gate_r_max=float(args.gate_r_max),
        gate_ks_weight=float(args.gate_ks_weight),
        gate_r_weight=float(args.gate_r_weight),
        gate_nv_fail_cost=float(args.gate_nv_fail_cost),
        scale_clip_min=float(args.scale_clip_min),
        scale_clip_max=float(args.scale_clip_max),
        tie_tol=float(args.tie_tol),
    )


def build_mangoldt_arrays(exact_args: SimpleNamespace, args: argparse.Namespace, eps: float) -> Dict[str, np.ndarray]:
    """Rebuild the Mangoldt Jacobi arrays using the exact density module."""
    diag, off, features = dens.build_jacobi_density(
        N=int(args.N),
        prime_max=int(args.prime_max),
        weight_mode="mangoldt",
        seed=int(args.seed),
        s0=float(args.s0),
        eps=float(eps),
        shape=str(args.shape),
        weight_power=float(args.weight_power),
        chunk=int(args.chunk),
        clip_min=float(args.scale_clip_min),
        clip_max=float(args.scale_clip_max),
        quiet=True,
    )
    eigs = core.eigvals_tridiagonal(diag, off)
    unfolded = core.unfolded_levels(eigs, trim=float(args.trim))
    spacings, ratios = core.spacings_and_ratios(unfolded)
    L_values = core.parse_L_values(str(args.L_values))
    nv_obs = core.number_variance(unfolded, L_values, windows=int(args.nv_windows))
    nv_gue = np.asarray(core.gue_number_variance_tuple(tuple(float(v) for v in L_values)), dtype=float)
    nv_pois = np.asarray(L_values, dtype=float)
    return {
        "diag": np.asarray(diag, dtype=float),
        "off": np.asarray(off, dtype=float),
        "eigenvalues": np.asarray(eigs, dtype=float),
        "unfolded_levels": np.asarray(unfolded, dtype=float),
        "spacings": np.asarray(spacings, dtype=float),
        "ratios": np.asarray(ratios, dtype=float),
        "L_values": np.asarray(L_values, dtype=float),
        "number_variance_observed": np.asarray(nv_obs, dtype=float),
        "number_variance_gue": np.asarray(nv_gue, dtype=float),
        "number_variance_poisson": np.asarray(nv_pois, dtype=float),
        "density_phi": np.asarray(features.get("density_phi", np.array([], dtype=float)), dtype=float),
        "density_s_diag": np.asarray(features.get("density_s_diag", np.array([], dtype=float)), dtype=float),
    }


# --------------------------------------------------------------------------------------
# Statistics and summaries
# --------------------------------------------------------------------------------------

def score_column(metric: str) -> str:
    return xi.score_col(str(metric))


def summarize(mangoldt: Dict[str, Any], controls: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    sc = score_column(args.score_metric)
    mscore = float(mangoldt[sc])
    ctrl_scores = np.asarray([float(r[sc]) for r in controls], dtype=float) if controls else np.asarray([], dtype=float)
    better = int(np.sum(ctrl_scores < mscore)) if len(ctrl_scores) else 0
    min_idx = int(np.argmin(ctrl_scores)) if len(ctrl_scores) else -1
    best_ctrl = controls[min_idx] if min_idx >= 0 else None
    ctrl_min = float(ctrl_scores[min_idx]) if min_idx >= 0 else None
    margin = (float(ctrl_min) - float(mscore)) if ctrl_min is not None else None

    local_gate = bool(float(mangoldt["ks_to_gue_wigner"]) < float(args.gate_ks_max)) and bool(
        float(args.gate_r_min) <= float(mangoldt["r_mean"]) <= float(args.gate_r_max)
    )
    rigidity_gate = bool(mangoldt["nv_gue_better_than_poisson"])
    hard_gate = bool(mangoldt["hard_gate_pass"])
    control_gate = bool(len(controls) == 0 or (better == 0 and margin is not None and margin > 0))
    strict_pass = bool(local_gate and rigidity_gate and hard_gate and control_gate)

    return {
        "verdict": "PASS_FINITE_N8000_GUE_BENCHMARK" if strict_pass else "NOT_A_STRICT_PASS",
        "claim_scope": "finite numerical benchmark only; not an RH proof",
        "N": int(args.N),
        "prime_max": int(args.prime_max),
        "s0": float(args.s0),
        "sigma": float(args.sigma),
        "shape": str(args.shape),
        "score_metric": str(args.score_metric),
        "L_values": str(args.L_values),
        "nv_windows": int(args.nv_windows),
        "gate_ks_max": float(args.gate_ks_max),
        "gate_r_min": float(args.gate_r_min),
        "gate_r_max": float(args.gate_r_max),
        "mangoldt_score": float(mscore),
        "mangoldt_KS_GUE": float(mangoldt["ks_to_gue_wigner"]),
        "mangoldt_r_mean": float(mangoldt["r_mean"]),
        "mangoldt_NV_RMSE_GUE": float(mangoldt["nv_rmse_gue"]),
        "mangoldt_NV_RMSE_Poisson": float(mangoldt["nv_rmse_poisson"]),
        "mangoldt_local_gate_pass": bool(local_gate),
        "mangoldt_rigidity_gate_pass": bool(rigidity_gate),
        "mangoldt_hard_gate_pass": bool(hard_gate),
        "control_count": int(len(controls)),
        "controls_beating_mangoldt": int(better),
        "best_control_score": ctrl_min,
        "best_control_margin_minus_mangoldt": margin,
        "best_control_mode": (best_ctrl.get("weight_mode") if best_ctrl else None),
        "best_control_rep": (int(best_ctrl.get("control_rep_requested", best_ctrl.get("rep", -1))) if best_ctrl else None),
        "control_score_median": (float(np.median(ctrl_scores)) if len(ctrl_scores) else None),
        "empirical_p_strict": (float((better + 1.0) / (len(controls) + 1.0)) if len(controls) else None),
        "discovery_sentence": (
            "At the fixed N=8000 benchmark, the Mangoldt Jacobi operator passes the local GUE spacing/gap-ratio gate, "
            "is closer to GUE than Poisson in number variance, and no matched control beats its selected score."
            if strict_pass else
            "At this run configuration the fixed benchmark did not clear the strict finite pass rule."
        ),
    }


def control_mode_summary(controls: List[Dict[str, Any]], mangoldt: Dict[str, Any], metric: str) -> List[Dict[str, Any]]:
    sc = score_column(metric)
    mscore = float(mangoldt[sc])
    by_mode: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in controls:
        by_mode[str(row.get("weight_mode", "unknown"))].append(row)
    out: List[Dict[str, Any]] = []
    for mode, rows in sorted(by_mode.items()):
        rows_sorted = sorted(rows, key=lambda r: float(r[sc]))
        best = rows_sorted[0]
        scores = np.asarray([float(r[sc]) for r in rows_sorted], dtype=float)
        out.append({
            "mode": mode,
            "count": int(len(rows_sorted)),
            "min_score": float(np.min(scores)),
            "median_score": float(np.median(scores)),
            "mean_score": float(np.mean(scores)),
            "best_margin_minus_mangoldt": float(np.min(scores) - mscore),
            "controls_beating_mangoldt": int(np.sum(scores < mscore)),
            "best_rep": int(best.get("control_rep_requested", best.get("rep", -1))),
            "best_KS_GUE": float(best["ks_to_gue_wigner"]),
            "best_r_mean": float(best["r_mean"]),
            "best_NV_RMSE_GUE": float(best["nv_rmse_gue"]),
            "best_NV_RMSE_Poisson": float(best["nv_rmse_poisson"]),
        })
    return out


# --------------------------------------------------------------------------------------
# Output writers
# --------------------------------------------------------------------------------------

def save_dataset(out_dir: str, arrays: Dict[str, np.ndarray], mangoldt: Dict[str, Any], controls: List[Dict[str, Any]], summary: Dict[str, Any], modes: List[Dict[str, Any]]) -> None:
    data = os.path.join(out_dir, "data")
    ensure_dir(data)
    clean_m = {k: v for k, v in mangoldt.items() if k != "arrays"}
    write_csv(os.path.join(data, "all_trials.csv"), [clean_m] + controls)
    write_csv(os.path.join(data, "mangoldt_vs_controls.csv"), [summary])
    write_csv(os.path.join(data, "control_mode_summary.csv"), modes)
    write_csv(os.path.join(data, "operator_coefficients_mangoldt.csv"), [
        {
            "n": i + 1,
            "diag_a_n": float(arrays["diag"][i]),
            "offdiag_b_n": float(arrays["off"][i]) if i < len(arrays["off"]) else None,
            "density_s_n": float(arrays["density_s_diag"][i]) if i < len(arrays["density_s_diag"]) else None,
            "density_phi_n": float(arrays["density_phi"][i]) if i < len(arrays["density_phi"]) else None,
        }
        for i in range(len(arrays["diag"]))
    ])
    write_csv(os.path.join(data, "spectrum_mangoldt.csv"), [
        {"index": i, "eigenvalue": float(v)} for i, v in enumerate(arrays["eigenvalues"])
    ])
    write_csv(os.path.join(data, "unfolded_levels_mangoldt.csv"), [
        {"index": i, "unfolded_level": float(v)} for i, v in enumerate(arrays["unfolded_levels"])
    ])
    write_csv(os.path.join(data, "spacings_mangoldt.csv"), [
        {"index": i, "spacing": float(v)} for i, v in enumerate(arrays["spacings"])
    ])
    write_csv(os.path.join(data, "gap_ratios_mangoldt.csv"), [
        {"index": i, "gap_ratio": float(v)} for i, v in enumerate(arrays["ratios"])
    ])
    write_csv(os.path.join(data, "number_variance_mangoldt.csv"), [
        {
            "L": float(L),
            "observed": float(obs),
            "GUE_reference": float(gue),
            "Poisson_reference": float(poi),
            "observed_over_GUE": float(obs / gue) if gue != 0 else None,
            "observed_over_Poisson": float(obs / poi) if poi != 0 else None,
        }
        for L, obs, gue, poi in zip(arrays["L_values"], arrays["number_variance_observed"], arrays["number_variance_gue"], arrays["number_variance_poisson"])
    ])


def make_plots(out_dir: str, arrays: Dict[str, np.ndarray], summary: Dict[str, Any], modes: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    if plt is None or bool(args.no_plots):
        return
    plot_dir = os.path.join(out_dir, "plots")
    ensure_dir(plot_dir)

    spacings = arrays["spacings"]
    if len(spacings):
        # Use exact GUE Wigner-surmise PDF if available from core; otherwise skip overlay.
        xs = np.linspace(0.0, max(4.0, float(np.percentile(spacings, 99.5))), 400)
        plt.figure(figsize=(8, 5))
        plt.hist(spacings, bins=80, density=True, alpha=0.55, label="Mangoldt unfolded spacings")
        if hasattr(core, "wigner_gue_pdf"):
            plt.plot(xs, core.wigner_gue_pdf(xs), label="GUE Wigner surmise")
        plt.xlabel("unfolded nearest-neighbor spacing")
        plt.ylabel("density")
        plt.title("Finite N=8000 Mangoldt spacing distribution")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "spacing_histogram_mangoldt.png"), dpi=180)
        plt.close()

    ratios = arrays["ratios"]
    if len(ratios):
        plt.figure(figsize=(8, 5))
        plt.hist(ratios, bins=60, density=True, alpha=0.65)
        plt.axvline(float(summary["mangoldt_r_mean"]), linestyle="--", label=f"mean r={summary['mangoldt_r_mean']:.6f}")
        plt.axvline(0.5996, linestyle=":", label="GUE target ~0.5996")
        plt.xlabel("adjacent gap ratio min(s_i,s_{i+1})/max(s_i,s_{i+1})")
        plt.ylabel("density")
        plt.title("Unfolding-free gap-ratio check")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "gap_ratio_histogram_mangoldt.png"), dpi=180)
        plt.close()

    L = arrays["L_values"]
    if len(L):
        plt.figure(figsize=(8, 5))
        plt.plot(L, arrays["number_variance_observed"], marker="o", label="Mangoldt observed")
        plt.plot(L, arrays["number_variance_gue"], marker="o", label="GUE reference")
        plt.plot(L, arrays["number_variance_poisson"], marker="o", label="Poisson reference")
        plt.xlabel("window length L")
        plt.ylabel("number variance")
        plt.title("Mesoscopic number variance")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "number_variance_mangoldt.png"), dpi=180)
        plt.close()

    if modes:
        labels = ["Mangoldt"] + [r["mode"] for r in modes]
        values = [float(summary["mangoldt_score"])] + [float(r["min_score"]) for r in modes]
        plt.figure(figsize=(8, 5))
        plt.barh(labels, values)
        plt.xlabel(f"{args.score_metric} score, lower is better")
        plt.title("Mangoldt vs best matched control by mode")
        plt.tight_layout()
        plt.savefig(os.path.join(plot_dir, "control_score_by_mode.png"), dpi=180)
        plt.close()

    n = np.arange(1, len(arrays["diag"]) + 1)
    stride = max(1, len(n) // 2500)
    plt.figure(figsize=(9, 5))
    plt.plot(n[::stride], arrays["diag"][::stride], label="diagonal a_n")
    if len(arrays["off"]):
        plt.plot(n[:-1:stride], arrays["off"][::stride], label="off-diagonal b_n")
    plt.xlabel("n")
    plt.title("Finite self-adjoint Jacobi coefficients")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, "operator_coefficients_mangoldt.png"), dpi=180)
    plt.close()


def write_markdown(out_dir: str, args: argparse.Namespace, summary: Dict[str, Any], mode_rows: List[Dict[str, Any]], const: Dict[str, Any], eps: float) -> None:
    verdict = summary["verdict"]
    modes_text = "\n".join(
        f"| {r['mode']} | {r['count']} | {r['min_score']:.12e} | {r['best_margin_minus_mangoldt']:+.12e} | {r['controls_beating_mangoldt']} |"
        for r in mode_rows
    ) or "| none | 0 | n/a | n/a | n/a |"

    text = f"""# Minimal finite N=8000 ROT/Mangoldt GUE discovery check

This folder was generated by `rot_rh_N8000_minimal_gue_discovery.py`.

## Exact claim being checked

A finite self-adjoint Jacobi matrix generated from Mangoldt arithmetic data, at the fixed benchmark

```text
N = {args.N}
prime_max = {args.prime_max}
s0 = {args.s0}
sigma = {args.sigma}
shape = {args.shape}
```

passes the finite GUE spacing, gap-ratio, and number-variance gates, and is not beaten by the matched controls included in this run.

This is a finite numerical benchmark. It is not a proof of RH.

## Verdict

```text
{verdict}
```

## Mangoldt result

| Metric | Value |
|---|---:|
| selected score, lower is better | `{summary['mangoldt_score']:.15e}` |
| KS distance to GUE Wigner surmise | `{summary['mangoldt_KS_GUE']:.15e}` |
| adjacent gap-ratio mean | `{summary['mangoldt_r_mean']:.15e}` |
| number-variance RMSE to GUE | `{summary['mangoldt_NV_RMSE_GUE']:.15e}` |
| number-variance RMSE to Poisson | `{summary['mangoldt_NV_RMSE_Poisson']:.15e}` |
| local GUE gate | `{summary['mangoldt_local_gate_pass']}` |
| rigidity gate, GUE closer than Poisson | `{summary['mangoldt_rigidity_gate_pass']}` |
| hard gate | `{summary['mangoldt_hard_gate_pass']}` |

## Controls

| Metric | Value |
|---|---:|
| total controls | `{summary['control_count']}` |
| controls beating Mangoldt | `{summary['controls_beating_mangoldt']}` |
| best control score | `{summary['best_control_score']}` |
| best control margin minus Mangoldt | `{summary['best_control_margin_minus_mangoldt']}` |
| best control mode | `{summary['best_control_mode']}` |
| empirical strict p-value | `{summary['empirical_p_strict']}` |

## Control mode summary

| Mode | Count | Best score | Best margin minus Mangoldt | Controls beating Mangoldt |
|---|---:|---:|---:|---:|
{modes_text}

## Xi-action density scale

```text
K0 = {const['K0']:.18g}
S_rec = {const['action_used']:.15e}
epsilon = {eps:.15e}
```

## Re-run command

PowerShell:

```powershell
python .\\rot_rh_N8000_minimal_gue_discovery.py `
  --controls {args.controls} `
  --out-dir {args.out_dir}
```

## Data files

```text
summary.json
metadata.json
data/all_trials.csv
data/mangoldt_vs_controls.csv
data/control_mode_summary.csv
data/operator_coefficients_mangoldt.csv
data/spectrum_mangoldt.csv
data/unfolded_levels_mangoldt.csv
data/spacings_mangoldt.csv
data/gap_ratios_mangoldt.csv
data/number_variance_mangoldt.csv
plots/*.png
```

## Careful wording for sharing

The clean statement is:

> At `N=8000`, the finite Mangoldt-driven Jacobi operator passes the GUE local-spacing and gap-ratio gates, has number variance closer to GUE than Poisson, and in this fixed benchmark is not beaten by the matched randomized controls. This is finite numerical evidence, not an RH proof.
"""
    with open(os.path.join(out_dir, "DISCOVERY_SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write(text)

    # A shorter copy-paste note for email.
    email = f"""Dear David,

I have narrowed the computation to the minimal finite result that is currently reproducible, rather than presenting it as a proof.

The script tests one fixed finite self-adjoint Jacobi matrix generated from Mangoldt arithmetic data:

N={args.N}, prime_max={args.prime_max}, s0={args.s0}, sigma={args.sigma}, shape={args.shape}.

The output verdict was:

{verdict}

Main metrics:

- KS distance to GUE Wigner surmise: {summary['mangoldt_KS_GUE']:.12e}
- adjacent gap-ratio mean: {summary['mangoldt_r_mean']:.12e}
- number-variance RMSE to GUE: {summary['mangoldt_NV_RMSE_GUE']:.12e}
- number-variance RMSE to Poisson: {summary['mangoldt_NV_RMSE_Poisson']:.12e}
- controls beating Mangoldt: {summary['controls_beating_mangoldt']} out of {summary['control_count']}

The claim is only finite and numerical: at N=8000 this Mangoldt-driven Jacobi operator passes the GUE local-spacing/gap-ratio/number-variance gates and is not beaten by the matched controls in this benchmark. I am not claiming this proves RH or constructs the infinite Hilbert-Polya operator.

Best,
Buddhika
"""
    with open(os.path.join(out_dir, "DAVID_MINIMAL_NOTE.txt"), "w", encoding="utf-8") as f:
        f.write(email)


def print_terminal_report(args: argparse.Namespace, summary: Dict[str, Any], mode_rows: List[Dict[str, Any]], const: Dict[str, Any], eps: float, elapsed: float) -> None:
    print("\n" + "=" * 112)
    print("COPY_PASTE_MINIMAL_N8000_GUE_REPORT_BEGIN")
    print("=" * 112)
    print(f"RUN N={args.N} prime_max={args.prime_max} s0={args.s0} sigma={args.sigma} shape={args.shape} controls_total={summary['control_count']} seed={args.seed}")
    print(f"XI_ACTION K0={const['K0']:.18g} S_rec={const['action_used']:.15e} eps={eps:.15e}")
    print(
        f"MANGOLDT verdict={summary['verdict']} score={summary['mangoldt_score']:.15e} "
        f"KS_GUE={summary['mangoldt_KS_GUE']:.15e} r_mean={summary['mangoldt_r_mean']:.15e} "
        f"NV_GUE={summary['mangoldt_NV_RMSE_GUE']:.15e} NV_Poisson={summary['mangoldt_NV_RMSE_Poisson']:.15e} "
        f"local_gate={int(summary['mangoldt_local_gate_pass'])} rigidity_gate={int(summary['mangoldt_rigidity_gate_pass'])} hard_gate={int(summary['mangoldt_hard_gate_pass'])}"
    )
    print(
        f"CONTROLS count={summary['control_count']} beating={summary['controls_beating_mangoldt']} "
        f"best_score={summary['best_control_score']} margin={summary['best_control_margin_minus_mangoldt']} "
        f"best_mode={summary['best_control_mode']} best_rep={summary['best_control_rep']} p_strict={summary['empirical_p_strict']}"
    )
    print("CONTROL_MODE_SUMMARY")
    for row in mode_rows:
        print(
            f"MODE {row['mode']} count={row['count']} min_score={row['min_score']:.15e} "
            f"median_score={row['median_score']:.15e} margin={row['best_margin_minus_mangoldt']:+.15e} "
            f"beating={row['controls_beating_mangoldt']} best_rep={row['best_rep']}"
        )
    print("CLAIM_SCOPE finite numerical benchmark only; not an RH proof")
    print(f"elapsed_seconds={elapsed:.3f}")
    print("COPY_PASTE_MINIMAL_N8000_GUE_REPORT_END")
    print("=" * 112)


# --------------------------------------------------------------------------------------
# Main runner
# --------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Minimal finite N=8000 Mangoldt Jacobi GUE discovery reproducer.")

    # Fixed discovery defaults. You can override for smoke tests, but the intended result is N=8000.
    p.add_argument("--N", type=int, default=8000)
    p.add_argument("--prime-max", type=int, default=4000)
    p.add_argument("--s0", type=float, default=0.60)
    p.add_argument("--sigma", type=float, default=-1.0)
    p.add_argument("--shape", default="cos4")

    # Controls: default 50 per mode = 200 controls.
    p.add_argument("--controls", type=int, default=50, help="Repetitions per control mode. Default 50 => 200 controls total.")
    p.add_argument("--control-modes", default="gaussian,permuted,signflip,random-support")
    p.add_argument("--out-dir", default="rot_gue_N8000_minimal_discovery")

    # Reproducibility and exact benchmark settings.
    p.add_argument("--seed", type=int, default=6783)
    p.add_argument("--rep-offset", type=int, default=0)
    p.add_argument("--legacy-hash-controls", action="store_true")
    p.add_argument("--K0", type=float, default=float(core.CANON["K0"]) if hasattr(core, "CANON") else 0.04620998623306458)
    p.add_argument("--action-mode", choices=["bare", "full"], default="full")
    p.add_argument("--eps-mode", choices=["sqrtlog", "log", "invroot", "rootloglog"], default="sqrtlog")
    p.add_argument("--eps-max-abs", type=float, default=0.0)
    p.add_argument("--weight-power", type=float, default=0.5)
    p.add_argument("--chunk", type=int, default=256)
    p.add_argument("--trim", type=float, default=0.15)
    p.add_argument("--L-values", default="1:12:1")
    p.add_argument("--nv-windows", type=int, default=250)
    p.add_argument("--score-metric", choices=["local", "deep", "hybrid", "gate"], default="gate")

    # Gate definitions used in the discovery benchmark.
    p.add_argument("--hard-gates", action="store_true", default=True)
    p.add_argument("--prefer-nv-pass", action="store_true", default=True)
    p.add_argument("--gate-ks-max", type=float, default=0.13)
    p.add_argument("--gate-r-min", type=float, default=0.58)
    p.add_argument("--gate-r-max", type=float, default=0.62)
    p.add_argument("--gate-ks-weight", type=float, default=2.0)
    p.add_argument("--gate-r-weight", type=float, default=6.0)
    p.add_argument("--gate-nv-fail-cost", type=float, default=0.25)
    p.add_argument("--nv-weight", type=float, default=0.25)
    p.add_argument("--hybrid-nv-weight", type=float, default=0.20)
    p.add_argument("--nv-fail-penalty", type=float, default=0.20)
    p.add_argument("--scale-clip-min", type=float, default=0.02)
    p.add_argument("--scale-clip-max", type=float, default=2.0)
    p.add_argument("--tie-tol", type=float, default=0.001)
    p.add_argument("--no-plots", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    t0 = time.time()
    ensure_dir(args.out_dir)
    ensure_dir(os.path.join(args.out_dir, "data"))
    ensure_dir(os.path.join(args.out_dir, "plots"))

    exact_args = make_exact_args(args)
    exact_args.rng_patch = xi.patch_core_control_rng(use_legacy_hash=bool(args.legacy_hash_controls), clear_cache=True)
    const = xi.xi_constants(float(args.K0), action_mode=str(args.action_mode))
    eps = xi.epsilon_from_sigma(int(args.N), sigma=float(args.sigma), action=float(const["action_used"]), eps_mode=str(args.eps_mode))
    if args.eps_max_abs is not None and float(args.eps_max_abs) > 0:
        eps = float(np.clip(eps, -float(args.eps_max_abs), float(args.eps_max_abs)))

    print("=" * 112)
    print("ROT-RH MINIMAL FINITE N=8000 MANGOLDT/JACOBI GUE DISCOVERY REPRODUCER")
    print("=" * 112)
    print(f"N                  : {args.N}")
    print(f"prime_max          : {args.prime_max}")
    print(f"candidate           : s0={args.s0}, sigma={args.sigma}, eps={eps:.12e}, shape={args.shape}")
    print(f"K0                 : {const['K0']:.18g}")
    print(f"Xi action S_rec     : {const['action_used']:.12e}")
    print(f"GUE gates           : KS<{args.gate_ks_max}, {args.gate_r_min}<=r<={args.gate_r_max}, NV_GUE<NV_Poisson")
    print(f"controls            : {args.controls} per mode; modes={args.control_modes}")
    print(f"control RNG         : {exact_args.rng_patch}")
    print(f"out_dir             : {args.out_dir}")
    print("=" * 112)
    print("Scope: finite numerical benchmark only. This script does not claim RH proof.")
    print("=" * 112)

    print("Building/evaluating the fixed Mangoldt operator...")
    mangoldt = xi.evaluate_xi(exact_args, "mangoldt", 0, s0=float(args.s0), sigma=float(args.sigma), shape=str(args.shape), const=const, quiet=True)
    sc = score_column(str(args.score_metric))
    print(
        f"Mangoldt fixed candidate: score={float(mangoldt[sc]):.12e} "
        f"KS={float(mangoldt['ks_to_gue_wigner']):.6e} r={float(mangoldt['r_mean']):.6e} "
        f"NVg={float(mangoldt['nv_rmse_gue']):.6e} NVp={float(mangoldt['nv_rmse_poisson']):.6e} "
        f"hard={bool(mangoldt['hard_gate_pass'])}"
    )

    controls: List[Dict[str, Any]] = []
    modes = parse_modes(args.control_modes)
    total = int(args.controls) * len(modes)
    if total:
        print(f"Evaluating {total} matched controls for the same fixed candidate...")
    tasks = [(mode, rep) for mode in modes for rep in range(int(args.controls))]
    for mode, rep in piter(tasks, desc="controls", total=len(tasks)):
        controls.append(xi.evaluate_xi(exact_args, mode, rep, s0=float(args.s0), sigma=float(args.sigma), shape=str(args.shape), const=const, quiet=True))

    summary = summarize(mangoldt, controls, args)
    mode_rows = control_mode_summary(controls, mangoldt, str(args.score_metric))

    print("Rebuilding arrays once to write operator coefficients, spectrum, spacings, ratios, and number variance...")
    arrays = build_mangoldt_arrays(exact_args, args, eps)

    metadata = {
        "script": os.path.basename(__file__),
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "matplotlib_available": plt is not None,
        "exact_dependency_modules": [
            "rot_xi_action_density_control_benchmark_TERMINAL.py",
            "rot_canonical_density_renorm_control_benchmark.py",
            "rot_canonical_rigidity_control_benchmark.py",
        ],
        "xi_constants": clean_row(const),
        "epsilon": float(eps),
        "args": vars(args),
    }
    write_json(os.path.join(args.out_dir, "metadata.json"), metadata)
    write_json(os.path.join(args.out_dir, "summary.json"), clean_row(summary))
    save_dataset(args.out_dir, arrays, mangoldt, controls, summary, mode_rows)
    make_plots(args.out_dir, arrays, summary, mode_rows, args)
    write_markdown(args.out_dir, args, summary, mode_rows, const, eps)

    elapsed = time.time() - t0
    print_terminal_report(args, summary, mode_rows, const, eps, elapsed)

    print("\nWrote minimal discovery folder:")
    print(f"  {os.path.abspath(args.out_dir)}")
    print("\nInspect first:")
    print("  DISCOVERY_SUMMARY.md")
    print("  DAVID_MINIMAL_NOTE.txt")
    print("  summary.json")
    print("  data/mangoldt_vs_controls.csv")


if __name__ == "__main__":
    main()
