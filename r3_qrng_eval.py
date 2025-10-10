#!/usr/bin/env python3
"""
R3-Polynomial Temporal Recursion Discriminator — QRNG runner (laptop-safe)

- Implements R3 (v2): NMSE + null-referenced differencing check
- Adds resource guards for 1M-point sequences (subsample + cap)
- Loads QRNG CSVs (one numeric column) and prints/saves a verdict table

USAGE (PowerShell/CMD):
  python r3_qrng_eval.py
  # optional flags:
  python r3_qrng_eval.py --max-n 300000 --embed-d 12 --degree 3 --iaaft-iters 250 --seed 42

Outputs:
  - Console table
  - CSV report: r3_qrng_report.csv
"""

import argparse
import csv
import os
from pathlib import Path

import numpy as np
from numpy.fft import rfft, irfft
from scipy.signal import lfilter
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

# ------------------------- Defaults (laptop-safe) -------------------------

DEF_MAX_N       = 300_000   # cap effective length (was 1M; 300k is safer for poly features)
DEF_SUBSAMPLE   = 1         # keep every k-th sample (set >1 if you still hit RAM limits)
DEF_EMBED_D     = 10        # embedding dimension (10 w/ deg=3 → ~220 poly features)
DEF_POLY_DEG    = 3         # cubic captures nonlinearity without exploding RAM
DEF_RIDGE_A     = 1e-3
DEF_TEST_FRAC   = 0.30
DEF_NULL_TRIALS = 10
DEF_IAAFT_ITERS = 300
DEF_SEED        = 42

# Thresholds (same spirit as R3 v2)
Z_ERR_THR   = -2.0          # original must beat white-noise null strongly
Z_DIFF_THR  = 1.75          # falsifier z-diff threshold
RATIO_THR   = 1.35          # or ratio threshold
Z_DD_THR    = -1.0          # differenced series must NOT be far more predictable than its null
R_D_MAX     = 3.0           # and not crazily larger than original

# ------------------------------ Utilities ---------------------------------

def set_seed(seed: int): np.random.seed(seed)

def load_csv_1col(path: Path) -> np.ndarray:
    """Load one numeric column from CSV, ignoring headers/empty cells."""
    data = []
    with open(path, "r", newline="") as f:
        rdr = csv.reader(f)
        for row in rdr:
            if not row: continue
            try:
                v = float(row[0])
            except ValueError:
                # skip header or non-numeric
                continue
            data.append(v)
    if not data:
        raise ValueError(f"No numeric data found in {path}")
    return np.asarray(data, dtype=float)

def standardize(x: np.ndarray) -> tuple[np.ndarray, float, float]:
    mu = float(np.mean(x))
    sd = float(np.std(x) + 1e-12)
    return (x - mu) / sd, mu, sd

def hankel_supervised(x: np.ndarray, d: int) -> tuple[np.ndarray, np.ndarray]:
    """X[t] = [x_{t-d},...,x_{t-1}], y[t] = x_t"""
    T = len(x)
    if T <= d:
        raise ValueError(f"Signal too short for embedding: {T} <= {d}")
    # Build lag-matrix efficiently
    X = np.column_stack([x[i:T - d + i] for i in range(d)])
    y = x[d:]
    return X, y

def nmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mse = np.mean((y_true - y_pred) ** 2)
    denom = np.mean((y_true - np.mean(y_true)) ** 2) + 1e-12
    return float(mse / denom)

def train_test_split_ordered(X, y, frac=DEF_TEST_FRAC):
    n = len(y)
    n_te = max(1000, int(n * frac))
    n_tr = n - n_te
    return X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]

# -------------------- Predictive layer (polynomial ridge) ------------------

def predictive_error_poly(x: np.ndarray, d: int, degree: int, lmbd: float) -> float:
    xz, _, _ = standardize(x)
    X, y = hankel_supervised(xz, d=d)
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    Xp = poly.fit_transform(X)                      # beware of memory (controlled by d & degree)
    Xtr, ytr, Xte, yte = train_test_split_ordered(Xp, y, DEF_TEST_FRAC)
    model = Ridge(alpha=lmbd, fit_intercept=False, random_state=DEF_SEED)
    model.fit(Xtr, ytr)
    yhat = model.predict(Xte)
    return nmse(yte, yhat)

# ------------------------------ Falsifiers ---------------------------------

def permute_global(x: np.ndarray) -> np.ndarray:
    y = np.array(x, copy=True)
    np.random.shuffle(y)
    return y

def reverse_time(x: np.ndarray) -> np.ndarray:
    return x[::-1].copy()

def iaaft_surrogate(x: np.ndarray, iters=DEF_IAAFT_ITERS, seed=321) -> np.ndarray:
    """Preserve amplitude spectrum and marginal distribution; randomize phases."""
    set_seed(seed)
    x = np.asarray(x, float)
    n = len(x)
    amp = np.abs(rfft(x))
    sorted_x = np.sort(x)
    y = np.random.permutation(x)
    for _ in range(iters):
        Y = rfft(y)
        Y = amp * np.exp(1j * np.angle(Y))
        y_tmp = irfft(Y, n=n)
        ranks = np.argsort(np.argsort(y_tmp))
        y = sorted_x[ranks]
    return y

def diff_once(x: np.ndarray) -> np.ndarray:
    return np.diff(x)

# ------------------------------ Null models --------------------------------

def white_null_nmse(N: int, trials: int, d: int, degree: int, lmbd: float) -> tuple[float, float]:
    errs = []
    for t in range(trials):
        set_seed(10_000 + t)
        xw = np.random.randn(N)
        errs.append(predictive_error_poly(xw, d, degree, lmbd))
    errs = np.asarray(errs, float)
    return float(np.mean(errs)), float(np.std(errs) + 1e-12)

def white_null_nmse_diff(N: int, trials: int, d: int, degree: int, lmbd: float) -> tuple[float, float]:
    errs = []
    for t in range(trials):
        set_seed(20_000 + t)
        xw = np.random.randn(N)
        xd = diff_once(xw)
        # guard: diff shrinks by 1 length
        N_eff = min(len(xd), N-1)
        errs.append(predictive_error_poly(xd[:N_eff], d, degree, lmbd))
    errs = np.asarray(errs, float)
    return float(np.mean(errs)), float(np.std(errs) + 1e-12)

# ----------------------------- Evaluation core -----------------------------

def evaluate_temporal_recursion(
    x: np.ndarray,
    d: int, degree: int, lmbd: float,
    mu_null: float, sd_null: float,
    mu_diff_null: float, sd_diff_null: float
):
    # Original
    Eo = predictive_error_poly(x, d, degree, lmbd)
    z_err = (Eo - mu_null) / sd_null

    # Falsifiers
    Ep = predictive_error_poly(permute_global(x), d, degree, lmbd)
    Es = predictive_error_poly(iaaft_surrogate(x, DEF_IAAFT_ITERS, 321), d, degree, lmbd)
    Er = predictive_error_poly(reverse_time(x), d, degree, lmbd)

    # Differenced series (robustness to integrative processes)
    xd = diff_once(x)
    N_eff = max(d + 1000, len(xd))  # just to satisfy embedding length
    Ed = predictive_error_poly(xd, d, degree, lmbd)
    z_dd = (Ed - mu_diff_null) / sd_diff_null
    rD = Ed / (Eo + 1e-12)

    # Evidence calculations
    rP, rS, rR = Ep / (Eo + 1e-12), Es / (Eo + 1e-12), Er / (Eo + 1e-12)
    z_dp, z_ds, z_dr = (Ep - Eo) / (sd_null + 1e-12), (Es - Eo) / (sd_null + 1e-12), (Er - Eo) / (sd_null + 1e-12)

    hits = sum([
        (z_dp > Z_DIFF_THR) or (rP > RATIO_THR),
        (z_ds > Z_DIFF_THR) or (rS > RATIO_THR),
        (z_dr > Z_DIFF_THR) or (rR > RATIO_THR),
    ])

    # Verdict
    cond_predict = (z_err < Z_ERR_THR)
    cond_fals    = (hits >= 2)
    cond_diff    = (z_dd > Z_DD_THR) and (rD < R_D_MAX)

    verdict = "TEMPORALLY_LAWFUL" if (cond_predict and cond_fals and cond_diff) else "NON_LAWFUL"

    return {
        "verdict": verdict, "hits": int(hits),
        "E_orig": Eo, "E_perm": Ep, "E_surr": Es, "E_rev": Er, "E_diff": Ed,
        "z_err": z_err, "z_dp": z_dp, "z_ds": z_ds, "z_dr": z_dr, "z_dd": z_dd,
        "rP": rP, "rS": rS, "rR": rR, "rD": rD
    }

# --------------------------------- Main ------------------------------------

def main():
    global DEF_IAAFT_ITERS   # ← move this up here

    ap = argparse.ArgumentParser(description="R3 (v2) — QRNG temporal recursion discriminator")
    ap.add_argument("--anu",   default="anu_qrng_1M.csv")
    ap.add_argument("--nist",  default="nist_qrng_1M.csv")
    ap.add_argument("--idq",   default="idq_qrng_1M.csv")
    ap.add_argument("--max-n", type=int, default=DEF_MAX_N)
    ap.add_argument("--subsample", type=int, default=DEF_SUBSAMPLE)
    ap.add_argument("--embed-d", type=int, default=DEF_EMBED_D)
    ap.add_argument("--degree", type=int, default=DEF_POLY_DEG)
    ap.add_argument("--alpha", type=float, default=DEF_RIDGE_A)
    ap.add_argument("--iaaft-iters", type=int, default=DEF_IAAFT_ITERS)
    ap.add_argument("--null-trials", type=int, default=DEF_NULL_TRIALS)
    ap.add_argument("--seed", type=int, default=DEF_SEED)
    args = ap.parse_args()

    DEF_IAAFT_ITERS = args.iaaft_iters  # ← leave this line as is
    set_seed(args.seed)

    # Load datasets
    files = {
        "ANU":  Path(args.anu),
        "NIST": Path(args.nist),
        "IDQ":  Path(args.idq),
    }
    series = {}
    for name, p in files.items():
        if not p.exists():
            print(f"[WARN] Missing file: {p} (skipping)")
            continue
        x = load_csv_1col(p)
        if args.subsample > 1:
            x = x[::args.subsample]
        if len(x) > args.max_n:
            x = x[:args.max_n]
        series[name] = x

    if not series:
        print("No datasets found. Put anu_qrng_1M.csv, nist_qrng_1M.csv, idq_qrng_1M.csv next to this script.")
        return

    # Build nulls (based on effective N)
    N_eff = max(len(x) for x in series.values())
    mu_null, sd_null = white_null_nmse(N_eff, args.null_trials, args.embed_d, args.degree, args.alpha)
    mu_dnull, sd_dnull = white_null_nmse_diff(N_eff, args.null_trials, args.embed_d, args.degree, args.alpha)

    # Header
    header = (
        f"\n=== R3-Polynomial Temporal Recursion Discriminator (QRNG) ===\n\n"
        f"Params: N_eff≈{N_eff:,}, d={args.embed_d}, degree={args.degree}, IAAFT_iters={args.iaaft_iters}\n"
        f"Thresholds: Z_ERR_THR={Z_ERR_THR}, Z_DIFF_THR={Z_DIFF_THR}, RATIO_THR={RATIO_THR}, Z_DD_THR={Z_DD_THR}, R_D_MAX={R_D_MAX}\n"
        f"\n{'Source':<8} | {'Verdict':<18} | {'hits':>4} | {'E_orig':>10} | {'E_perm':>10} | {'E_surr':>10} | "
        f"{'E_rev':>10} | {'E_diff':>10} | {'z_err':>7} | {'z_dp':>7} | {'z_ds':>7} | {'z_dr':>7} | {'z_dd':>7} | "
        f"{'rP':>5} | {'rS':>5} | {'rR':>5} | {'rD':>6}"
    )
    print(header)
    print("-" * len(header))

    # CSV report
    out_rows = []
    for name, x in series.items():
        res = evaluate_temporal_recursion(
            x, args.embed_d, args.degree, args.alpha,
            mu_null, sd_null, mu_dnull, sd_dnull
        )
        print(
            f"{name:<8} | {res['verdict']:<18} | {res['hits']:>4d} | "
            f"{res['E_orig']:>10.3e} | {res['E_perm']:>10.3e} | {res['E_surr']:>10.3e} | "
            f"{res['E_rev']:>10.3e} | {res['E_diff']:>10.3e} | "
            f"{res['z_err']:>7.2f} | {res['z_dp']:>7.2f} | {res['z_ds']:>7.2f} | {res['z_dr']:>7.2f} | {res['z_dd']:>7.2f} | "
            f"{res['rP']:>5.2f} | {res['rS']:>5.2f} | {res['rR']:>5.2f} | {res['rD']:>6.2f}"
        )
        row = {"source": name, **res}
        out_rows.append(row)

    print("\nInterpretation:")
    print("  z_err < -2.0 ⇒ original much more predictable than white-noise null.")
    print("  Falsifier evidence: need ≥2 of (perm, surr, rev) with z_diff > 1.75 OR ratio > 1.35.")
    print("  Differencing robustness (null-referenced): z_dd > -1.0 AND rD < 3.0 (pink/random-walk should fail; chaos/AR should pass).")
    print("  Verdict = TEMPORALLY_LAWFUL if all three hold.\n")

    # Write CSV report
    out_path = Path("r3_qrng_report.csv")
    with open(out_path, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=out_rows[0].keys())
        wr.writeheader()
        wr.writerows(out_rows)
    print(f"Saved report → {out_path.resolve()}")
    print("Done.")
    

if __name__ == "__main__":
    main()
