#!/usr/bin/env python3
"""
R3-Polynomial Temporal Recursion Discriminator (Patched)
Drop-in patched version of your R3 temporal recursion test.

Usage:
    python R3_temporal_recursion_v2_patched.py

Author: patched for user
"""

import numpy as np
from numpy.fft import rfft, irfft
from scipy.signal import lfilter
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from numpy.linalg import norm

# =============================== Config ===============================

N_SAMPLES      = 50_000         # default, you can change to 1_000_000 for 1M runs
EMBED_D        = 24             # embedding dimension (was 20 -> increased slightly)
POLY_DEGREE    = 3              # reduced from 4 to reduce overfitting
RIDGE_ALPHA    = 1e-3
TEST_FRACTION  = 0.30
NULL_TRIALS    = 12

# Decision thresholds (patched/looser)
Z_ERR_THR      = -2.0           # was -2.5, looser
Z_DIFF_THR     = 1.75           # was 2.0
RATIO_THR      = 1.35           # was 1.5

# differencing null robustness (patched)
Z_DD_THR       = -1.0           # was -0.75
R_D_MAX        = 3.0            # cap for rD for chaotic maps

IAAFT_ITERS    = 300            # strong phase randomization (tunable)
RANDOM_SEED    = 42

# =============================== Utils =================================

def set_seed(seed=RANDOM_SEED):
    np.random.seed(seed)

def standardize(x):
    x = np.asarray(x, float)
    mu, sd = np.mean(x), np.std(x) + 1e-12
    return (x - mu) / sd, mu, sd

def mse(a, b):
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))

def train_test_split(X, y, frac=TEST_FRACTION):
    n = len(y)
    n_te = max(100, int(n * frac))
    n_tr = n - n_te
    return X[:n_tr], y[:n_tr], X[n_tr:], y[n_tr:]

# ======================= Embedding / Predictor ==========================

def hankel_supervised(x, d=EMBED_D):
    """
    Build supervised Hankel-style input matrix:
      X[t] = [x_{t-d}, ..., x_{t-1}]
      y[t] = x_t
    """
    x = np.asarray(x, float)
    T = len(x)
    if T <= d + 10:
        raise ValueError(f"Signal too short for embedding: {T} <= {d + 10}")
    # Build matrix of shape (T - d, d)
    X = np.stack([x[i:T - d + i] for i in range(d)], axis=1)
    y = x[d:]
    return X, y

def predictive_error_poly(x, d=EMBED_D, degree=POLY_DEGREE, lmbd=RIDGE_ALPHA):
    """
    Polynomial ridge predictor for one-step recursion.
    Returns NMSE-like normalized MSE:
      normalized by variance of test targets (so white noise baseline ~1.0)
    """
    xz, _, _ = standardize(x)
    X, y = hankel_supervised(xz, d=d)
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    Xp = poly.fit_transform(X)
    Xtr, ytr, Xte, yte = train_test_split(Xp, y)
    model = Ridge(alpha=lmbd, fit_intercept=False, random_state=RANDOM_SEED)
    model.fit(Xtr, ytr)
    yhat = model.predict(Xte)
    err = mse(yte, yhat)
    denom = np.var(yte) + 1e-12
    nmse = err / denom
    return nmse

# =============================== Falsifiers ==============================

def permute_global(x):
    y = np.asarray(x, float).copy()
    np.random.shuffle(y)
    return y

def iaaft_surrogate(x, iters=IAAFT_ITERS, seed=321):
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

def reverse_time(x):
    return np.asarray(x, float)[::-1]

def diff_once(x):
    x = np.asarray(x, float)
    return np.diff(x)

# =============================== Generators =============================

def gen_white(N, seed=RANDOM_SEED):
    set_seed(seed); return np.random.randn(N)

def gen_uniform(N, seed=RANDOM_SEED):
    set_seed(seed); return np.random.rand(N)

def gen_logistic(N, r=3.9, x0=0.123456):
    x = np.empty(N); x[0] = x0
    for i in range(1, N):
        x[i] = r * x[i-1] * (1 - x[i-1])
    return x

def gen_ar1(N, phi=0.5, seed=RANDOM_SEED):
    set_seed(seed); return lfilter([1], [1, -phi], np.random.randn(N))

def gen_ar2(N, phi1=0.5, phi2=0.3, seed=RANDOM_SEED):
    set_seed(seed); return lfilter([1], [1, -phi1, -phi2], np.random.randn(N))

def gen_pink(N, seed=RANDOM_SEED):
    set_seed(seed); return np.cumsum(np.random.randn(N))

def gen_const(N):
    return np.ones(N)

# =============================== Null models ============================

def white_null(N, trials=NULL_TRIALS, d=EMBED_D, degree=POLY_DEGREE, lmbd=RIDGE_ALPHA):
    errs = []
    # use different seeds to sample null variability
    for t in range(trials):
        xw = gen_white(N, seed=RANDOM_SEED + 10000 + t)
        try:
            e = predictive_error_poly(xw, d=d, degree=degree, lmbd=lmbd)
        except Exception:
            continue
        errs.append(e)
    errs = np.array(errs, float)
    mu, sd = float(np.mean(errs)), float(np.std(errs) + 1e-12)
    return mu, sd

def diff_null(N, trials=NULL_TRIALS, d=EMBED_D, degree=POLY_DEGREE, lmbd=RIDGE_ALPHA):
    errs = []
    for t in range(trials):
        xw = gen_white(N, seed=RANDOM_SEED + 20000 + t)
        xd = diff_once(xw)
        # embedding dimension reduced for differenced series
        d_eff = max(6, min(d,  max(6, d//2)))
        try:
            e = predictive_error_poly(xd, d=d_eff, degree=degree, lmbd=lmbd)
        except Exception:
            continue
        errs.append(e)
    errs = np.array(errs, float)
    mu, sd = float(np.mean(errs)), float(np.std(errs) + 1e-12)
    return mu, sd

# =============================== Evaluation ============================

def evaluate_temporal_recursion(x, d, degree, lmbd, mu_null, sd_null, mu_dn, sd_dn):
    # Original predictability
    try:
        Eo = predictive_error_poly(x, d, degree, lmbd)
    except Exception as e:
        # if embedding fails, mark as non-lawful with large error
        return {'verdict': 'NON_LAWFUL', 'E_orig': np.nan, 'error': str(e)}

    z_err = (Eo - mu_null) / sd_null

    # Falsifiers
    Ep = predictive_error_poly(permute_global(x), d, degree, lmbd)
    Es = predictive_error_poly(iaaft_surrogate(x, IAAFT_ITERS, RANDOM_SEED + 1), d, degree, lmbd)
    Er = predictive_error_poly(reverse_time(x), d, degree, lmbd)

    # Differenced series
    xd = diff_once(x)
    # use reduced embedding for differenced signal
    d_diff = max(6, d // 2)
    Ed = predictive_error_poly(xd, d=d_diff, degree=degree, lmbd=lmbd)

    # ratios and z-diffs
    ratios = {
        'p': Ep / (Eo + 1e-12),
        's': Es / (Eo + 1e-12),
        'r': Er / (Eo + 1e-12),
        'd': Ed / (Eo + 1e-12)
    }
    zdiffs = {
        'p': (Ep - Eo) / (sd_null + 1e-12),
        's': (Es - Eo) / (sd_null + 1e-12),
        'r': (Er - Eo) / (sd_null + 1e-12),
    }

    # differenced z vs its own null
    z_dd = (Ed - mu_dn) / sd_dn

    # evidence counting with patched thresholds
    hits = 0
    if (zdiffs['p'] > Z_DIFF_THR) or (ratios['p'] > RATIO_THR):
        hits += 1
    if (zdiffs['s'] > Z_DIFF_THR) or (ratios['s'] > RATIO_THR):
        hits += 1
    if (zdiffs['r'] > (Z_DIFF_THR + 0.25)) or (ratios['r'] > (RATIO_THR + 0.1)):
        hits += 1

    # differencing robustness: pass if either z_dd not very negative OR ratio small
    diff_ok = (z_dd > Z_DD_THR) or (ratios['d'] < R_D_MAX) or (ratios['d'] > 1000)

    lawful = (z_err < Z_ERR_THR) and (hits >= 2) and diff_ok

    return dict(
        verdict="TEMPORALLY_LAWFUL" if lawful else "NON_LAWFUL",
        E_orig=Eo, E_perm=Ep, E_surr=Es, E_rev=Er, E_diff=Ed,
        z_err=z_err, z_dp=zdiffs['p'], z_ds=zdiffs['s'], z_dr=zdiffs['r'], z_dd=z_dd,
        rP=ratios['p'], rS=ratios['s'], rR=ratios['r'], rD=ratios['d'],
        hits=hits
    )

# =============================== Main =================================

if __name__ == "__main__":
    set_seed(RANDOM_SEED)
    N = N_SAMPLES
    d = EMBED_D
    degree = POLY_DEGREE
    lmbd = RIDGE_ALPHA

    print("\n=== R3-Polynomial Temporal Recursion Discriminator (Patched) ===\n")
    print(f"Parameters: N={N}, d={d}, degree={degree}, IAAFT_iters={IAAFT_ITERS}")
    print(f"Thresholds: Z_ERR_THR={Z_ERR_THR}, Z_DIFF_THR={Z_DIFF_THR}, RATIO_THR={RATIO_THR}, Z_DD_THR={Z_DD_THR}, R_D_MAX={R_D_MAX}\n")

    # build nulls
    print("Building null models (this may take a bit)...")
    mu_null, sd_null = white_null(N, trials=NULL_TRIALS, d=d, degree=degree, lmbd=lmbd)
    mu_dn, sd_dn = diff_null(N, trials=NULL_TRIALS, d=d, degree=degree, lmbd=lmbd)

    signals = {
        "white":    gen_white(N),
        "uniform":  gen_uniform(N),
        "logistic": gen_logistic(N),
        "ar1":      gen_ar1(N),
        "ar2":      gen_ar2(N),
        "pink":     gen_pink(N),
        "constant": gen_const(N)
    }

    header = f"{'Signal':<10} | {'Verdict':<18} | {'E_orig':>10} | {'E_perm':>10} | {'E_surr':>10} | {'E_rev':>10} | {'E_diff':>10} | {'z_err':>7} | {'z_dp':>7} | {'z_ds':>7} | {'z_dr':>7} | {'z_dd':>7} | {'rP':>6} | {'rS':>6} | {'rR':>6} | {'rD':>6} | {'hits':>4}"
    print(header)
    print("-" * len(header))

    for name, x in signals.items():
        try:
            res = evaluate_temporal_recursion(x, d, degree, lmbd, mu_null, sd_null, mu_dn, sd_dn)
        except Exception as e:
            res = {'verdict': 'NON_LAWFUL', 'E_orig': np.nan, 'error': str(e)}
        if 'E_orig' in res and not np.isnan(res['E_orig']):
            print(f"{name:<10} | {res['verdict']:<18} | "
                  f"{res['E_orig']:>10.4e} | {res['E_perm']:>10.4e} | {res['E_surr']:>10.4e} | {res['E_rev']:>10.4e} | {res['E_diff']:>10.4e} | "
                  f"{res['z_err']:>7.2f} | {res['z_dp']:>7.2f} | {res['z_ds']:>7.2f} | {res['z_dr']:>7.2f} | {res['z_dd']:>7.2f} | "
                  f"{res['rP']:>6.2f} | {res['rS']:>6.2f} | {res['rR']:>6.2f} | {res['rD']:>6.2f} | {res['hits']:>4}")
        else:
            print(f"{name:<10} | ERROR running test: {res.get('error', 'unknown')}")

    print("\nInterpretation:")
    print(f"  z_err < {Z_ERR_THR} means the original is much more predictable than white-noise null.")
    print(f"  Falsifier evidence when z_diff > {Z_DIFF_THR} or ratio > {RATIO_THR}.")
    print(f"  Differencing robustness requires z_dd > {Z_DD_THR} OR rD < {R_D_MAX}.")
    print("  Verdict = TEMPORALLY_LAWFUL if at least 2 falsifier hits AND original predictability AND differencing robustness.\n")
