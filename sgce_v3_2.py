#!/usr/bin/env python3
# SGCE v3.2 — maximum rigor engine
# (Bug-fix only: imports + safe numpy config; no logic changes)

from typing import Any, List, Dict, Tuple

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np
from numpy.fft import rfft, irfft

# ------------------ UTIL ------------------

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()

def write_json(path: str, obj: Any) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted(set().union(*[r.keys() for r in rows]))
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

def env_manifest() -> Dict[str, Any]:
    import numpy
    man = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": numpy.__version__,
        "fft_impl": "numpy.fft",
        "env_vars_subset": {k: os.environ.get(k) for k in ["PYTHONHASHSEED"] if k in os.environ},
    }
    # Try to capture BLAS/LAPACK info if available (safe subset)
    try:
        import numpy.__config__ as nc
        info = {}
        for name in ("blas_opt_info", "openblas_info", "lapack_opt_info"):
            try:
                info[name] = nc.get_info(name)
            except Exception:
                info[name] = {}
        man["numpy_config"] = info
    except Exception as e:
        man["numpy_config"] = {"error": str(e)}
    return man

# ------------------ I/O ------------------

def load_bits_from_bin(path: str, msb_first: bool = True) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    bits = np.unpackbits(data, bitorder='big' if msb_first else 'little')
    return bits.astype(np.uint8)

def load_bits_from_csv(path: str, column: int = 0, header: bool = False, binary: bool = True) -> np.ndarray:
    arr = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i == 0 and header:
                continue
            parts = line.strip().split(',')
            if not parts or column >= len(parts):
                continue
            val = parts[column].strip()
            if binary:
                if val in ('0', '1'):
                    arr.append(int(val))
            else:
                try:
                    arr.append(float(val))
                except Exception:
                    continue
    a = np.array(arr)
    if binary:
        return a.astype(np.uint8)
    med = np.median(a)
    bits = (a > med).astype(np.uint8)
    return bits

# ------------------ CORE HELPERS ------------------

def late_segment(bits: np.ndarray, N: int) -> np.ndarray:
    if N <= 0 or N > len(bits):
        N = len(bits)
    return bits[-N:].copy()

def split_windows(x: np.ndarray, W: int) -> List[np.ndarray]:
    n = (len(x) // W) * W
    x = x[:n]
    return np.array_split(x, W)

def rotate_tail(x: np.ndarray, k: int) -> np.ndarray:
    if len(x) == 0:
        return x.copy()
    k = k % len(x)
    if k == 0:
        return x.copy()
    return np.concatenate([x[-k:], x[:-k]])

def block_entropy(bits: np.ndarray, k: int) -> float:
    if len(bits) < k:
        return 0.0
    m = len(bits) - k + 1
    if m <= 0:
        return 0.0
    view = np.lib.stride_tricks.sliding_window_view(bits, k)
    weights = (1 << np.arange(k-1, -1, -1)).astype(np.uint64)
    codes = (view * weights).sum(axis=1)
    counts = np.bincount(codes, minlength=(1<<k)).astype(np.float64)
    tot = counts.sum()
    if tot <= 0:
        return 0.0
    p = counts / tot
    nz = p[p > 0]
    H = -(nz * np.log2(nz)).sum()
    return float(H)

def spectral_flatness(x_pm1: np.ndarray) -> float:
    X = rfft(x_pm1.astype(float))
    P = np.abs(X)**2
    if len(P) > 1:
        P = P[1:]
    P = np.clip(P, 1e-15, None)
    gmean = np.exp(np.mean(np.log(P)))
    amean = np.mean(P)
    return float(gmean/amean)

def run_lengths(bits: np.ndarray) -> np.ndarray:
    if len(bits) == 0:
        return np.array([], dtype=int)
    diffs = np.diff(bits)
    idx = np.where(diffs != 0)[0] + 1
    segments = np.split(bits, idx)
    return np.array([len(seg) for seg in segments], dtype=int)

def w1_geometric(emp_runs: np.ndarray, p: float, max_len: int = 256) -> float:
    if len(emp_runs) == 0:
        return 0.0
    L = min(max_len, emp_runs.max() if emp_runs.size else 1)
    hist = np.bincount(emp_runs, minlength=L+1).astype(float)[1:L+1]
    emp_pmf = hist / hist.sum() if hist.sum() > 0 else np.ones(L)/L
    ks = np.arange(1, L+1)
    geo = (p * (1-p)**(ks-1))
    geo = geo / geo.sum()
    emp_cdf = np.cumsum(emp_pmf)
    geo_cdf = np.cumsum(geo)
    w1 = np.abs(emp_cdf - geo_cdf).sum() / L
    return float(w1)

def observers_for_window(bits: np.ndarray, kmax: int = 8, Lmax: int = 256) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    n1 = bits.sum()
    n = len(bits)
    p1 = n1 / max(1, n)
    out['p0'] = 1 - p1
    for k in range(1, kmax+1):
        out[f'H{k}'] = block_entropy(bits, k)
    x = 2*bits.astype(float) - 1.0
    out['sf'] = spectral_flatness(x)
    runs = run_lengths(bits)
    mu = runs.mean() if runs.size else 0.0
    phat = 1.0/mu if mu > 1e-9 else 0.5
    phat = float(np.clip(phat, 1e-6, 1-1e-6))
    out['w1_geo_05'] = w1_geometric(runs, 0.5, Lmax)
    out['w1_geo_phat'] = w1_geometric(runs, phat, Lmax)
    out['phat_runs'] = phat
    return out

def deltas_between(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, float]:
    d: Dict[str, float] = {}
    d['tv_bits'] = abs(a['p0'] - b['p0'])
    for k in [k for k in a.keys() if k.startswith('H')]:
        d[f'delta_{k}'] = abs(a[k] - b[k])
    d['delta_sf'] = abs(a['sf'] - b['sf'])
    d['delta_w1_geo_05'] = abs(a['w1_geo_05'] - b['w1_geo_05'])
    d['delta_w1_geo_phat'] = abs(a['w1_geo_phat'] - b['w1_geo_phat'])
    return d

def all_pairs_deltas(windows: List[np.ndarray], kmax: int, Lmax: int) -> List[Dict[str, float]]:
    obs = [observers_for_window(w, kmax=kmax, Lmax=Lmax) for w in windows]
    pairs = []
    for i in range(len(obs)-1):
        pairs.append(deltas_between(obs[i], obs[i+1]))
    return pairs

def passes_thresholds(pair_delta: Dict[str, float], taus: Dict[str, float]) -> bool:
    for metric, val in pair_delta.items():
        if metric == 'tv_bits':
            key = 'tau_bits'
        elif metric.startswith('delta_H'):
            key = 'tau_H'
        elif metric == 'delta_sf':
            key = 'tau_sf'
        elif metric.startswith('delta_w1'):
            key = 'tau_w1'
        else:
            continue
        if key in taus and val >= taus[key]:
            return False
    return True

def verdict_for_windows(pairs: List[Dict[str, float]], taus: Dict[str, float]) -> bool:
    return all(passes_thresholds(p, taus) for p in pairs)

def stability_pass(bits: np.ndarray, N: int, W_list: List[int], kmax: int, Lmax: int, taus: Dict[str, float]) -> Tuple[bool, Dict[str, Any], float]:
    late = late_segment(bits, N)
    per_scale = []
    all_pass = True
    margins = []
    for W in W_list:
        wins = split_windows(late, W)
        deltas = all_pairs_deltas(wins, kmax, Lmax)
        passed = verdict_for_windows(deltas, taus)
        per_scale.append({"W": W, "passed": passed, "pairs": deltas})
        if not passed:
            all_pass = False
        for pair in deltas:
            for metric, val in pair.items():
                if metric == 'tv_bits':
                    key = 'tau_bits'
                elif metric.startswith('delta_H'):
                    key = 'tau_H'
                elif metric == 'delta_sf':
                    key = 'tau_sf'
                elif metric.startswith('delta_w1'):
                    key = 'tau_w1'
                else:
                    continue
                tau = taus.get(key, 1.0)
                margins.append(val / max(1e-12, tau) - 1.0)
    stab_score = float(max(margins) if margins else 0.0)
    return all_pass, {"per_scale": per_scale}, stab_score

def permutation_collapse(bits: np.ndarray, N: int, W_list: List[int],
                         kmax: int, Lmax: int, taus: Dict[str, float],
                         R: int, seed: int, orig_score: float) -> Dict[str, Any]:
    rng = np.random.default_rng(seed if seed else None)
    late = late_segment(bits, N)
    stable_passes = 0
    scores = []
    for _ in range(R):
        perm = late.copy()
        rng.shuffle(perm)
        all_scales_pass = True
        margins = []
        for W in W_list:
            wins = split_windows(perm, W)
            deltas = all_pairs_deltas(wins, kmax, Lmax)
            if not verdict_for_windows(deltas, taus):
                all_scales_pass = False
            for pair in deltas:
                for metric, val in pair.items():
                    if metric == 'tv_bits':
                        key = 'tau_bits'
                    elif metric.startswith('delta_H'):
                        key = 'tau_H'
                    elif metric == 'delta_sf':
                        key = 'tau_sf'
                    elif metric.startswith('delta_w1'):
                        key = 'tau_w1'
                    else:
                        continue
                    tau = taus.get(key, 1.0)
                    margins.append(val / max(1e-12, tau) - 1.0)
        score = float(max(margins) if margins else 0.0)
        scores.append(score)
        if all_scales_pass:
            stable_passes += 1
    collapse_rate = 1.0 - stable_passes / max(1, R)
    pval = sum(1 for s in scores if s <= orig_score) / max(1, R)
    return {"R": R, "passes": stable_passes, "collapse_rate": collapse_rate, "perm_pvalue": pval}

def iaaft_surrogate(x: np.ndarray, iters: int, rng) -> np.ndarray:
    x = np.asarray(x, float)
    n = len(x)
    Xamp = np.abs(rfft(x))
    xs = np.sort(x)
    y = rng.permutation(x)
    for _ in range(iters):
        Y = rfft(y)
        Y = Xamp * np.exp(1j * np.angle(Y))
        y = irfft(Y, n=n)
        ranks = np.argsort(np.argsort(y))
        y = xs[ranks]
    return y

def surrogate_false_positive(bits: np.ndarray, N: int, W_list: List[int],
                             kmax: int, Lmax: int, taus: Dict[str, float],
                             S: int, seed: int = 0, iters: int = 50) -> Dict[str, Any]:
    rng = np.random.default_rng(seed if seed else None)
    late = late_segment(bits, N)
    s = 2*late.astype(float) - 1.0
    stable_count = 0
    for _ in range(S):
        y = iaaft_surrogate(s, iters=iters, rng=rng)
        b = (y > 0).astype(np.uint8)
        all_scales_pass = True
        for W in W_list:
            wins = split_windows(b, W)
            deltas = all_pairs_deltas(wins, kmax, Lmax)
            if not verdict_for_windows(deltas, taus):
                all_scales_pass = False
        if all_scales_pass:
            stable_count += 1
    fp_rate = stable_count / max(1, S)
    return {"S": S, "stable_count": stable_count, "fp_rate": fp_rate}

def sampling_reparameterization(bits: np.ndarray, N: int, W_list: List[int],
                                kmax: int, Lmax: int, taus: Dict[str, float]) -> Dict[str, Any]:
    late = late_segment(bits, N)
    runs = run_lengths(late)
    vals = []
    if len(late) > 0:
        cur = late[0]
        for r in runs:
            vals.append(int(cur))
            cur = 1 - cur
    if not vals:
        return {"matched": True, "note": "degenerate constant stream"}
    m = len(vals)
    reps = [N // m] * m
    for i in range(N % m):
        reps[i] += 1
    ev = np.concatenate([np.full(r, v, dtype=np.uint8) for v, r in zip(vals, reps)])
    same_class = True
    for W in W_list:
        winsU = split_windows(late, W)
        winsE = split_windows(ev, W)
        delU = all_pairs_deltas(winsU, kmax, Lmax)
        delE = all_pairs_deltas(winsE, kmax, Lmax)
        if verdict_for_windows(delU, taus) != verdict_for_windows(delE, taus):
            same_class = False
    return {"matched": same_class}

def jitter_robustness(bits: np.ndarray, N: int, W_list: List[int],
                      kmax: int, Lmax: int, taus: Dict[str, float],
                      eps_list: List[float], seed: int = 0) -> Dict[str, Any]:
    rng = np.random.default_rng(seed if seed else None)
    late = late_segment(bits, N)
    curve = []
    for eps in eps_list:
        flips = rng.random(len(late)) < eps
        b = late.copy()
        b[flips] ^= 1
        ok = True
        for W in W_list:
            wins = split_windows(b, W)
            deltas = all_pairs_deltas(wins, kmax, Lmax)
            if not verdict_for_windows(deltas, taus):
                ok = False
        curve.append({"eps": eps, "stable": ok})
    return {"curve": curve}

def endianness_check(path: str, N: int, W_list: List[int], kmax: int, Lmax: int, taus: Dict[str, float]) -> Dict[str, Any]:
    try:
        bits_m = load_bits_from_bin(path, msb_first=True)
        bits_l = load_bits_from_bin(path, msb_first=False)
    except Exception as e:
        return {"error": f"endianness check only valid for --bin files: {e}"}
    stab_m, _, _ = stability_pass(bits_m, N, W_list, kmax, Lmax, taus)
    stab_l, _, _ = stability_pass(bits_l, N, W_list, kmax, Lmax, taus)
    return {"msb_first_stable": bool(stab_m), "lsb_first_stable": bool(stab_l)}

def tail_rotation_sensitivity(bits: np.ndarray, N: int, W_list: List[int], kmax: int, Lmax: int, taus: Dict[str, float], K: int = 8) -> Dict[str, Any]:
    late = late_segment(bits, N)
    votes = []
    for i in range(K):
        rot = rotate_tail(late, i * (len(late)//K if len(late)>=K else 1))
        ok_all = True
        for W in W_list:
            wins = split_windows(rot, W)
            deltas = all_pairs_deltas(wins, kmax, Lmax)
            if not verdict_for_windows(deltas, taus):
                ok_all = False
        votes.append(ok_all)
    return {"rotations": K, "stable_fraction": sum(votes)/max(1,K)}

def null_calibration(N: int, W_list: List[int], kmax: int, Lmax: int, taus: Dict[str, float], M: int, seed: int = 0) -> Dict[str, Any]:
    rng = np.random.default_rng(seed if seed else None)
    metrics = ["tv_bits", "delta_sf", "delta_w1_geo_05", "delta_w1_geo_phat"] + [f"delta_H{k}" for k in range(1, kmax+1)]
    dist = {m: [] for m in metrics}
    for _ in range(M):
        b = rng.integers(0, 2, size=N, dtype=np.uint8)
        for W in W_list:
            wins = split_windows(b, W)
            deltas = all_pairs_deltas(wins, kmax, Lmax)
            for pair in deltas:
                for m in metrics:
                    if m in pair:
                        dist[m].append(float(pair[m]))
    q = {m: {"q95": float(np.quantile(dist[m], 0.95)) if dist[m] else None,
             "q99": float(np.quantile(dist[m], 0.99)) if dist[m] else None,
             "mean": float(np.mean(dist[m])) if dist[m] else None,
             "n": len(dist[m])} for m in metrics}
    return {"samples": M, "summaries": q}

def bh_correction(pvals: List[float], alpha: float = 0.05) -> Tuple[float, List[bool]]:
    m = len(pvals)
    if m == 0:
        return alpha, []
    order = np.argsort(pvals)
    thresh = 0.0
    flags = [False]*m
    for i, idx in enumerate(order, start=1):
        if pvals[idx] <= alpha*i/m:
            thresh = max(thresh, alpha*i/m)
    for i, p in enumerate(pvals):
        flags[i] = p <= thresh and p <= alpha
    return thresh, flags

TEMPLATE_TAUS = {"tau_bits": 0.002, "tau_sf": 0.01, "tau_w1": 0.01, "tau_H": 0.008}

# ------------------ CLI ------------------

def main():
    ap = argparse.ArgumentParser(description="SGCE v3.2 — maximum rigor")
    ap.add_argument("--bin"); ap.add_argument("--csv")
    ap.add_argument("--csv_col", type=int, default=0)
    ap.add_argument("--csv_header", action="store_true")
    ap.add_argument("--csv_binary", action="store_true")
    ap.add_argument("--N", type=int, default=1_000_000)
    ap.add_argument("--W_list", default="2,4,8")
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--Lmax", type=int, default=256)
    ap.add_argument("--taus")
    ap.add_argument("--perm_R", type=int, default=0)
    ap.add_argument("--surr_S", type=int, default=0)
    ap.add_argument("--rng_seed", type=int, default=0)
    ap.add_argument("--jitter", default="0.0,0.001,0.005,0.01")
    ap.add_argument("--endianness_check", action="store_true")
    ap.add_argument("--tail_rotations", type=int, default=0)
    ap.add_argument("--null_calibrate", type=int, default=0, help="M null samples; 0 to skip")
    ap.add_argument("--out_json", default="/mnt/data/sgce_v3_2_results.json")
    ap.add_argument("--out_csv",  default="/mnt/data/sgce_v3_2_results.csv")
    ap.add_argument("--env_out",  default="/mnt/data/sgce_env_manifest.json")
    args = ap.parse_args([]) if "ipykernel" in sys.modules else ap.parse_args()

    # Load data
    if not args.bin and not args.csv:
        print("Provide --bin or --csv", file=sys.stderr); sys.exit(2)
    if args.bin:
        bits = load_bits_from_bin(args.bin, msb_first=True)
        src = args.bin
        endianness_path = args.bin
    else:
        bits = load_bits_from_csv(args.csv, column=args.csv_col, header=args.csv_header, binary=args.csv_binary)
        src = args.csv
        endianness_path = None

    W_list = [int(x) for x in args.W_list.split(",") if x.strip()]
    taus = TEMPLATE_TAUS.copy()
    if args.taus:
        with open(args.taus, 'r', encoding='utf-8') as f:
            taus.update(json.load(f))

    N = min(args.N, len(bits))

    # Environment manifest
    man = env_manifest()
    write_json(args.env_out, man)

    t0 = time.time()
    stable, stability_detail, stab_score = stability_pass(bits, N, W_list, args.kmax, args.Lmax, taus)
    res: Dict[str, Any] = {
        "engine_version": "3.2",
        "source_path": src,
        "source_sha256": sha256_file(src),
        "N": N,
        "W_list": W_list,
        "kmax": args.kmax,
        "Lmax": args.Lmax,
        "taus": taus,
        "env_manifest_path": args.env_out,
        "stable_original": stable,
        "stability_details": stability_detail,
        "stability_score": stab_score,
        "falsifiers": {},
        "robustness": {},
        "null_calibration": None,
        "runtime_sec": None
    }

    # Falsifiers
    if args.perm_R > 0:
        res["falsifiers"]["permutation"] = permutation_collapse(
            bits, N, W_list, args.kmax, args.Lmax, taus,
            R=args.perm_R, seed=args.rng_seed, orig_score=stab_score
        )
    if args.surr_S > 0:
        res["falsifiers"]["surrogate"] = surrogate_false_positive(
            bits, N, W_list, args.kmax, args.Lmax, taus,
            S=args.surr_S, seed=args.rng_seed, iters=50
        )

    # Robustness
    eps_list = [float(x) for x in args.jitter.split(",") if x.strip()]
    res["robustness"]["jitter"] = jitter_robustness(bits, N, W_list, args.kmax, args.Lmax, taus, eps_list, seed=args.rng_seed)
    res["robustness"]["sampling_reparameterization"] = sampling_reparameterization(bits, N, W_list, args.kmax, args.Lmax, taus)
    if args.tail_rotations > 0:
        res["robustness"]["tail_rotation"] = tail_rotation_sensitivity(bits, N, W_list, args.kmax, args.Lmax, taus, K=args.tail_rotations)
    if args.endianness_check and endianness_path:
        res["robustness"]["endianness"] = endianness_check(endianness_path, N, W_list, args.kmax, args.Lmax, taus)

    # Null calibration
    if args.null_calibrate > 0:
        res["null_calibration"] = null_calibration(N, W_list, args.kmax, args.Lmax, taus, M=args.null_calibrate, seed=args.rng_seed)

    res["runtime_sec"] = round(time.time() - t0, 3)

    # Flat CSV row
    row = {
        "engine_version": res["engine_version"],
        "source_path": res["source_path"],
        "source_sha256": res["source_sha256"],
        "N": res["N"],
        "W_list": "|".join(map(str, res["W_list"])),
        "kmax": res["kmax"],
        "Lmax": res["Lmax"],
        "tau_bits": taus.get("tau_bits"),
        "tau_sf": taus.get("tau_sf"),
        "tau_w1": taus.get("tau_w1"),
        "tau_H": taus.get("tau_H"),
        "stable_original": res["stable_original"],
        "stab_score": res["stability_score"],
        "perm_R": args.perm_R,
        "surr_S": args.surr_S,
        "perm_collapse_rate": res.get("falsifiers", {}).get("permutation", {}).get("collapse_rate"),
        "perm_pvalue": res.get("falsifiers", {}).get("permutation", {}).get("perm_pvalue"),
        "surr_fp_rate": res.get("falsifiers", {}).get("surrogate", {}).get("fp_rate"),
        "jitter_curve": ";".join([f"{pt['eps']}:{int(pt['stable'])}" for pt in res["robustness"]["jitter"]["curve"]]) if "jitter" in res["robustness"] else "",
        "sampling_matched": res["robustness"]["sampling_reparameterization"].get("matched"),
        "tail_rotation_stable_frac": res["robustness"].get("tail_rotation", {}).get("stable_fraction"),
        "endianness_msb_stable": res["robustness"].get("endianness", {}).get("msb_first_stable") if "endianness" in res["robustness"] else "",
        "endianness_lsb_stable": res["robustness"].get("endianness", {}).get("lsb_first_stable") if "endianness" in res["robustness"] else "",
        "null_M": res["null_calibration"]["samples"] if res["null_calibration"] else "",
        "runtime_sec": res["runtime_sec"],
    }

    write_json(args.out_json, res)
    write_csv(args.out_csv, [row])
    print(f"Wrote JSON: {args.out_json}")
    print(f"Wrote CSV : {args.out_csv}")
    print(f"Wrote ENV  : {args.env_out}")

if __name__ == "__main__":
    main()
