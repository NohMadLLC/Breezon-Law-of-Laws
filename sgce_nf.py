#!/usr/bin/env python3
# SGCE_NF — threshold-free invariance engine
# Stability is measured by raw deltas; no tau cutoffs are used.
# Outputs: stability score S (lower = more stable), permutation collapse stats,
# surrogate false-positive rate, robustness panels, and environment manifest.

import argparse, csv, json, os, sys, time, hashlib, platform
from typing import Any, Dict, List, Tuple
import numpy as np
from numpy.fft import rfft, irfft

# ---------------------- I/O and environment ----------------------

def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def write_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fields = sorted({k for r in rows for k in r.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})

def env_manifest() -> Dict[str, Any]:
    import numpy
    man = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "numpy_version": numpy.__version__,
        "fft_impl": "numpy.fft",
        "env_vars_subset": {k: os.environ.get(k) for k in ["PYTHONHASHSEED"] if k in os.environ},
    }
    try:
        import numpy.__config__ as nc
        man["numpy_config"] = {k: v for k, v in nc.get_info().items()}
    except Exception as e:
        man["numpy_config"] = {"error": str(e)}
    return man

# ---------------------- loaders ----------------------

def load_bits_from_bin(path: str, msb_first: bool = True) -> np.ndarray:
    data = np.fromfile(path, dtype=np.uint8)
    bits = np.unpackbits(data, bitorder="big" if msb_first else "little")
    return bits.astype(np.uint8)

def load_bits_from_csv(path: str, column: int = 0, header: bool = False, binary: bool = True) -> np.ndarray:
    arr = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0 and header:
                continue
            parts = line.strip().split(",")
            if not parts or column >= len(parts):
                continue
            val = parts[column].strip()
            if binary:
                if val in ("0", "1"):
                    arr.append(int(val))
            else:
                try:
                    arr.append(float(val))
                except:
                    continue
    a = np.array(arr)
    if binary:
        return a.astype(np.uint8)
    med = np.median(a)
    bits = (a > med).astype(np.uint8)
    return bits

# ---------------------- helpers ----------------------

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
    view = np.lib.stride_tricks.sliding_window_view(bits, k)
    weights = (1 << np.arange(k - 1, -1, -1)).astype(np.uint64)
    codes = (view * weights).sum(axis=1)
    counts = np.bincount(codes, minlength=(1 << k)).astype(np.float64)
    tot = counts.sum()
    if tot <= 0:
        return 0.0
    p = counts / tot
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum())

def spectral_flatness(x_pm1: np.ndarray) -> float:
    X = rfft(x_pm1.astype(float))
    P = np.abs(X) ** 2
    if len(P) > 1:
        P = P[1:]  # drop DC
    P = np.clip(P, 1e-15, None)
    g = np.exp(np.mean(np.log(P)))
    a = np.mean(P)
    return float(g / a)

def run_lengths(bits: np.ndarray) -> np.ndarray:
    if len(bits) == 0:
        return np.array([], dtype=int)
    idx = np.where(np.diff(bits) != 0)[0] + 1
    segs = np.split(bits, idx)
    return np.array([len(s) for s in segs], dtype=int)

def w1_geometric(emp_runs: np.ndarray, p: float, max_len: int = 256) -> float:
    if len(emp_runs) == 0:
        return 0.0
    L = min(max_len, int(emp_runs.max()) if emp_runs.size else 1)
    hist = np.bincount(emp_runs, minlength=L + 1).astype(float)[1 : L + 1]
    emp_pmf = hist / hist.sum() if hist.sum() > 0 else np.ones(L) / L
    ks = np.arange(1, L + 1)
    geo = p * (1 - p) ** (ks - 1)
    geo = geo / geo.sum()
    emp_cdf = np.cumsum(emp_pmf)
    geo_cdf = np.cumsum(geo)
    return float(np.abs(emp_cdf - geo_cdf).sum() / L)

# ---------------------- observers and score ----------------------

def observers_for_window(bits: np.ndarray, kmax: int = 8, Lmax: int = 256) -> Dict[str, float]:
    out: Dict[str, float] = {}
    n = len(bits)
    p1 = bits.sum() / max(1, n)
    out["p0"] = 1.0 - p1
    for k in range(1, kmax + 1):
        out[f"H{k}"] = block_entropy(bits, k)
    x = 2 * bits.astype(float) - 1.0
    out["sf"] = spectral_flatness(x)
    runs = run_lengths(bits)
    mu = runs.mean() if runs.size else 0.0
    phat = 1.0 / mu if mu > 1e-9 else 0.5
    phat = float(np.clip(phat, 1e-6, 1 - 1e-6))
    out["w1_geo_05"] = w1_geometric(runs, 0.5, Lmax)
    out["w1_geo_phat"] = w1_geometric(runs, phat, Lmax)
    out["phat_runs"] = phat
    return out

def deltas_between(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, float]:
    d: Dict[str, float] = {}
    d["tv_bits"] = abs(a["p0"] - b["p0"])
    for k, v in a.items():
        if k.startswith("H"):
            d[f"delta_{k}"] = abs(v - b[k])
    d["delta_sf"] = abs(a["sf"] - b["sf"])
    d["delta_w1_geo_05"] = abs(a["w1_geo_05"] - b["w1_geo_05"])
    d["delta_w1_geo_phat"] = abs(a["w1_geo_phat"] - b["w1_geo_phat"])
    return d

def all_pairs_deltas(windows: List[np.ndarray], kmax: int, Lmax: int) -> List[Dict[str, float]]:
    obs = [observers_for_window(w, kmax=kmax, Lmax=Lmax) for w in windows]
    pairs = []
    for i in range(len(obs) - 1):
        pairs.append(deltas_between(obs[i], obs[i + 1]))
    return pairs

def stability_score(bits: np.ndarray, N: int, W_list: List[int], kmax: int, Lmax: int) -> Tuple[float, Dict[str, Any]]:
    """
    Score S = max over all adjacent-window metric deltas across all W.
    Lower S means more stable/invariant.
    """
    late = late_segment(bits, N)
    per_scale = []
    maxima = []
    for W in W_list:
        wins = split_windows(late, W)
        deltas = all_pairs_deltas(wins, kmax, Lmax)
        per_scale.append({"W": W, "pairs": deltas})
        for pair in deltas:
            maxima.append(max(pair.values()))
    S = float(max(maxima) if maxima else 0.0)
    return S, {"per_scale": per_scale}

# ---------------------- falsifiers (threshold-free) ----------------------

def permutation_panel(bits: np.ndarray, N: int, W_list: List[int], kmax: int, Lmax: int,
                      R: int, seed: int, S_orig: float) -> Dict[str, Any]:
    rng = np.random.default_rng(seed if seed else None)
    late = late_segment(bits, N)
    scores = []
    for _ in range(R):
        perm = late.copy()
        rng.shuffle(perm)
        S, _ = stability_score(perm, N, W_list, kmax, Lmax)
        scores.append(S)
    scores = np.array(scores, float)
    # collapse: permuted sequence is typically *less stable*, so S_perm > S_orig
    collapse_rate = float(np.mean(scores > S_orig)) if len(scores) else None
    # permutation p-value: probability permuted score <= original (how often perm looks as stable or more stable)
    perm_pvalue = float(np.mean(scores <= S_orig)) if len(scores) else None
    return {"R": R, "scores_summary": summary(scores), "collapse_rate": collapse_rate, "perm_pvalue": perm_pvalue}

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

def surrogate_panel(bits: np.ndarray, N: int, W_list: List[int], kmax: int, Lmax: int,
                    S: int, seed: int = 0, iters: int = 50, S_orig: float = None) -> Dict[str, Any]:
    rng = np.random.default_rng(seed if seed else None)
    late = late_segment(bits, N)
    s = 2 * late.astype(float) - 1.0
    scores = []
    for _ in range(S):
        y = iaaft_surrogate(s, iters=iters, rng=rng)
        b = (y > 0).astype(np.uint8)
        Sp, _ = stability_score(b, N, W_list, kmax, Lmax)
        scores.append(Sp)
    scores = np.array(scores, float)
    # surrogate FP: fraction of surrogates with score <= S_orig (surrogate appears as stable or more stable than original)
    fp_rate = float(np.mean(scores <= S_orig)) if (S_orig is not None and len(scores)) else None
    return {"S": S, "scores_summary": summary(scores), "surrogate_fp_rate": fp_rate}

# ---------------------- robustness and calibration ----------------------

def summary(arr: np.ndarray) -> Dict[str, Any]:
    if arr is None or len(arr) == 0:
        return {"n": 0}
    return {
        "n": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        "min": float(np.min(arr)),
        "q25": float(np.quantile(arr, 0.25)),
        "median": float(np.median(arr)),
        "q75": float(np.quantile(arr, 0.75)),
        "max": float(np.max(arr)),
    }

def sampling_reparameterization(bits: np.ndarray, N: int, W_list: List[int], kmax: int, Lmax: int, S_orig: float) -> Dict[str, Any]:
    late = late_segment(bits, N)
    runs = run_lengths(late)
    vals = []
    if len(late) > 0:
        cur = late[0]
        for r in runs:
            vals.append(int(cur))
            cur = 1 - cur
    if not vals:
        return {"matched": True, "S_event": None}
    m = len(vals)
    reps = [N // m] * m
    for i in range(N % m):
        reps[i] += 1
    ev = np.concatenate([np.full(r, v, dtype=np.uint8) for v, r in zip(vals, reps)])
    S_ev, _ = stability_score(ev, N, W_list, kmax, Lmax)
    return {"matched": bool(S_ev <= S_orig), "S_event": S_ev}

def jitter_curve(bits: np.ndarray, N: int, W_list: List[int], kmax: int, Lmax: int,
                 eps_list: List[float], seed: int, S_orig: float) -> Dict[str, Any]:
    rng = np.random.default_rng(seed if seed else None)
    late = late_segment(bits, N)
    pts = []
    for eps in eps_list:
        flips = rng.random(len(late)) < eps
        b = late.copy()
        b[flips] ^= 1
        S_eps, _ = stability_score(b, N, W_list, kmax, Lmax)
        pts.append({"eps": eps, "S": S_eps, "stable_relative": bool(S_eps <= S_orig)})
    return {"points": pts}

def null_calibration(N: int, W_list: List[int], kmax: int, Lmax: int, M: int, seed: int) -> Dict[str, Any]:
    rng = np.random.default_rng(seed if seed else None)
    scores = []
    for _ in range(M):
        b = rng.integers(0, 2, size=N, dtype=np.uint8)
        S, _ = stability_score(b, N, W_list, kmax, Lmax)
        scores.append(S)
    arr = np.array(scores, float)
    return {"samples": M, "scores_summary": summary(arr)}

# ---------------------- main ----------------------

def main():
    ap = argparse.ArgumentParser(description="SGCE_NF — threshold-free invariance engine")
    ap.add_argument("--bin")
    ap.add_argument("--csv")
    ap.add_argument("--csv_col", type=int, default=0)
    ap.add_argument("--csv_header", action="store_true")
    ap.add_argument("--csv_binary", action="store_true")
    ap.add_argument("--N", type=int, default=1_000_000)
    ap.add_argument("--W_list", default="2,4,8")
    ap.add_argument("--kmax", type=int, default=8)
    ap.add_argument("--Lmax", type=int, default=256)
    ap.add_argument("--perm_R", type=int, default=0)
    ap.add_argument("--surr_S", type=int, default=0)
    ap.add_argument("--rng_seed", type=int, default=0)
    ap.add_argument("--jitter", default="0,0.001,0.005,0.01")
    ap.add_argument("--tail_rotations", type=int, default=0)
    ap.add_argument("--null_calibrate", type=int, default=0)
    ap.add_argument("--out_json", default="sgce_nf_results.json")
    ap.add_argument("--out_csv",  default="sgce_nf_results.csv")
    ap.add_argument("--env_out",  default="sgce_nf_env.json")
    args = ap.parse_args([]) if "ipykernel" in sys.modules else ap.parse_args()

    if not args.bin and not args.csv:
        print("Provide --bin or --csv", file=sys.stderr); sys.exit(2)

    if args.bin:
        bits = load_bits_from_bin(args.bin, msb_first=True)
        src = args.bin
    else:
        bits = load_bits_from_csv(args.csv, column=args.csv_col, header=args.csv_header, binary=args.csv_binary)
        src = args.csv

    W_list = [int(x) for x in args.W_list.split(",") if x.strip()]
    N = min(args.N, len(bits))

    # environment
    write_json(args.env_out, env_manifest())

    t0 = time.time()
    S_orig, detail = stability_score(bits, N, W_list, args.kmax, args.Lmax)

    res: Dict[str, Any] = {
        "engine": "SGCE_NF",
        "source_path": src,
        "source_sha256": sha256_file(src),
        "N": N,
        "W_list": W_list,
        "kmax": args.kmax,
        "Lmax": args.Lmax,
        "stability": {"S": S_orig, "details": detail},
        "falsifiers": {},
        "robustness": {},
        "null_calibration": None,
        "runtime_sec": None,
    }

    if args.perm_R > 0:
        res["falsifiers"]["permutation"] = permutation_panel(bits, N, W_list, args.kmax, args.Lmax, args.perm_R, args.rng_seed, S_orig)

    if args.surr_S > 0:
        res["falsifiers"]["surrogates"] = surrogate_panel(bits, N, W_list, args.kmax, args.Lmax, args.surr_S, args.rng_seed, 50, S_orig)

    # robustness
    eps_list = [float(x) for x in args.jitter.split(",") if x.strip()]
    res["robustness"]["jitter"] = jitter_curve(bits, N, W_list, args.kmax, args.Lmax, eps_list, args.rng_seed, S_orig)
    # event-driven vs uniform
    res["robustness"]["sampling_reparameterization"] = sampling_reparameterization(bits, N, W_list, args.kmax, args.Lmax, S_orig)

    # tail rotations vote (stability fraction relative to S_orig)
    if args.tail_rotations > 0:
        late = late_segment(bits, N)
        votes = []
        for i in range(args.tail_rotations):
            rot = rotate_tail(late, i * (len(late) // args.tail_rotations if len(late) >= args.tail_rotations else 1))
            S_rot, _ = stability_score(rot, N, W_list, args.kmax, args.Lmax)
            votes.append(S_rot <= S_orig)
        res["robustness"]["tail_rotation"] = {"rotations": args.tail_rotations, "stable_fraction": float(np.mean(votes) if votes else 0.0)}

    if args.null_calibrate > 0:
        res["null_calibration"] = null_calibration(N, W_list, args.kmax, args.Lmax, args.null_calibrate, args.rng_seed)

    res["runtime_sec"] = round(time.time() - t0, 3)

    # flat CSV row
    row = {
        "engine": res["engine"],
        "source_path": res["source_path"],
        "source_sha256": res["source_sha256"],
        "N": res["N"],
        "W_list": "|".join(map(str, res["W_list"])),
        "kmax": res["kmax"],
        "Lmax": res["Lmax"],
        "S_orig": res["stability"]["S"],
        "perm_R": args.perm_R,
        "perm_collapse_rate": res.get("falsifiers", {}).get("permutation", {}).get("collapse_rate"),
        "perm_pvalue": res.get("falsifiers", {}).get("permutation", {}).get("perm_pvalue"),
        "surr_S": args.surr_S,
        "surr_fp_rate": res.get("falsifiers", {}).get("surrogates", {}).get("surrogate_fp_rate"),
        "jitter_points": ";".join(f"{pt['eps']}:{pt['S']:.6f}" for pt in res["robustness"]["jitter"]["points"]),
        "sampling_matched": res["robustness"]["sampling_reparameterization"]["matched"],
        "tail_rotation_stable_frac": res.get("robustness", {}).get("tail_rotation", {}).get("stable_fraction"),
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
