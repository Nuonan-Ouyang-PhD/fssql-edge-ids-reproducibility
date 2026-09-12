#!/usr/bin/env python3
"""Seed-level paired statistics for the TSUSC major revision.

Preferred input (long-by-method summary):
    regime,seed,method,utility_mean,f1,any_violations,...

Also accepts a simple wide paired file:
    seed,FSSQL-R,Safe-Greedy
with optional ``regime``. In wide mode a single metric is emitted, named by
``--metric-name`` (default: value).

Usage:
    python paired_seed_stats.py input.csv output.csv
    python paired_seed_stats.py wide.csv output.csv --metric-name utility_mean
"""
from __future__ import annotations

import argparse
import itertools
import numpy as np
import pandas as pd

DEFAULT_METRICS = [
    "utility_mean", "f1", "any_violations", "latency_violations",
    "thermal_violations", "p99_latency_ms", "p99_temp_C",
    "energy_proxy_mean", "switch_count", "fallback_rate",
]


def paired_stats(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, bootstrap_reps: int) -> dict:
    d = x - y
    n = len(d)
    boots = np.empty(bootstrap_reps, dtype=float)
    for i in range(bootstrap_reps):
        idx = rng.integers(0, n, n)
        boots[i] = float(d[idx].mean())
    lo, hi = np.quantile(boots, [0.025, 0.975])

    obs = abs(float(d.mean()))
    if n <= 20:
        exceed = 0
        total = 0
        for signs in itertools.product((-1.0, 1.0), repeat=n):
            stat = abs(float((d * np.asarray(signs)).mean()))
            exceed += stat >= obs - 1e-15
            total += 1
        p = exceed / total
    else:
        reps = 100000
        exceed = 0
        for _ in range(reps):
            signs = rng.choice((-1.0, 1.0), size=n)
            exceed += abs(float((d * signs).mean())) >= obs
        p = (1 + exceed) / (1 + reps)

    sd = float(d.std(ddof=1)) if n > 1 else float("nan")
    dz = float(d.mean() / sd) if n > 1 and sd > 0 else float("nan")
    return {
        "n_pairs": n,
        "mean_a": float(x.mean()),
        "sd_a": float(x.std(ddof=1)) if n > 1 else float("nan"),
        "mean_b": float(y.mean()),
        "sd_b": float(y.std(ddof=1)) if n > 1 else float("nan"),
        "mean_diff": float(d.mean()),
        "median_diff": float(np.median(d)),
        "sd_diff": sd,
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "exact_signflip_p": float(p),
        "cohen_dz": dz,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csv")
    ap.add_argument("output_csv")
    ap.add_argument("--method-a", default="FSSQL-R")
    ap.add_argument("--method-b", default="Safe-Greedy")
    ap.add_argument("--metric-name", default="value", help="wide-input metric label")
    ap.add_argument("--bootstrap-reps", type=int, default=20000)
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    if "seed" not in df.columns:
        raise SystemExit("input must contain a 'seed' column")
    if "regime" not in df.columns:
        df["regime"] = "unspecified"

    rng = np.random.default_rng(20260909)
    rows = []

    if "method" in df.columns:
        metrics = [m for m in DEFAULT_METRICS if m in df.columns]
        if not metrics:
            raise SystemExit("long input has no recognised numeric metric columns")
        for regime in sorted(df["regime"].dropna().astype(str).unique()):
            sub = df[df["regime"].astype(str) == regime]
            a = sub[sub["method"] == args.method_a].set_index("seed")
            b = sub[sub["method"] == args.method_b].set_index("seed")
            seeds = sorted(set(a.index) & set(b.index))
            if not seeds:
                continue
            for metric in metrics:
                x = a.loc[seeds, metric].astype(float).to_numpy()
                y = b.loc[seeds, metric].astype(float).to_numpy()
                valid = np.isfinite(x) & np.isfinite(y)
                x, y = x[valid], y[valid]
                if len(x) == 0:
                    continue
                r = paired_stats(x, y, rng, args.bootstrap_reps)
                rows.append({"regime": regime, "metric": metric, "method_a": args.method_a,
                             "method_b": args.method_b, **r})
    else:
        for col in (args.method_a, args.method_b):
            if col not in df.columns:
                raise SystemExit(f"wide input missing method column: {col}")
        for regime in sorted(df["regime"].dropna().astype(str).unique()):
            sub = df[df["regime"].astype(str) == regime].sort_values("seed")
            x = sub[args.method_a].astype(float).to_numpy()
            y = sub[args.method_b].astype(float).to_numpy()
            valid = np.isfinite(x) & np.isfinite(y)
            x, y = x[valid], y[valid]
            if len(x) == 0:
                continue
            r = paired_stats(x, y, rng, args.bootstrap_reps)
            rows.append({"regime": regime, "metric": args.metric_name, "method_a": args.method_a,
                         "method_b": args.method_b, **r})

    if not rows:
        raise SystemExit("no paired rows found for requested methods")
    out = pd.DataFrame(rows)
    out.to_csv(args.output_csv, index=False)
    print(f"wrote {len(out)} rows -> {args.output_csv}")


if __name__ == "__main__":
    main()
