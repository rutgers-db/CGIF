#!/usr/bin/env python3
"""Generate the P-Corr / Neutral / N-Corr predicate workloads used in Fig. 8.

For each query q and each selectivity level r (28 levels, 1,000 .. 1,000,000
qualifying points out of N = 1M), we rank all base vectors by squared L2
distance to q (rank 0 = nearest) and draw exactly r of them WITHOUT
replacement, with a rank-dependent weight:

    positive (P-Corr):  w(rank) = (N - rank) ** alpha   near points favoured
    neutral:            w(rank) = 1                     uniform random
    negative (N-Corr):  w(rank) = (rank + 1) ** alpha   far points favoured

Fig. 8 uses alpha = 1 and the first 100 queries. The bias is deliberately
soft: under P-Corr the qualifying set is enriched around q but does NOT, in
general, contain q's unfiltered top-10.

Weighted sampling without replacement uses exponential keys
(Efraimidis-Spirakis): key_i = log(U_i) / w_i, keep the r largest keys.

Ground truth comes straight from the ranks. The qualifying set is
sorted_idx[sel_ranks], so the 10 nearest qualifying points are
sorted_idx[sort(sel_ranks)[:10]]. No extra distance computations are needed.

Output, for each regime in {positive, neutral, negative}:
    <out_dir>/correlated-<regime>-top10-deep-1000000.gt       (CSV)
    <out_dir>/correlated-<regime>-top10-deep-1000000.bitsets  (binary)
The layout is range-major (outer loop over the 28 levels, inner loop over
queries), which is the format cgif_search's loader reads. See the
"Data format" section of the top-level README.

Runtime is roughly 10 minutes and peak memory roughly 1 GB (100 queries).
"""

import argparse
import os
import struct
import time

import numpy as np

RANGE_LIST = [
    1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000,
    20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000,
    200000, 300000, 400000, 500000, 600000, 700000, 800000, 900000,
    1000000,
]
REGIMES = ["positive", "neutral", "negative"]
K = 10


def read_fvecs(path, n_max=None):
    raw = np.fromfile(path, dtype=np.float32)
    dim = int(raw[:1].view(np.int32)[0])
    stride = 1 + dim
    n = len(raw) // stride
    if n_max is not None:
        n = min(n, n_max)
    return raw[: n * stride].reshape(n, stride)[:, 1:].copy()


def recip_weights(n, regime, alpha):
    ranks = np.arange(n, dtype=np.float64)
    if regime == "positive":
        w = (n - ranks) ** alpha
    elif regime == "negative":
        w = (ranks + 1) ** alpha
    elif regime == "neutral":
        w = np.ones(n, dtype=np.float64)
    else:
        raise ValueError(regime)
    return 1.0 / w


def weighted_sample_ranks(n, r, recip_w, rng):
    """Sample r distinct ranks in [0, n) with probability proportional to w."""
    u = rng.random(n)
    np.clip(u, 1e-12, 1.0, out=u)
    np.log(u, out=u)
    u *= recip_w                       # log(U) / w
    return np.argpartition(u, -r)[-r:]


def fill_bitset(qualifying, buf):
    """bit = 1 -> excluded, bit = 0 -> satisfies the predicate."""
    buf[:] = b"\xff" * len(buf)
    for idx in qualifying:
        buf[idx >> 3] &= ~(1 << (idx & 7)) & 0xFF


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", required=True, help="base vectors (.fvecs), first N are used")
    ap.add_argument("--query", required=True, help="query vectors (.fvecs)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--N", type=int, default=1_000_000)
    ap.add_argument("--num_queries", type=int, default=100,
                    help="first Q queries are used (Fig. 8: 100)")
    ap.add_argument("--alpha", type=float, default=1.0, help="bias strength (Fig. 8: 1)")
    ap.add_argument("--seed", type=int, default=42,
                    help="regime i in (positive, neutral, negative) uses seed + i")
    ap.add_argument("--regimes", nargs="+", default=REGIMES, choices=REGIMES)
    args = ap.parse_args()

    N, Q = args.N, args.num_queries
    if max(RANGE_LIST) > N:
        raise SystemExit(f"--N must be >= {max(RANGE_LIST)}")
    os.makedirs(args.out_dir, exist_ok=True)
    bytes_per_bitset = (N + 7) // 8
    t0 = time.time()

    data = read_fvecs(args.base, N)
    queries = read_fvecs(args.query, Q)
    assert data.shape[0] == N, f"base has only {data.shape[0]} vectors"
    assert queries.shape[0] == Q, f"query file has only {queries.shape[0]} vectors"
    print(f"base {data.shape}  queries {queries.shape}  alpha={args.alpha}", flush=True)

    # Distance ranking of all N points for every query.
    sorted_idx = np.empty((Q, N), dtype=np.uint32)
    for q in range(Q):
        diff = data - queries[q]
        d = np.einsum("ij,ij->i", diff, diff)
        sorted_idx[q] = np.argsort(d).astype(np.uint32)
    print(f"ranked {Q} queries in {time.time() - t0:.0f}s", flush=True)

    buf = bytearray(bytes_per_bitset)
    for regime in args.regimes:
        rng = np.random.default_rng(args.seed + REGIMES.index(regime))
        rw = recip_weights(N, regime, args.alpha)
        stem = os.path.join(args.out_dir, f"correlated-{regime}-top{K}-deep-{N}")
        t1 = time.time()
        with open(stem + ".bitsets", "wb") as fbs, open(stem + ".gt", "w") as fgt:
            fbs.write(struct.pack("QQ", len(RANGE_LIST) * Q, bytes_per_bitset))
            for r in RANGE_LIST:
                for q in range(Q):
                    sel = weighted_sample_ranks(N, r, rw, rng)
                    gt = sorted_idx[q][np.sort(sel)[:K]]
                    fill_bitset(sorted_idx[q][sel].tolist(), buf)
                    fbs.write(buf)
                    fgt.write(f"{q},{r},{K},0.0," + " ".join(map(str, gt)) + " \n")
        print(f"[{regime}] {stem}.{{gt,bitsets}}  ({time.time() - t1:.0f}s)", flush=True)

    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
