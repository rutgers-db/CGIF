# Reproducing Fig. 8: effect of predicate–embedding correlation

This folder holds everything needed to regenerate the Deep-1M workload behind
Fig. 8 and to rerun **CGIF** and **PostFilter-HNSW** on it.

> **Correction to the paper.** Section 5.3.5 says the predicate set is "the
> φn nearest neighbours of an anchor" (anchor = query / random point / farthest
> point). **That is not how Fig. 8 was produced.** The figure uses the
> rank-weighted sampling described below. The anchor construction is much
> more extreme: under P-Corr it keeps the query's true top-10 inside the
> predicate set. It gives very different curves, for example PostFilter-HNSW
> recall ≈ 0.99 at ef = 100 instead of 0.63.

## Workload

For each query `q` and each selectivity level `r` (the 28 levels 1,000 …
1,000,000 out of N = 1M), rank all base vectors by L2 distance to `q`
(rank 0 = nearest). Then draw exactly `r` of them **without replacement**,
with probability proportional to

| regime | weight `w(rank)` |
|---|---|
| P-Corr (`positive`) | `N − rank` |
| Neutral (`neutral`) | `1` (uniform) |
| N-Corr (`negative`) | `rank + 1` |

This is exponent α = 1 in `gen_predicates.py`. The sampler uses exponential
keys (`log(U)/w`, keep the `r` largest), and regime *i* uses
`numpy.random.default_rng(42 + i)`. The bias is soft. Under P-Corr the
qualifying set is enriched near `q` but usually does **not** contain `q`'s
unfiltered top-10, especially at low selectivity. At 0.1%, PostFilter-HNSW
recall is about 0.03.

* **Queries:** the first **100** Deep queries (not 1,000), so each regime has
  100 × 28 = 2,800 (query, selectivity) rows.
* **One point in Fig. 8** = one search setting (ef for the graph methods,
  nprobe for Faiss-IVF). Recall@10 is the **unweighted mean over all 2,800
  rows**. QPS = 2,800 / total search time, single-threaded on one pinned
  core. For CGIF, QPS is the trimmed mean of 3 runs.
* **Sweeps:** CGIF ef ∈ {10, 20, …, 80, 100, 120, 150}; PostFilter-HNSW
  ef ∈ {100, 200, 300, 400, 500}.

## Index parameters (Deep)

| | value |
|---|---|
| IVF clusters (`nlist`) | **4096** (the 1024 in the top-level README example is the Wiki setting) |
| Graph degree | hnswlib `M = 32`, so the base-layer max degree is `2M = 64` |
| `efConstruction` | 400 |
| Pruning α | **1.2**, set with `cgif_search -alpha 1.2` (Deep only; other datasets use 1.0) |

CGIF and PostFilter-HNSW search **the same graph file**.

## PostFilter-HNSW

`apps/postfilter_hnsw.cc` runs an ordinary unfiltered HNSW search with beam
width ef, which returns the top-ef candidates. It removes the candidates that
fail the predicate and returns the 10 closest survivors. It does **not**
re-search or enlarge ef when fewer than 10 survive.

## Running

Build the repository as described in the top-level README; this also builds
`build/postfilter_hnsw`. Then run:

```bash
BASE=/path/to/deep_base_1M.fvecs \
QUERY=/path/to/deep_query.fvecs \
WORK=/path/to/workdir \
reproduce/fig8/run_fig8.sh
```

The script does five steps, and each one is skipped if its output already
exists:

1. `gen_predicates.py` writes `correlated-{positive,neutral,negative}-top10-deep-1000000.{gt,bitsets}`
   (≈ 10 min, ≈ 1.1 GB on disk).
2. Builds the Faiss IVF index with nlist = 4096. Pass `IVF=` to reuse one.
3. Builds the CGIF graph with M = 32, efC = 400, α = 1.2. Pass `GRAPH=` to reuse one.
4. Runs CGIF and PostFilter-HNSW on each regime, pinned to core `CORE` (default 0).
5. Runs `compare.py`, which prints recall and QPS next to the paper's numbers
   (`expected_fig8.csv`) and writes overlay plots to `$WORK/plots/`.

`expected_fig8.csv` also contains the paper's numbers for the other four
methods in Fig. 8 (ACORN, Faiss-IVF, InlineFilter-HNSW, NaVix). Those
baselines are not part of this repository.

## Base vectors

Fig. 8 was run on a fixed 1M-vector sample of Deep (96-d) and the first 100
vectors of the standard 10,000-vector Deep query set. With the same input
files, `gen_predicates.py` reproduces the original `.gt` / `.bitsets` files
byte for byte:

| file | md5 |
|---|---|
| base, 1M × 96 `.fvecs` | `b1e31ceaf74031ba0e56a7e2237e9455` |
| query, 10,000 × 96 `.fvecs` | `d3d6f21c28bb09ba177829451f5f4cd7` |
| `correlated-positive-top10-deep-1000000.gt` | `d287c51097307e312017f344532a1ed3` |
| `correlated-neutral-top10-deep-1000000.gt` | `5864ad7a52ee0a889dc97ce9b5e3ce9e` |
| `correlated-negative-top10-deep-1000000.gt` | `f9f0dbd571267eb86ad18ed745cf7199` |

With a different 1M subset of Deep, the workload follows the same
distribution but individual numbers will not match exactly.

## Our verification run

We ran this folder on the machine used for the paper (Intel Xeon Gold 6212U).

* **Predicates:** the regenerated files for all three regimes are
  byte-identical to the ones used for Fig. 8.
* **Same graph and IVF index as the paper:** recall@10 from `cgif_search`
  and `postfilter_hnsw` matches every point in Fig. 8 exactly (difference
  0.0000 on all 48 points).
* **Indexes rebuilt from scratch by `run_fig8.sh`:**
  * The rebuilt IVF index gives exactly the paper's recall.
  * Graph construction is multi-threaded and therefore not deterministic.
  * PostFilter-HNSW is within ±0.0004 of the paper.
  * CGIF P-Corr / Neutral is within +0.003.
  * CGIF N-Corr is 0.01–0.03 higher than the paper, depending on the build.
    N-Corr is the regime most sensitive to the particular graph.
* **QPS:** depends on the CPU and on other load on the machine. On the
  paper's machine, unloaded runs were 1.1–1.35× the published QPS. Compare
  the shape of the curves and the relative order of the methods rather than
  absolute QPS.

For reference, CGIF recall@10 over ef = 10 → 150 in Fig. 8 is
0.897 → 0.994 (P-Corr), 0.900 → 0.988 (Neutral), and 0.632 → 0.793 (N-Corr).
PostFilter-HNSW at ef = 100 is 0.626 / 0.520 / 0.036.
