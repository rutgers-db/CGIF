# CGIF

Reference implementation of our paper:

> **[CGIF: Combining Proximity Graphs and Inverted Files for Efficient Filtered
> Vector Search over Arbitrary Predicates](https://doi.org/10.14778/3836663.3836716)**

CGIF answers *filtered* approximate nearest-neighbor queries — find the `k`
nearest vectors to a query among the subset of points that satisfy an arbitrary
per-query predicate. It combines a proximity graph (an HNSW-style graph) with an
IVF (inverted-file / cluster) index: during search the graph is navigated
normally, and whenever it stalls inside the filtered region CGIF uses the IVF
clusters and a two-hop expansion to jump back to satisfying points.

---

## Repository layout

```
.
├── CMakeLists.txt
├── include/                # public headers
│   ├── ivf.h               # IVF cluster index
│   ├── bitset.h            # BitsetView: per-query predicate as a bitmask
│   ├── bm25.h
│   ├── faiss_ivf_adapter.h # wraps a trained Faiss IVF index behind IVFIndex
│   ├── base_hnsw/          # HNSW graph (fork of hnswlib, see its README)
│   └── common/             # data loading & utilities (data_utils.h, reader.h, ...)
├── src/
│   ├── ivf_connect.h       # >>> the CGIF index & search algorithm <<<
│   ├── base_index.h
│   ├── twophase_bfs.h      # two-hop expansion used by CGIF
│   └── common/             # reader.cc, utils.cc (compiled into the UTIL library)
├── apps/                   # CLI applications (program entry points)
│   ├── build_ivf_index.cc  # build the IVF index that CGIF consumes
│   ├── cgif_search.cc      # build the CGIF graph and run filtered queries
│   └── postfilter_hnsw.cc  # PostFilter-HNSW baseline (Fig. 8)
├── reproduce/
│   └── fig8/               # scripts to reproduce Fig. 8 (predicate–embedding correlation)
└── third_party/
    └── faiss/              # git submodule (used to train the IVF index)
```

The core method lives in [src/ivf_connect.h](src/ivf_connect.h) (class
`IVFConnectHNSW`).

---

## Building

### 1. Dependencies

* A C++20 compiler (GCC ≥ 10 or Clang ≥ 12)
* CMake ≥ 3.24 — the vendored Faiss requires it (CGIF itself only needs 3.9). If
  your system CMake is older, install a newer one, e.g. `pip install cmake`.
* OpenMP
* Boost (`program_options`)
* BLAS + LAPACK (any implementation; MKL/OpenBLAS both work)
* [Faiss](https://github.com/facebookresearch/faiss) — vendored as a submodule,
  pinned to the exact commit used in the paper's experiments.

### 2. Build Faiss (once)

Faiss is included as a git submodule and is used to train the IVF index. Fetch
and build it first:

```bash
git submodule update --init --recursive
cd third_party/faiss
cmake -B build . -DFAISS_ENABLE_GPU=OFF -DFAISS_ENABLE_PYTHON=OFF \
      -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --target faiss -j
cd ../..
```

This produces `third_party/faiss/build/faiss/libfaiss.a`, which the top-level
`CMakeLists.txt` links against.

### 3. Build CGIF

```bash
mkdir build && cd build
cmake .. -DFAST_MODE=ON -DNOSIMD=OFF
make -j
```

Executables are written to `build/`:

* `build/build_ivf_index`
* `build/cgif_search`
* `build/postfilter_hnsw` (PostFilter-HNSW baseline, used by `reproduce/fig8/`)

Build options:

| Option        | Default | Meaning                                                              |
|---------------|---------|---------------------------------------------------------------------|
| `FAST_MODE`   | `OFF`   | Skip the internal per-query profiling counters. Turn **ON** for benchmarking (higher QPS); leave OFF to print the detailed statistics. |
| `NOSIMD`      | `OFF`   | Disable the hand-written SIMD distance kernels in the HNSW code.     |

---

## Data format

A dataset consists of three parts: **base vectors**, **query vectors**, and a
**ground-truth + predicate** pair.

### Base / query vectors

Stored in the standard `.fvecs` format. Each vector is one `int32` dimension
count `d` followed by `d` × `float32` values:

```
[int32 d][float32 x_0]...[float32 x_{d-1}]   # repeated, one record per vector
```

(YFCC uses a `u8bin` layout instead — `[uint32 n][uint32 d]` header followed by
`n·d` `uint8` values; it is up-converted to float on load.)

### Ground truth (`.gt`, CSV)

One row per query, comma-separated:

```
query_id , filter_range , k , gt_gen_time , "gt_id_1 gt_id_2 ... gt_id_k "
```

Only `query_id` (column 0) and the space-separated ground-truth ids (column 4,
with a trailing space) are read back at query time. `filter_range` is the number
of points that pass the predicate.

### Predicate bitsets (`.bitsets`, binary)

Sits next to the `.gt` file with the same basename (the loader derives it by
replacing the `.gt` suffix with `.bitsets`). One bitmask per query:

```
[size_t rows][size_t cols]                 # rows = #queries, cols = ceil(N/8) bytes
[row_0: cols bytes][row_1: cols bytes]...
```

**Bit semantics:** bit `i` **set (1)** ⇒ point `i` is *excluded* by the
predicate; bit **clear (0)** ⇒ point `i` *satisfies* the predicate and is a
candidate result. (A bitmask of all-`0xFF` means "nothing passes"; clearing a
bit admits that point.)

The `-dataset` flag selects how these files are interpreted. Built-in datasets
used in the paper are `deep`, `wiki-image-range`, `laion`, and `yfcc`
(plus `yfcc-arbitrary` for the heterogeneous-predicate workload). Datasets such
as `laion`/`yfcc` additionally take tag files via `-data_tag_path` /
`-query_tag_path`. See
[include/common/data_utils.h](include/common/data_utils.h) for the exact
per-dataset loaders.

### Adding a new dataset

1. Add a branch for your dataset name in `ReadDataWrapper()` in
   [src/common/reader.cc](src/common/reader.cc) telling it how to read your base
   and query vectors.
2. Produce a `.gt` + `.bitsets` pair in the format above (the simplest path is
   to reuse the generic `bitset` dataset type, which reads exactly these files).

---

## Running

The workflow has three steps: build the IVF index, build the CGIF graph, then
query. The example below uses `deep` with 1M points.

### Step 1 — Build the IVF index

```bash
build/build_ivf_index \
    -dataset deep -N 1000000 \
    -dataset_path /path/to/base.fvecs \
    -query_path   /path/to/query.fvecs \
    -gt_path      /path/to/deep.gt \
    -num_clusters 1024 \
    -save_index   /path/to/deep_faiss_ivf.index
```

> **Note:** the saved index path must contain the substring `faiss` — that is
> how `cgif_search` recognizes it as a Faiss IVF index (rather than a raw
> cluster-assignment `.npy`). To skip the built-in IVF search benchmark and only
> write the index, prefix the command with `FAISS_IVF_BUILD_ONLY=1`.

> **Paper settings.** The commands in this section are generic examples. The
> paper uses `-num_clusters 1024` for Wiki and `4096` for every other dataset.
> For **Deep**, the graph is additionally built with `-alpha 1.2`. All graphs
> use `-M 32 -efConstruction 400`. A complete, scripted reproduction of
> Fig. 8 is in [reproduce/fig8/](reproduce/fig8/).

### Step 2 — Build the CGIF graph

`cgif_search` builds its own proximity graph. Pass `-build_only` with a
`-graph_path` to build the graph once and save it:

```bash
build/cgif_search \
    -dataset deep -N 1000000 \
    -dataset_path   /path/to/base.fvecs \
    -query_path     /path/to/query.fvecs \
    -gt_path        /path/to/deep.gt \
    -ivf_index_path /path/to/deep_faiss_ivf.index \
    -graph_path     /path/to/deep_cgif.hnsw \
    -M 32 -efConstruction 400 \
    -build_only
```

### Step 3 — Run filtered queries

Point CGIF at the pre-built graph (omit `-build_only`); it sweeps a range of
`efSearch` values and reports recall / QPS / distance computations per selectivity
bucket:

```bash
build/cgif_search \
    -dataset deep -N 1000000 \
    -dataset_path   /path/to/base.fvecs \
    -query_path     /path/to/query.fvecs \
    -gt_path        /path/to/deep.gt \
    -ivf_index_path /path/to/deep_faiss_ivf.index \
    -graph_path     /path/to/deep_cgif.hnsw
```

### Command-line reference

Common flags (both programs):

| Flag              | Meaning                                                        |
|-------------------|---------------------------------------------------------------|
| `-dataset`        | Dataset name (selects the loader), e.g. `deep`, `yfcc`, `laion` |
| `-N`              | Number of base points to load                                 |
| `-dataset_path`   | Path to base vectors                                           |
| `-query_path`     | Path to query vectors                                          |
| `-gt_path`        | Path to the `.gt` file (the `.bitsets` file is inferred)       |
| `-data_tag_path` / `-query_tag_path` | Tag/predicate files (tag-based datasets)   |

`build_ivf_index` additionally:

| Flag             | Meaning                                              |
|------------------|------------------------------------------------------|
| `-num_clusters`  | Number of IVF clusters (`nlist`), default `1024`     |
| `-save_index`    | Where to write the trained IVF index (must contain `faiss`) |
| `-ivf_index_path`| Load an existing IVF index instead of training       |

`cgif_search` additionally:

| Flag                    | Meaning                                                    |
|-------------------------|------------------------------------------------------------|
| `-ivf_index_path`       | Trained Faiss IVF index from Step 1                        |
| `-graph_path`           | CGIF graph file (built in Step 2, loaded in Step 3)        |
| `-build_only`           | Build & save the graph, then exit (no querying)            |
| `-M` / `-efConstruction`| Graph build parameters                                     |
| `-alpha`                | Pruning factor used during graph construction              |
| `-monotonicity_window`  | Search knob `w` (stall-detection window)                   |
| `-replacement_ratio`    | Search knob `x/ef` (candidate replacement ratio)           |
| `-num_runs`             | Repeat each `efSearch` setting N times; report trimmed-mean QPS |
| `-ablation_disable_ivf` / `-ablation_disable_twohop` / `-ablation_force_navigate` | Component-ablation toggles |

---

## Citation

If you use this code, please cite the CGIF paper:

Jiarui Luo, Chaoji Zuo, and Dong Deng. 2026. CGIF: Combining Proximity Graphs
and Inverted Files for Efficient Filtered Vector Search over Arbitrary
Predicates. *Proceedings of the VLDB Endowment* 19, 11 (2026), 3663–3675.
[https://doi.org/10.14778/3836663.3836716](https://doi.org/10.14778/3836663.3836716)

```bibtex
@article{luo2026cgif,
  author  = {Luo, Jiarui and Zuo, Chaoji and Deng, Dong},
  title   = {{CGIF}: Combining Proximity Graphs and Inverted Files for Efficient
             Filtered Vector Search over Arbitrary Predicates},
  journal = {Proceedings of the VLDB Endowment},
  volume  = {19},
  number  = {11},
  pages   = {3663--3675},
  year    = {2026},
  doi     = {10.14778/3836663.3836716},
  url     = {https://doi.org/10.14778/3836663.3836716}
}
```
