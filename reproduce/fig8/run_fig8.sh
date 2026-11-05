#!/usr/bin/env bash
# Reproduce Fig. 8 (effect of predicate-embedding correlation on Deep-1M)
# for CGIF and PostFilter-HNSW.
#
# Usage:
#   BASE=/path/to/deep_1M.fvecs QUERY=/path/to/deep_query.fvecs \
#   WORK=/path/to/workdir  reproduce/fig8/run_fig8.sh
#
# Optional environment variables:
#   BUILD  directory holding cgif_search / build_ivf_index / postfilter_hnsw
#          (default: <repo>/build)
#   GRAPH  reuse an existing CGIF graph instead of building one
#   IVF    reuse an existing Faiss IVF index (path must contain "faiss")
#   CORE   CPU core for the single-threaded search runs (default: 0)
#
# Every step is skipped when its output already exists, so the script can be
# re-run after an interruption.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
: "${BASE:?set BASE to the Deep-1M base vectors (.fvecs)}"
: "${QUERY:?set QUERY to the Deep query vectors (.fvecs)}"
: "${WORK:?set WORK to an output directory (~1.2 GB of predicates + ~1 GB of indexes)}"
BUILD="${BUILD:-$REPO/build}"
CORE="${CORE:-0}"
N=1000000

PRED="$WORK/predicates"
LOGS="$WORK/logs"
GRAPH="${GRAPH:-$WORK/deep-cgif-M32-efc400-alpha1.2.hnsw}"
IVF="${IVF:-$WORK/deep-faiss-ivfflat-4096.index}"
mkdir -p "$PRED" "$LOGS"
COMMON=(-dataset deep -N "$N" -dataset_path "$BASE" -query_path "$QUERY")

# 1. Predicate workloads: 100 queries x 28 selectivities x 3 regimes.
if [[ ! -f "$PRED/correlated-negative-top10-deep-$N.bitsets" ]]; then
    echo "[1/5] generating predicates -> $PRED"
    python3 "$HERE/gen_predicates.py" --base "$BASE" --query "$QUERY" --out_dir "$PRED" \
        | tee "$LOGS/gen_predicates.log"
fi
GT_ANY="$PRED/correlated-neutral-top10-deep-$N.gt"

# 2. IVF index: nlist = 4096 for Deep. (The 1024 in the top-level README is
#    the Wiki setting.)
if [[ ! -f "$IVF" ]]; then
    echo "[2/5] building IVF index (nlist=4096) -> $IVF"
    FAISS_IVF_BUILD_ONLY=1 "$BUILD/build_ivf_index" "${COMMON[@]}" -gt_path "$GT_ANY" \
        -num_clusters 4096 -save_index "$IVF" > "$LOGS/build_ivf.log" 2>&1
fi

# 3. Graph: hnswlib M=32 (base-layer degree 2M = 64), efConstruction=400, and
#    pruning alpha=1.2. alpha=1.2 is used for Deep only; the other datasets
#    use 1.0.
if [[ ! -f "$GRAPH" ]]; then
    echo "[3/5] building CGIF graph (M=32, efC=400, alpha=1.2) -> $GRAPH"
    "$BUILD/cgif_search" "${COMMON[@]}" -gt_path "$GT_ANY" -ivf_index_path "$IVF" \
        -graph_path "$GRAPH" -M 32 -efConstruction 400 -alpha 1.2 -build_only \
        > "$LOGS/build_graph.log" 2>&1
fi

# 4. Search: single-threaded, pinned to one core. CGIF reports a trimmed-mean
#    QPS over 3 runs.
for regime in positive neutral negative; do
    GT="$PRED/correlated-$regime-top10-deep-$N.gt"
    if [[ ! -s "$LOGS/cgif_$regime.log" ]]; then
        echo "[4/5] CGIF            $regime"
        taskset -c "$CORE" "$BUILD/cgif_search" "${COMMON[@]}" -gt_path "$GT" \
            -ivf_index_path "$IVF" -graph_path "$GRAPH" -num_runs 3 > "$LOGS/cgif_$regime.log" 2>&1
    fi
    if [[ ! -s "$LOGS/postfilter_$regime.log" ]]; then
        echo "[4/5] PostFilter-HNSW $regime"
        taskset -c "$CORE" "$BUILD/postfilter_hnsw" "${COMMON[@]}" -gt_path "$GT" \
            -graph_path "$GRAPH" > "$LOGS/postfilter_$regime.log" 2>&1
    fi
done

# 5. Compare with the paper.
echo "[5/5] comparing with expected_fig8.csv"
python3 "$HERE/compare.py" "$LOGS" --plot "$WORK/plots" | tee "$WORK/compare.txt"
