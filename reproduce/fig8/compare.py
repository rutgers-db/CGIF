#!/usr/bin/env python3
"""Compare a reproduction run against the numbers behind Fig. 8 of the paper.

Reads <log_dir>/{cgif,postfilter}_{positive,neutral,negative}.log, which
run_fig8.sh writes, and expected_fig8.csv (the paper's numbers for all six
methods). It prints recall and QPS side by side for every ef. With --plot,
it also writes one QPS-recall panel per regime, overlaying the reproduced
curves (solid) on the paper's curves (dashed).

Recall should match to about +-0.01. QPS depends on the CPU. The paper's
numbers were measured single-threaded, pinned to one core with taskset, on
an Intel Xeon Gold 6212U (2.4 GHz).
"""

import argparse
import csv
import os
import re
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
REGIMES = ["positive", "neutral", "negative"]
RUN_METHODS = [("CGIF", "cgif"), ("PostFilter-HNSW", "postfilter")]

PAT_R = re.compile(r"Average Recall:\s*([\d.]+)")
PAT_Q = re.compile(r"Average QPS:\s*([\d.]+)")
PAT_EF = re.compile(r"(?:Result for )?Search ef:\s*(\d+)")


def parse_log(path):
    """Return {ef: (recall, qps)} from a cgif_search / postfilter_hnsw log."""
    out, rec, qps = {}, None, None
    with open(path) as f:
        for line in f:
            if m := PAT_R.search(line):
                rec, qps = float(m.group(1)), None
            elif (m := PAT_Q.search(line)) and "legacy" not in line:
                qps = float(m.group(1))
            elif (m := PAT_EF.search(line)) and rec is not None and qps is not None:
                out[int(m.group(1))] = (rec, qps)
                rec = qps = None
    return out


def load_expected():
    exp = defaultdict(dict)  # (method, regime) -> {param: (recall, qps)}
    with open(os.path.join(HERE, "expected_fig8.csv")) as f:
        for row in csv.DictReader(f):
            exp[(row["method"], row["regime"])][row["param"]] = (float(row["recall"]), float(row["qps"]))
    return exp


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log_dir")
    ap.add_argument("--plot", metavar="OUT_DIR", help="also write fig8_<regime>.pdf here")
    args = ap.parse_args()

    expected = load_expected()
    got = {}
    worst = 0.0
    for name, stem in RUN_METHODS:
        for reg in REGIMES:
            path = os.path.join(args.log_dir, f"{stem}_{reg}.log")
            if not os.path.exists(path):
                print(f"(missing {path})")
                continue
            got[(name, reg)] = run = parse_log(path)
            ref = expected[(name, reg)]
            print(f"\n{name} / {reg}")
            print(f"  {'ef':>4} | {'recall paper':>12} {'repro':>7} {'diff':>7} | {'QPS paper':>9} {'repro':>8}")
            for ef in sorted(run):
                r, q = run[ef]
                pr, pq = ref.get(str(ef), (float("nan"), float("nan")))
                worst = max(worst, abs(r - pr))
                print(f"  {ef:>4} | {pr:>12.4f} {r:>7.4f} {r - pr:>+7.4f} | {pq:>9.0f} {q:>8.0f}")
    print(f"\nmax |recall diff| = {worst:.4f}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        os.makedirs(args.plot, exist_ok=True)
        colors = {"CGIF": "b", "ACORN": "tab:orange", "Faiss-IVF": "tab:green",
                  "InlineFilter-HNSW": "tab:purple", "PostFilter-HNSW": "tab:red", "NaVix": "tab:brown"}
        for reg in REGIMES:
            fig, ax = plt.subplots(figsize=(6, 3.6))
            for name, c in colors.items():
                pts = sorted(expected[(name, reg)].values())
                if pts:
                    ax.plot(*zip(*pts), ls="--", color=c, marker="o", mfc="none", lw=1,
                            alpha=0.6, label=f"{name} (paper)")
                if (name, reg) in got:
                    pts = sorted(got[(name, reg)].values())
                    ax.plot(*zip(*pts), ls="-", color=c, marker="x", lw=1.5, label=f"{name} (repro)")
            ax.set_yscale("log")
            ax.set_xlabel("Recall@10")
            ax.set_ylabel("QPS")
            ax.set_title({"positive": "P-Corr", "neutral": "Neutral", "negative": "N-Corr"}[reg])
            ax.grid(True, ls="--", alpha=0.5)
            ax.legend(fontsize=6, ncol=2)
            out = os.path.join(args.plot, f"fig8_{reg}.pdf")
            fig.tight_layout()
            fig.savefig(out)
            plt.close(fig)
            print(f"wrote {out}")


if __name__ == "__main__":
    main()
