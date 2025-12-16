#!/usr/bin/env bash
#
# Sweep dt values for dynamical QRC across multiple encodings and plot accuracy.
# Results are written to results/dt_sweep.csv and results/dt_sweep.png.
#
# Usage:
#   bash scripts/sweep_dt_and_plot.sh
# Optional environment overrides (linear sweep):
#   N_RUN=3 N_JOBS=-1 N_TRAIN=200 N_TEST=100 LAM=1e-4 DT_START=1 DT_END=15 \
#     bash scripts/sweep_dt_and_plot.sh
# Optional log sweep:
#   SWEEP_MODE=log DT_LOG_MIN=0.01 DT_LOG_MAX=100 DT_LOG_COUNT=10 bash scripts/sweep_dt_and_plot.sh

set -euo pipefail

ENCODINGS=("bare" "gamma" "rich")
DT_START="${DT_START:-1}"
DT_END="${DT_END:-15}"
SWEEP_MODE="${SWEEP_MODE:-log}" # linear | log
DT_LOG_MIN="${DT_LOG_MIN:-0.01}"
DT_LOG_MAX="${DT_LOG_MAX:-100}"
DT_LOG_COUNT="${DT_LOG_COUNT:-10}"
N_RUN="${N_RUN:-1}"
N_JOBS="${N_JOBS:--1}"
N_TRAIN="${N_TRAIN:-50}"
N_TEST="${N_TEST:-10}"
LAM="${LAM:-1e-4}"
SS_U0="${SS_U0:-0.0}"
INIT_FROM_SS="${INIT_FROM_SS:-1}"   # 1 -> start from steady state; 0 -> use INIT_STATE
INIT_STATE="${INIT_STATE:-}"        # used only when INIT_FROM_SS=0 (e.g., mixed, random, 00, 01, 10, 11)

RESULTS_DIR="results"
RESULTS_CSV="${RESULTS_DIR}/dt_sweep.csv"
RESULTS_PNG="${RESULTS_DIR}/dt_sweep.png"

mkdir -p "${RESULTS_DIR}"

python - <<'PY'
import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from run_dynamical import run_single_experiment

# Use maps from run_dynamical to avoid duplication
from run_dynamical import ENCODING_MAP, FEATURE_MAP

encodings = os.environ.get("ENCODINGS", "").split(",") if os.environ.get("ENCODINGS") else ["bare", "gamma", "rich"]
dt_start = float(os.environ.get("DT_START", 1))
dt_end = float(os.environ.get("DT_END", 15))
dt_log_min = float(os.environ.get("DT_LOG_MIN", 0.01))
dt_log_max = float(os.environ.get("DT_LOG_MAX", 100))
dt_log_count = int(os.environ.get("DT_LOG_COUNT", 10))
sweep_mode = os.environ.get("SWEEP_MODE", "linear").lower()  # linear or log
n_run = int(os.environ.get("N_RUN", 1))
n_jobs = int(os.environ.get("N_JOBS", -1))  # unused inside run_single_experiment but kept for parity
n_train = int(os.environ.get("N_TRAIN", 50))
n_test = int(os.environ.get("N_TEST", 10))
lam = float(os.environ.get("LAM", 1e-4))
ss_u0 = float(os.environ.get("SS_U0", 0.0))
init_from_ss = os.environ.get("INIT_FROM_SS", "1") == "1"
init_state = os.environ.get("INIT_STATE") or None

results_csv = os.environ.get("RESULTS_CSV", "results/dt_sweep.csv")
results_png = os.environ.get("RESULTS_PNG", "results/dt_sweep.png")

rows = []

# Build dt grid
if sweep_mode == "log":
    dts = np.logspace(np.log10(dt_log_min), np.log10(dt_log_max), dt_log_count)
else:
    # inclusive linear grid (integer steps if endpoints are integers)
    step = 1 if dt_end >= dt_start else -1
    num = int(abs(dt_end - dt_start) / abs(step)) + 1
    dts = np.linspace(dt_start, dt_end, num)

for enc in encodings:
    if enc not in ENCODING_MAP:
        print(f"Skipping unknown encoding '{enc}'")
        continue
    for dt in dts:
        train_accs = []
        test_accs = []
        base_seed = 0
        for i in range(n_run):
            seed = base_seed + i
            train_acc, test_acc = run_single_experiment(
                n_train=n_train,
                n_test=n_test,
                lam=lam,
                seed=seed,
                dt=float(dt),
                encoding=enc,
                feature_map="features",
                init_state_name=None if init_from_ss else init_state,
                init_from_ss=init_from_ss,
                ss_u0=ss_u0,
                verbose=False,
            )
            train_accs.append(train_acc)
            test_accs.append(test_acc)
        rows.append({
            "encoding": enc,
            "dt": float(dt),
            "train_mean": float(np.mean(train_accs)),
            "train_std": float(np.std(train_accs)),
            "test_mean": float(np.mean(test_accs)),
            "test_std": float(np.std(test_accs)),
        })
        print(f"[{enc}] dt={float(dt):.4g} -> test_mean={rows[-1]['test_mean']:.4f} ± {rows[-1]['test_std']:.4f}")

# Write CSV
os.makedirs(os.path.dirname(results_csv), exist_ok=True)
with open(results_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["encoding", "dt", "train_mean", "train_std", "test_mean", "test_std"])
    writer.writeheader()
    writer.writerows(rows)

# Plot: test_mean vs dt for each encoding
plt.figure(figsize=(7, 4))
for enc in encodings:
    xs = [r["dt"] for r in rows if r["encoding"] == enc]
    ys = [r["test_mean"] for r in rows if r["encoding"] == enc]
    if not xs:
        continue
    plt.plot(xs, ys, marker="o", label=enc)

plt.xlabel("dt")
plt.ylabel("Test accuracy")
plt.title("Dynamical QRC: test accuracy vs dt")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(results_png, dpi=150)
print(f"\nSaved CSV to {results_csv}")
print(f"Saved plot to {results_png}")
PY

