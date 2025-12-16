#!/usr/bin/env python3
"""
Plot the time evolution of the dynamical quantum reservoir features on the
Sine/Square task and save the figures as PDFs.
"""

import sys
import os
# Add project root to path so we can import modules
# If we're in scripts/ directory, go up one level to project root
current_dir = os.getcwd()
if current_dir.endswith('scripts') or 'scripts' in current_dir:
    project_root = os.path.dirname(current_dir) if current_dir.endswith('scripts') else current_dir.split('scripts')[0].rstrip(os.sep)
    sys.path.insert(0, project_root)
else:
    # Already in project root
    sys.path.insert(0, current_dir)

import argparse
import warnings

import matplotlib.pyplot as plt
import numpy as np

from tasks.sinesquare import SineSquare
from reservoirs.quantum_dynamical import QuantumDynamicalReservoir
from quantum.dynamical_encodings import (
    bare_encoding_1d,
    bare_asymmetric_encoding_1d,
    dephase_encoding_1d,
    drive_encoding_1d,
    gamma_encoding_1d,
    rich_encoding_1d,
)
from quantum.quantum_system import rho_random
from quantum.features import rho_to_features

# Suppress noisy NumPy warnings on some platforms
warnings.filterwarnings("ignore", category=UserWarning, module="numpy._core.getlimits")


FEATURE_NAMES = [
    "r00",
    "r01",
    "r10",
    "alpha_R",
    "alpha_I",
    "beta_R",
    "beta_I",
    "v_R",
    "v_I",
    "x_R",
    "x_I",
    "y_R",
    "y_I",
    "z_R",
    "z_I",
]

ENCODING_MAP = {
    "bare": bare_encoding_1d,
    "bare_asym": bare_asymmetric_encoding_1d,
    "drive": drive_encoding_1d,
    "gamma": gamma_encoding_1d,
    "dephase": dephase_encoding_1d,
    "rich": rich_encoding_1d,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulate dynamical QRC feature evolution on the Sine/Square task.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=120,
        help="Number of timesteps to simulate.",
    )
    parser.add_argument(
        "--encoding",
        choices=ENCODING_MAP.keys(),
        default="bare",
        help="Input encoding used for the dynamical reservoir.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.1,
        help="Time step for the mesolve evolution.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for the task sequence.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="fig_dynamical_evolution",
        help="Directory where PDF figures are stored.",
    )
    parser.add_argument(
        "--save-data",
        action="store_true",
        help="Also save the raw sequence, labels, and features as NPZ.",
    )
    return parser.parse_args()


def build_reservoir(encoding_name: str, dt: float) -> QuantumDynamicalReservoir:
    encoding_fn = ENCODING_MAP[encoding_name]
    return QuantumDynamicalReservoir(
        input_dim=1,
        encoding_fn=encoding_fn,
        feature_fn=rho_to_features,
        dt=dt,
        init_from_ss=True,
        ss_u0=1.0,
        name=f"qrc_dyn_{encoding_name}",
    )


def plot_sequence(t: np.ndarray, sequence: np.ndarray, labels: np.ndarray, outdir: str):
    fig, ax1 = plt.subplots(figsize=(10, 3))
    ax1.plot(t, sequence, lw=2, color="tab:blue", label="input u_t")
    ax1.set_xlabel("time")
    ax1.set_ylabel("input amplitude")
    ax1.grid(alpha=0.25)

    if labels is not None:
        ax2 = ax1.twinx()
        ax2.step(t, labels, where="post", color="tab:red", alpha=0.4, label="label")
        ax2.set_ylabel("label")
        ax2.set_yticks([0, 1])
        ax2.set_ylim(-0.2, 1.2)
        lines, labels_list = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels_list + labels2, loc="upper right")
    else:
        ax1.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "input_sequence.pdf"))
    plt.close(fig)


def plot_feature_traces(
    t: np.ndarray,
    features: np.ndarray,
    outdir: str,
    feature_names: list[str] | None = None,
    sequence: np.ndarray | None = None,
    labels: np.ndarray | None = None,
):
    """
    Plot each feature trace; optionally overlay the input sequence (and labels)
    on a light secondary axis.
    """
    n_feat = features.shape[1]
    n_cols = 5
    n_rows = int(np.ceil(n_feat / n_cols))
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3.4 * n_cols, 2.4 * n_rows),
        sharex=True,
    )
    axes = axes.ravel()

    for k in range(n_feat):
        ax = axes[k]
        name = feature_names[k] if feature_names and k < len(feature_names) else f"f{k}"
        feat_line = ax.plot(t, features[:, k], lw=1.8, label=name)
        ax.set_title(name, fontsize=9)
        ax.grid(alpha=0.2)
        ax.set_xlim(t[0], t[-1])

        if sequence is not None:
            ax_seq = ax.twinx()
            ax_seq.plot(
                t,
                sequence,
                color="0.6",
                alpha=0.35,
                lw=1.0,
                label="u_t",
            )
            if labels is not None:
                ax_seq.step(
                    t,
                    labels,
                    where="post",
                    color="tab:red",
                    alpha=0.25,
                    lw=0.9,
                    label="label",
                )
            if k > 0:
                ax_seq.set_yticklabels([])
            else:
                ax_seq.set_ylabel("input", color="0.5")
            if k == 0:
                handles_feat, labels_feat = ax.get_legend_handles_labels()
                handles_seq, labels_seq = ax_seq.get_legend_handles_labels()
                ax.legend(
                    handles_feat + handles_seq,
                    labels_feat + labels_seq,
                    loc="upper right",
                    fontsize=8,
                )

    for k in range(n_feat, len(axes)):
        axes[k].axis("off")

    fig.supxlabel("time")
    fig.supylabel("feature value")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "feature_evolution.pdf"))
    plt.close(fig)


def main():
    args = parse_args()

    outdir = os.path.join(args.output_dir, args.encoding)
    os.makedirs(outdir, exist_ok=True)

    task = SineSquare(sequence_length=args.sequence_length)
    sequence, labels = task.generate_sequence(seed=args.seed)
    t = np.arange(len(sequence)) * args.dt

    reservoir = build_reservoir(args.encoding, args.dt)
    features = reservoir.process_sequence(sequence)

    #plot_sequence(t, sequence, labels, outdir)
    plot_feature_traces(
        t,
        features,
        outdir,
        feature_names=FEATURE_NAMES,
        sequence=sequence,
        labels=labels,
    )

    if args.save_data:
        np.savez(
            os.path.join(outdir, "dynamical_features.npz"),
            sequence=sequence,
            labels=labels,
            features=features,
            dt=args.dt,
            encoding=args.encoding,
        )

    print(f"Saved plots to {outdir}")


if __name__ == "__main__":
    main()

