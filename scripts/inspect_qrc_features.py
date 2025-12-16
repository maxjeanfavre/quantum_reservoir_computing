# inspect_qrc_features.py
"""
Visualise how the QRC steady-state features AND quantumness / entanglement
measures depend on (x, y) for different encodings.

For each encoding (bare, couplings, gamma, dephase, rich1, rich2), we:
- build a QRC reservoir
- sample a grid of (x, y) in [-1, 1]^2
- compute:
    * reservoir.features(X) -> density-matrix-based features
    * quantumness / entanglement measures from the steady state:
        - l1_coherence
        - local_coherence_L, local_coherence_R
        - concurrence
        - negativity
        - log_negativity
        - chsh_max
- plot:
    * 2D heatmaps of all features
    * 1D slices at y = 0 for all features
    * 2D heatmaps of all measures
    * 1D slices at y = 0 for all measures
- save the results as PDFs in a dedicated subfolder:

    fig_feat_encodings/<encoding_name>/
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt

from reservoirs.quantum_reservoir import QuantumSteadyStateReservoir
from quantum.encodings import (
    bare_encoding_2d,
    simple_bare_gamma_encoding_2d,
    drive_encoding_2d,
    drive2_encoding_2d,
    couplings_encoding_2d,
    gamma_encoding_2d,
    gamma1_encoding_2d,
    gamma2_encoding_2d,
    dephase_encoding_2d,
    rich1_encoding_2d,
    rich2_encoding_2d,
    rich3_encoding_2d,
    rich4_encoding_2d,
    bare_drive_encoding_2d,
)

from quantum.entanglement_measures import (
    l1_coherence,
    local_coherence_L,
    local_coherence_R,
    concurrence,
    negativity,
    log_negativity,
    chsh_max,
)

# Adjust this import path if your quantum_system module lives elsewhere
from quantum.quantum_system import steady_state_from_params

# ---------------------------------------------------------------------
# Feature names for rho_to_features2
# ---------------------------------------------------------------------
FEATURE_NAMES_DM = [
    "r00", "r01", "r10",
    "alpha_R", "alpha_I",
    "beta_R", "beta_I",
    "v_R", "v_I",
    "x_R", "x_I",
    "y_R", "y_I",
    "z_R", "z_I",
]

# ---------------------------------------------------------------------
# Quantumness / entanglement measures
# ---------------------------------------------------------------------
MEASURE_FUNCS = [
    l1_coherence,
    local_coherence_L,
    local_coherence_R,
    concurrence,
    negativity,
    log_negativity,
    chsh_max,
]

MEASURE_NAMES = [
    "l1_coherence",
    "local_coh_L",
    "local_coh_R",
    "concurrence",
    "negativity",
    "log_negativity",
    "CHSH_max",
]


# ------------------------------------------------------------
# Compute grid of reservoir features
# ------------------------------------------------------------
def compute_feature_grid(reservoir, x_min=-1.0, x_max=1.0,
                         y_min=-1.0, y_max=1.0, n_points=30):
    """
    Sample a grid of (x, y), compute reservoir.features(X),
    and reshape into F[k, i, j] where k = feature index,
    i = x index, j = y index.
    """
    xs = np.linspace(x_min, x_max, n_points)
    ys = np.linspace(y_min, y_max, n_points)

    X = np.array([[x, y] for x in xs for y in ys])
    Phi = reservoir.features(X)   # (nx * ny, D)

    n_feat = Phi.shape[1]
    nx, ny = len(xs), len(ys)

    F = np.zeros((n_feat, nx, ny))
    for k in range(n_feat):
        F[k] = Phi[:, k].reshape(nx, ny)

    return xs, ys, F


# ------------------------------------------------------------
# Compute grid of quantumness / entanglement measures
# ------------------------------------------------------------
def compute_measure_grid(encoding_fn,
                         x_min=-1.0, x_max=1.0,
                         y_min=-1.0, y_max=1.0,
                         n_points=30):
    """
    Sample a grid of (x, y), compute the steady state rho(x,y)
    using the given encoding function and steady_state_from_params,
    then evaluate all measures in MEASURE_FUNCS.

    Returns
    -------
    xs : array of shape (nx,)
    ys : array of shape (ny,)
    M  : array of shape (n_meas, nx, ny)
         M[m, i, j] = measure_m( rho(x_i, y_j) )
    """
    xs = np.linspace(x_min, x_max, n_points)
    ys = np.linspace(y_min, y_max, n_points)

    nx, ny = len(xs), len(ys)
    n_meas = len(MEASURE_FUNCS)

    M = np.zeros((n_meas, nx, ny), dtype=float)

    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            u = np.array([x, y])
            params = encoding_fn(u)
            rho = steady_state_from_params(params)

            for m, func in enumerate(MEASURE_FUNCS):
                M[m, i, j] = func(rho)

    return xs, ys, M


# ------------------------------------------------------------
# 2D heatmaps for features
# ------------------------------------------------------------
def plot_feature_maps(xs, ys, F, outpath,
                      feature_names=None,
                      max_plots=15,
                      cmap="viridis"):
    """
    Plot 2D heatmaps of each feature over the (x, y) grid.
    """
    n_feat = F.shape[0]
    n_plot = min(n_feat, max_plots)

    n_cols = 5
    n_rows = int(np.ceil(n_plot / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3 * n_cols, 3 * n_rows),
                             squeeze=False)

    axes = axes.ravel()
    extent = [xs[0], xs[-1], ys[0], ys[-1]]

    for k in range(n_plot):
        ax = axes[k]
        im = ax.imshow(
            F[k].T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap=cmap,
        )
        # Use feature name if provided
        if feature_names is not None and k < len(feature_names):
            title = feature_names[k]
        else:
            title = f"Feature {k}"
        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(im, ax=ax)

    for k in range(n_plot, len(axes)):
        axes[k].axis("off")

    fig.suptitle("QRC steady-state feature maps", fontsize=14)
    plt.tight_layout()

    fig.savefig(os.path.join(outpath, "feature_maps.pdf"))
    plt.close(fig)


# ------------------------------------------------------------
# 3D surfaces for features (currently unused)
# ------------------------------------------------------------
def plot_feature_3d(xs, ys, F, k, outpath, feature_names=None):
    """
    Plot a single feature k as a 3D surface over (x, y).
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    Xs, Ys = np.meshgrid(xs, ys, indexing='ij')
    Z = F[k]

    if feature_names is not None and k < len(feature_names):
        name = feature_names[k]
    else:
        name = f"Feature {k}"

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")

    surf = ax.plot_surface(
        Xs, Ys, Z,
        cmap="viridis",
        edgecolor="none",
        antialiased=True,
    )

    ax.set_title(f"{name} in 3D")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel(name)
    fig.colorbar(surf, shrink=0.5)

    # Use the feature name in the filename if available
    safe_name = name.replace("/", "_")
    fig.savefig(os.path.join(outpath, f"{safe_name}_3d.pdf"))
    plt.close(fig)


# ------------------------------------------------------------
# 1D y = const slices for features
# ------------------------------------------------------------
def plot_feature_slices(xs, ys, F, outpath,
                        y_slice=0.0,
                        max_plots=15,
                        feature_names=None):
    """
    Plot slices phi_k(x, y_slice) for all features k.
    Saves as a single PDF.
    """
    idx_y = np.argmin(np.abs(ys - y_slice))

    n_feat = F.shape[0]
    n_plot = min(max_plots, n_feat)

    n_cols = 3
    n_rows = int(np.ceil(n_plot / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 3 * n_rows),
                             squeeze=False)
    axes = axes.ravel()

    for k in range(n_plot):
        ax = axes[k]
        slice_k = F[k, :, idx_y]

        if feature_names is not None and k < len(feature_names):
            name = feature_names[k]
        else:
            name = f"Feature {k}"

        ax.plot(xs, slice_k, lw=2)
        ax.set_title(f"{name}, slice at y={y_slice}")
        ax.set_xlabel("x")
        ax.set_ylabel(f"{name}(x, {y_slice})")

    for k in range(n_plot, len(axes)):
        axes[k].axis("off")

    fig.suptitle(f"QRC feature slices at y={y_slice}", fontsize=14)
    plt.tight_layout()

    fname = f"feature_slices_y{y_slice}.pdf"
    fig.savefig(os.path.join(outpath, fname))
    plt.close(fig)


# ------------------------------------------------------------
# 2D heatmaps for quantumness / entanglement measures
# ------------------------------------------------------------
def plot_measure_maps(xs, ys, M, outpath,
                      measure_names=None,
                      cmap="viridis"):
    """
    Plot 2D heatmaps of each measure over the (x, y) grid.
    """
    n_meas = M.shape[0]
    n_cols = 3
    n_rows = int(np.ceil(n_meas / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3.5 * n_rows),
        squeeze=False,
    )

    axes = axes.ravel()
    extent = [xs[0], xs[-1], ys[0], ys[-1]]

    for m in range(n_meas):
        ax = axes[m]
        im = ax.imshow(
            M[m].T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap=cmap,
        )

        if measure_names is not None and m < len(measure_names):
            title = measure_names[m]
        else:
            title = f"Measure {m}"

        ax.set_title(title)
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        plt.colorbar(im, ax=ax)

    # Hide any unused subplots
    for k in range(n_meas, len(axes)):
        axes[k].axis("off")

    fig.suptitle("QRC quantumness & entanglement maps", fontsize=14)
    plt.tight_layout()

    fig.savefig(os.path.join(outpath, "quantumness_maps.pdf"))
    plt.close(fig)


# ------------------------------------------------------------
# 1D y = const slices for quantumness / entanglement measures
# ------------------------------------------------------------
def plot_measure_slices(xs, ys, M, outpath,
                        y_slice=0.0,
                        measure_names=None):
    """
    Plot slices measure_m(x, y_slice) for all measures m.
    Saves as a single PDF.
    """
    idx_y = np.argmin(np.abs(ys - y_slice))

    n_meas = M.shape[0]
    n_cols = 3
    n_rows = int(np.ceil(n_meas / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4 * n_cols, 3 * n_rows),
        squeeze=False,
    )
    axes = axes.ravel()

    for m in range(n_meas):
        ax = axes[m]
        slice_m = M[m, :, idx_y]

        if measure_names is not None and m < len(measure_names):
            name = measure_names[m]
        else:
            name = f"Measure {m}"

        ax.plot(xs, slice_m, lw=2)
        ax.set_title(f"{name}, slice at y={y_slice}")
        ax.set_xlabel("x")
        ax.set_ylabel(f"{name}(x, {y_slice})")

    for k in range(n_meas, len(axes)):
        axes[k].axis("off")

    fig.suptitle(f"QRC quantumness slices at y={y_slice}", fontsize=14)
    plt.tight_layout()

    fname = f"quantumness_slices_y{y_slice}.pdf"
    fig.savefig(os.path.join(outpath, fname))
    plt.close(fig)


# ------------------------------------------------------------
# Available encodings mapping
# ------------------------------------------------------------
ALL_ENCODINGS = {
    "bare": bare_encoding_2d,
    "simple_bare_gamma": simple_bare_gamma_encoding_2d,
    "drive": drive_encoding_2d,
    "drive2": drive2_encoding_2d,
    "couplings": couplings_encoding_2d,
    "gamma": gamma_encoding_2d,
    "gamma1": gamma1_encoding_2d,
    "gamma2": gamma2_encoding_2d,
    "dephase": dephase_encoding_2d,
    "rich1": rich1_encoding_2d,
    "rich2": rich2_encoding_2d,
    "rich3": rich3_encoding_2d,
    "rich4": rich4_encoding_2d,
    "bare_drive": bare_drive_encoding_2d,
}


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Visualize quantum reservoir computing features and quantumness measures',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate figures for all encodings
  python inspect_qrc_features.py --all-encodings

  # Generate figures for specific encodings
  python inspect_qrc_features.py --encodings bare,drive,gamma

  # Specify custom output directory
  python inspect_qrc_features.py --encodings rich1,rich2 --output-dir my_figures
        """
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='fig_feat_encodings',
        help='Output directory for figures (default: fig_feat_encodings)'
    )
    
    # Encoding selection (mutually exclusive)
    encoding_group = parser.add_mutually_exclusive_group(required=True)
    encoding_group.add_argument(
        '--encodings',
        type=str,
        help='Comma-separated list of encodings to visualize (e.g., bare,drive,gamma). '
             f'Available: {", ".join(ALL_ENCODINGS.keys())}'
    )
    encoding_group.add_argument(
        '--all-encodings',
        action='store_true',
        help='Generate figures for all available encodings'
    )
    
    args = parser.parse_args()
    
    # Build encoding dictionary based on arguments
    if args.all_encodings:
        encoding_dict = ALL_ENCODINGS.copy()
    else:
        encoding_names = [e.strip() for e in args.encodings.split(',')]
        invalid = [e for e in encoding_names if e not in ALL_ENCODINGS]
        if invalid:
            print(f"Error: Unknown encodings: {invalid}", file=sys.stderr)
            print(f"Available encodings: {', '.join(ALL_ENCODINGS.keys())}", file=sys.stderr)
            sys.exit(1)
        encoding_dict = {name: ALL_ENCODINGS[name] for name in encoding_names}

    base_outdir = args.output_dir
    os.makedirs(base_outdir, exist_ok=True)

    for enc_name, enc_fn in encoding_dict.items():
        print(f"\n=== Encoding: {enc_name} ===")

        # Build reservoir for this encoding (for DM-based features)
        q_res = QuantumSteadyStateReservoir(
            input_dim=2,
            encoding_fn=enc_fn,
            name=f"qrc_steady_{enc_name}",
        )

        # Subfolder for this encoding
        outpath = os.path.join(base_outdir, enc_name)
        os.makedirs(outpath, exist_ok=True)

        # --------------------------------------------------------
        # 1) Compute feature grid
        # --------------------------------------------------------
        xs, ys, F = compute_feature_grid(
            q_res,
            x_min=-1, x_max=1,
            y_min=-1, y_max=1,
            n_points=30,
        )

        print(f"  -> Computed {F.shape[0]} features on "
              f"{len(xs)}×{len(ys)} grid.")

        # 2D heatmaps (features)
        plot_feature_maps(xs, ys, F, outpath,
                          feature_names=FEATURE_NAMES_DM)

        # 3D plots for first min(15, D) features (currently disabled)
        """
        for k in range(min(15, F.shape[0])):
            plot_feature_3d(xs, ys, F, k, outpath,
                            feature_names=FEATURE_NAMES_DM)
        """

        # 1D slices at y = 0 (features)
        plot_feature_slices(xs, ys, F, outpath,
                            y_slice=0.0,
                            max_plots=15,
                            feature_names=FEATURE_NAMES_DM)

        # --------------------------------------------------------
        # 2) Compute quantumness / entanglement measure grid
        # --------------------------------------------------------
        xs_m, ys_m, M = compute_measure_grid(
            enc_fn,
            x_min=-1, x_max=1,
            y_min=-1, y_max=1,
            n_points=30,
        )

        # Sanity check: should match feature grid axes
        assert np.allclose(xs, xs_m) and np.allclose(ys, ys_m)

        print(f"  -> Computed {M.shape[0]} quantumness measures on "
              f"{len(xs_m)}×{len(ys_m)} grid.")

        # 2D heatmaps (measures)
        plot_measure_maps(xs_m, ys_m, M, outpath,
                          measure_names=MEASURE_NAMES)

        # 1D slices at y = 0 (measures)
        plot_measure_slices(xs_m, ys_m, M, outpath,
                            y_slice=0.0,
                            measure_names=MEASURE_NAMES)


if __name__ == "__main__":
    main()
