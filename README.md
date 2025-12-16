# Quantum Reservoir Computing

A Python implementation of quantum reservoir computing for classification tasks. This project implements both **static (steady-state)** and **dynamical** quantum reservoirs using density matrices and compares them with classical Echo State Networks (ESN) and linear baselines.

## Installation

Install the required dependencies:

```bash
pip install -r requirements.txt
```

**Note:** Using a virtual environment is strongly recommended to avoid conflicts with other Python packages.

## Requirements

- Python 3.8, 3.9, 3.10, or 3.11
- NumPy >= 1.20.0, < 2.0.0
- QuTiP >= 4.7.0
- Joblib >= 1.0.0
- Matplotlib >= 3.3.0 (for visualization scripts)

## Usage

### Running Experiments

The project provides two main entry points:

- **`run_experiments.py`**: For static (steady-state) quantum reservoir experiments on 2D classification tasks
- **`run_dynamical.py`**: For dynamical quantum reservoir experiments on sequential tasks (e.g., Sine/Square waveform classification)

#### Static Reservoir Experiments (`run_experiments.py`)

This script runs quantum reservoirs using steady-state density matrices for 2D classification tasks.

#### Basic Examples

```bash
# Run specific encodings on TwoU task
python run_experiments.py --task two_u --encodings bare,drive,gamma_loss

# Run all encodings with parallel processing
python run_experiments.py --task two_u --all-encodings --n-jobs -1

# Run with custom parameters
python run_experiments.py --task two_spirals --n-train 512 --n-test 128 --n-run 10

# Include baseline and ESN
python run_experiments.py --task two_u --encodings bare --baseline --esn

# Run with custom ESN parameters
python run_experiments.py --task two_u --encodings bare --esn --esn-hidden 20 --esn-feat 20 --esn-input-scale 1.0
```

#### Command-Line Options

**Task Selection:**
- `--task`: Task to run (`two_u`, `two_spirals`, `two_circles`, `two_gaussians`), default: `two_u`

**Encoding Selection (required, choose one):**
- `--encodings`: Comma-separated list of encodings (e.g., `bare,drive,gamma_loss`)
- `--all-encodings`: Run all available encodings

**Feature Extraction:**
- `--feature-map`: Feature extraction method (`features`, `populations`, `pauli`), default: `features`

**Experiment Parameters:**
- `--n-train`: Number of training samples (default: 1024)
- `--n-test`: Number of test samples (default: 128)
- `--n-run`: Number of independent runs for statistics (default: 5)
- `--lam`: Ridge regression regularization parameter (default: 1e-4)
- `--seed-offset`: Base seed offset for random number generation (default: 0)
- `--n-jobs`: Number of parallel jobs (-1 for all cores, default: 1)

**Additional Experiments:**
- `--baseline`: Also run linear baseline experiment
- `--esn`: Also run ESN reservoir experiment

**ESN Parameters (only used with `--esn`):**
- `--esn-hidden`: Number of hidden units in ESN (default: 15)
- `--esn-feat`: Number of feature units in ESN (default: 15)
- `--esn-input-scale`: Input scaling for ESN (default: 0.5)
- `--esn-seed`: Random seed for ESN (default: 42)

**Output:**
- `--quiet`: Reduce output verbosity

#### Dynamical Reservoir Experiments (`run_dynamical.py`)

This script runs dynamical quantum reservoirs (or ESN) on sequential tasks where the reservoir state evolves over time.

**Example usage:**
```bash
# Quantum Reservoir (QRC) with dynamical evolution
python run_dynamical.py --reservoir qrc --encoding bare --feature-map features --dt 0.1

# Echo State Network (ESN)
python run_dynamical.py --reservoir esn --n-reservoir 300 --spectral-radius 0.9 --leak-rate 0.3 --washout 200

# Multiple runs with parallel processing
python run_dynamical.py --reservoir qrc --encoding gamma --n-run 10 --n-jobs -1
```

**Reservoir Selection:**
- `--reservoir`: Choose `qrc` (quantum) or `esn` (echo state network)

**QRC-specific Parameters:**
- `--encoding`: Encoding method (`bare`, `gamma`, `dephase`, `rich`)
- `--feature-map`: Feature extraction (`features`, `populations`, `pauli`)
- `--dt`: Time step for quantum evolution
- `--init-state`: Initial quantum state (optional)
- `--init-from-ss`: Start from steady state at u0

**ESN-specific Parameters:**
- `--n-reservoir`: Number of reservoir units
- `--spectral-radius`: Spectral radius of weight matrix
- `--sparsity`: Sparsity of connections
- `--leak-rate`: Leaky integration rate
- `--washout`: Number of initial timesteps to discard

**Common Parameters:**
- `--n-train`, `--n-test`: Number of training/test samples
- `--n-run`: Number of independent runs
- `--n-jobs`: Parallel jobs (-1 for all cores)
- `--lam`: Ridge regularization parameter

### Visualizing Features

The `scripts/inspect_qrc_features.py` script generates visualizations of quantum reservoir features and quantumness measures:

```bash
# Generate figures for all encodings
python scripts/inspect_qrc_features.py --all-encodings

# Generate figures for specific encodings
python scripts/inspect_qrc_features.py --encodings bare,drive,gamma

# Specify custom output directory
python scripts/inspect_qrc_features.py --encodings rich1,rich2 --output-dir my_figures
```

**Encoding Selection (required, choose one):**
- `--encodings`: Comma-separated list of encodings to visualize (e.g., `bare,drive,gamma`)
- `--all-encodings`: Generate figures for all available encodings

**Output:**
- `--output-dir`: Output directory for figures (default: `fig_feat_encodings`)

This script creates 2D heatmaps and 1D slices showing how features and quantumness measures vary across the input space for different encodings.

## Project Structure

```
quantum_reservoir_computing/
├── quantum/              # Quantum system implementation
│   ├── quantum_system.py      # Core quantum system dynamics
│   ├── encodings.py            # Input encoding functions
│   ├── features.py             # Feature extraction from density matrices
│   └── entanglement_measures.py # Quantumness/entanglement measures
├── reservoirs/           # Reservoir implementations
│   ├── base.py                # Base reservoir interface
│   ├── quantum_reservoir.py   # Quantum steady-state reservoir (static)
│   ├── quantum_dynamical.py   # Quantum dynamical reservoir
│   ├── esn.py                 # Echo State Network (classical baseline, static)
│   └── esn_dynamical.py       # Echo State Network (dynamical)
├── tasks/                # Classification tasks
│   ├── base.py                # Base task interface
│   ├── two_u.py               # Two U-shaped classes
│   ├── spirals.py             # Two spirals
│   ├── circles.py             # Two circles
│   ├── gaussians.py           # Two Gaussians
│   └── sinesquare.py          # Sine/Square waveform task (for dynamical experiments)
├── readout/              # Readout layer
│   └── linear.py             # Linear (ridge) classifier
├── experiments/         # Experiment framework
│   ├── config.py             # Experiment configuration
│   ├── core.py                # Core experiment execution functions
│   └── runner.py              # Batch experiment runner
├── scripts/              # Utility scripts
│   ├── inspect_qrc_features.py  # Feature visualization script
│   └── test_installation.py     # Installation verification script
├── notebooks/            # Jupyter notebooks for exploration
│   ├── run.ipynb
│   ├── run_circle.ipynb
│   ├── run_spiral.ipynb
│   └── run_u.ipynb
├── run_experiments.py    # Static reservoir experiments (steady-state)
└── run_dynamical.py      # Dynamical reservoir experiments (time-evolution)
```

## Available Encodings

The following encodings are available for quantum reservoirs:

- `bare`: Basic encoding
- `simple_bare_gamma`: Simple bare encoding with gamma
- `drive`: Drive-based encoding
- `drive2`: Alternative drive encoding
- `drive_gamma`: Drive with gamma
- `couplings`: Coupling-based encoding
- `gamma_loss`: Gamma loss encoding
- `gamma1`, `gamma2`: Variants of gamma encoding
- `dephase`: Dephasing encoding
- `rich1`, `rich2`, `rich3`, `rich4`: Rich encodings with various parameters
- `bare_drive`: Combined bare and drive encoding

## Available Tasks

- **TwoU**: Two U-shaped classes
- **TwoSpirals**: Two interleaved spirals
- **TwoCircles**: Two concentric circles
- **TwoGaussians**: Two Gaussian distributions

## How It Works

### Static (Steady-State) Reservoirs

1. **Input Encoding**: 2D input data is encoded into quantum system parameters (drives, couplings, decay rates, etc.)
2. **Quantum Evolution**: The quantum system evolves to a steady state, represented by a density matrix
3. **Feature Extraction**: Features are extracted from the steady-state density matrix (e.g., matrix elements, Pauli features, populations)
4. **Classification**: A linear (ridge) classifier is trained on the extracted features
5. **Evaluation**: Performance is measured via classification accuracy on train and test sets

### Dynamical Reservoirs

1. **Input Encoding**: Sequential input data is encoded into quantum system parameters at each timestep
2. **Time Evolution**: The quantum system evolves dynamically according to the Lindblad master equation, maintaining an internal quantum state
3. **Feature Extraction**: Features are extracted from the density matrix at each timestep
4. **Classification**: A linear (ridge) classifier is trained on the sequence of extracted features
5. **Evaluation**: Performance is measured via classification accuracy on train and test sequences

## Troubleshooting

### NumPy Warnings

If you see NumPy longdouble warnings, these are harmless and related to platform compatibility. They are automatically suppressed in the code.

## Notes

- NumPy longdouble warnings are suppressed (harmless, related to platform compatibility)
- The `good_runs_to_keep/` directory contains archival notebooks and is excluded from version control
- Generated figures are saved in `fig_feat_encodings/` by default (excluded from version control)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

