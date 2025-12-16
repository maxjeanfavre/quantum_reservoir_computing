# entanglement_measures.py
#
# Quantumness and nonlocality diagnostics for a 2-qubit steady state.
#
# Provided measures:
#   - l1_coherence(rho)
#   - local_coherence_L(rho), local_coherence_R(rho)
#   - concurrence(rho)
#   - negativity(rho, sys=0), log_negativity(rho, sys=0)
#   - chsh_max(rho)
#
# Assumes rho is a 4x4 density matrix (two qubits).

import numpy as np
from qutip import (
    Qobj,
    expect,
    ptrace,
    sigmay,
    tensor,
    partial_transpose,
)

# If you already define sx_L, sy_L, sz_L, sx_R, sy_R, sz_R in quantum_system.py,
# you can reuse them here:
try:
    from .quantum_system import (
        sx_L, sy_L, sz_L,
        sx_R, sy_R, sz_R,
    )
    _USE_LOCAL_PAULI = True
except ImportError:
    # Fallback: construct Pauli operators directly if quantum_system is not available.
    from qutip import sigmax, sigmay, sigmaz
    _USE_LOCAL_PAULI = False
    sx_L = tensor(sigmax(), Qobj(np.eye(2)))
    sy_L = tensor(sigmay(), Qobj(np.eye(2)))
    sz_L = tensor(sigmaz(), Qobj(np.eye(2)))
    sx_R = tensor(Qobj(np.eye(2)), sigmax())
    sy_R = tensor(Qobj(np.eye(2)), sigmay())
    sz_R = tensor(Qobj(np.eye(2)), sigmaz())


# ----------------------------------------------------------------------
# 1. Global coherence (ℓ1 norm of off-diagonals)
# ----------------------------------------------------------------------

def l1_coherence(rho: Qobj) -> float:
    """
    ℓ1 coherence of a 2-qubit state in the computational basis:

        C_l1(rho) = sum_{i != j} |rho_ij|

    This is a standard coherence measure in the chosen basis.
    """
    M = rho.full()
    diag = np.diag(np.diag(M))
    off_diag = M - diag
    return float(np.sum(np.abs(off_diag)))


# ----------------------------------------------------------------------
# 2. Local coherence on each qubit
# ----------------------------------------------------------------------

def local_coherence_L(rho: Qobj) -> float:
    """
    Local coherence of qubit L:

        C_L = |(rho_L)_{01}|

    where rho_L = Tr_R(rho) is the reduced 2x2 state of qubit L.
    """
    rhoL = ptrace(rho, 0)   # trace out R
    mat = rhoL.full()
    return float(np.abs(mat[0, 1]))


def local_coherence_R(rho: Qobj) -> float:
    """
    Local coherence of qubit R:

        C_R = |(rho_R)_{01}|

    where rho_R = Tr_L(rho) is the reduced 2x2 state of qubit R.
    """
    rhoR = ptrace(rho, 1)   # trace out L
    mat = rhoR.full()
    return float(np.abs(mat[0, 1]))


# ----------------------------------------------------------------------
# 3. Concurrence (Wootters formula)
# ----------------------------------------------------------------------

# Spin-flip operator Y ⊗ Y
_sigma_y = sigmay()
_Y_YY = tensor(_sigma_y, _sigma_y)


def concurrence(rho: Qobj) -> float:
    """
    Concurrence of a 2-qubit state (Wootters, 1998).

    Steps:
      1. ρ̃ = (σ_y ⊗ σ_y) ρ* (σ_y ⊗ σ_y)
      2. R = ρ ρ̃
      3. Let λ_i be eigenvalues of R, sorted descending.
      4. C = max(0, sqrt(λ_1) - sqrt(λ_2) - sqrt(λ_3) - sqrt(λ_4)).
    """
    M = rho.full()
    Y = _Y_YY.full()

    rho_tilde = Y @ M.conj() @ Y
    R = M @ rho_tilde

    eigvals = np.linalg.eigvals(R)
    eigvals = np.real(eigvals)
    eigvals = np.maximum(eigvals, 0.0)  # clip tiny negatives

    # Sort eigenvalues of R, then take square roots
    sqrts = np.sqrt(np.sort(eigvals)[::-1])
    # Pad in case of numerical issues
    if sqrts.size < 4:
        sqrts = np.pad(sqrts, (0, 4 - sqrts.size), mode="constant")

    l1, l2, l3, l4 = sqrts[:4]
    C = max(0.0, l1 - l2 - l3 - l4)
    return float(C)


# ----------------------------------------------------------------------
# 4. Negativity and log-negativity (partial transpose)
# ----------------------------------------------------------------------

def negativity(rho: Qobj, sys: int = 0) -> float:
    """
    Negativity of a 2-qubit state based on partial transpose.

    Parameters
    ----------
    rho : Qobj
        2-qubit density matrix.
    sys : int
        Which subsystem is transposed (0 or 1). For 2 qubits this
        does not change the negativity, but is included for generality.

    Returns
    -------
    N : float
        Sum of absolute values of negative eigenvalues of rho^T_sys.
    """
    if sys not in (0, 1):
        raise ValueError("sys must be 0 or 1 for 2-qubit state.")

    # indicator list: 1 => transpose that subsystem, 0 => leave as is
    if sys == 0:
        mask = [1, 0]
    else:
        mask = [0, 1]

    rho_pt = partial_transpose(rho, mask)
    eigvals = np.linalg.eigvals(rho_pt.full())
    eigvals = np.real(eigvals)

    neg = np.sum(np.abs(eigvals[eigvals < 0.0]))
    return float(neg)


def log_negativity(rho: Qobj, sys: int = 0) -> float:
    """
    Logarithmic negativity of a 2-qubit state:

        E_N = log2( || rho^T_sys ||_1 )

    where ||·||_1 is the trace norm.
    """
    if sys not in (0, 1):
        raise ValueError("sys must be 0 or 1 for 2-qubit state.")

    if sys == 0:
        mask = [1, 0]
    else:
        mask = [0, 1]

    rho_pt = partial_transpose(rho, mask)
    eigvals = np.linalg.eigvals(rho_pt.full())
    eigvals = np.real(eigvals)

    trace_norm = np.sum(np.abs(eigvals))
    # Avoid log of zero due to numerical issues
    trace_norm = max(trace_norm, 1e-16)
    return float(np.log2(trace_norm))


# ----------------------------------------------------------------------
# 5. CHSH maximal value (nonlocality measure)
# ----------------------------------------------------------------------

def chsh_max(rho: Qobj) -> float:
    """
    Maximal CHSH value S_max for a 2-qubit state rho.

    Uses the Horodecki criterion:

      1. T_ij = Tr[rho sigma_i ⊗ sigma_j], i,j in {x,y,z}
      2. U = T^T T, eigenvalues m1 >= m2 >= m3
      3. S_max = 2 sqrt(m1 + m2)

    Nonlocality iff S_max > 2.
    """
    TL = [sx_L, sy_L, sz_L]
    TR = [sx_R, sy_R, sz_R]

    T = np.zeros((3, 3), dtype=float)
    for i in range(3):
        for j in range(3):
            T[i, j] = float(expect(TL[i] * TR[j], rho))

    U = T.T @ T
    m = np.linalg.eigvalsh(U)
    m = np.sort(m)[::-1]  # descending

    Smax = 2.0 * np.sqrt(max(m[0] + m[1], 0.0))
    return float(Smax)

