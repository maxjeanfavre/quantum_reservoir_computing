# Warning filter set in quantum/__init__.py, but set here too for safety
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='numpy._core.getlimits')

import numpy as np
from qutip import (
    basis, qeye, tensor,
    sigmap, sigmam, sigmaz, sigmax, sigmay,
    steadystate, mesolve, propagator,
    ket2dm, Qobj
)

# Single-qubit operators
I = qeye(2)
sp = sigmap()
sm = sigmam()
sz = sigmaz()
sx = sigmax()
sy = sigmay()

# Two-qubit operators: left (L) and right (R)
sp_L = tensor(sp, I)
sp_R = tensor(I, sp)
sm_L = tensor(sm, I)
sm_R = tensor(I, sm)
sz_L = tensor(sz, I)
sz_R = tensor(I, sz)
sx_L = tensor(sx, I)
sx_R = tensor(I, sx)
sy_L = tensor(sy, I)
sy_R = tensor(I, sy)

# Correlators
sxL_sxR = tensor(sx, sx)
sxL_syR = tensor(sx, sy)
sxL_szR = tensor(sx, sz)

syL_sxR = tensor(sy, sx)
syL_syR = tensor(sy, sy)
syL_szR = tensor(sy, sz)

szL_sxR = tensor(sz, sx)
szL_syR = tensor(sz, sy)
szL_szR = tensor(sz, sz)

# Number operators n = |1><1| = (I - sz)/2
n_L = 0.5 * (tensor(I, I) - sz_L)
n_R = 0.5 * (tensor(I, I) - sz_R)

# Computational basis states (not strictly needed, but handy)
b0 = basis(2, 0)
b1 = basis(2, 1)
ket_00 = tensor(b0, b0)
ket_01 = tensor(b0, b1)
ket_10 = tensor(b1, b0)
ket_11 = tensor(b1, b1)

# Density matrices for computational basis states
rho_00 = ket2dm(ket_00)
rho_01 = ket2dm(ket_01)
rho_10 = ket2dm(ket_10)
rho_11 = ket2dm(ket_11)

rho_list = {
    "00": rho_00,
    "01": rho_01,
    "10": rho_10,
    "11": rho_11,
    "mixed": 0.5 * (rho_00 + rho_11),
    "all": 0.25 * (rho_00 + rho_01 + rho_10 + rho_11),
}

def rho_random():
    """Build and return a random density matrix which satisfies density matrix constraints."""
    # Generate random complex matrix
    A = np.random.randn(4, 4) + 1j * np.random.randn(4, 4)
    
    # Form A @ A† which is automatically positive semidefinite
    H = A @ A.T.conj()
    
    # Normalize to trace 1
    H = H / np.trace(H)
    
    # Convert to Qutip Qobj
    return Qobj(H, dims=[[2, 2], [2, 2]])


def rho_iss(u0=0.0, encoding_fn=None, params: dict | None = None, method: str = "direct"):
    """
    Return the steady-state density matrix for the system at input u0.
    
    Supply either:
      - params: explicit parameter dictionary, or
      - encoding_fn: callable mapping u0 -> params (same encoding used during dynamics)
    
    By default u0=0.0 so you can initialize the system at the fixed point
    corresponding to zero input before driving it with a task sequence.
    """
    if params is None:
        if encoding_fn is None:
            raise ValueError("Provide either params or encoding_fn to build the steady state.")
        params = encoding_fn(u0)
    return steady_state_from_params(params, method=method)


def build_hamiltonian(params: dict):
    """Return 2-qubit Hamiltonian for given parameter dict."""
    eps_L = params.get("eps_L", 0.0)
    eps_R = params.get("eps_R", 0.0)
    U     = params.get("U", 0.0)
    g_res = params.get("g_res", 0.0)
    g_off = params.get("g_off", 0.0)

    f_L   = params.get("f_L", 0.0)
    f_R   = params.get("f_R", 0.0)

    # Bare + interaction + tunnelling
    H0   = eps_L * n_L + eps_R * n_R + U * n_L * n_R
    Hres = g_res * (sp_L * sm_R + sm_L * sp_R)
    Hoff = g_off * (sp_L * sp_R + sm_L * sm_R)

    # Drive term: f_L (σ_L^+ + σ_L^-) + f_R (σ_R^+ + σ_R^-)
    Hdrive = f_L * (sp_L + sm_L) + f_R * (sp_R + sm_R)

    return H0 + Hres + Hoff + Hdrive


def build_c_ops(params: dict):
    """Return list of collapse operators for given parameter dict."""
    gLp = params.get("gamma_L_plus", 0.0)
    gLm = params.get("gamma_L_minus", 0.0)
    gRp = params.get("gamma_R_plus", 0.0)
    gRm = params.get("gamma_R_minus", 0.0)
    gzL = params.get("gamma_z_L", 0.0)
    gzR = params.get("gamma_z_R", 0.0)

    c_ops = []
    if gLp > 0: c_ops.append(np.sqrt(gLp) * sp_L)
    if gLm > 0: c_ops.append(np.sqrt(gLm) * sm_L)
    if gRp > 0: c_ops.append(np.sqrt(gRp) * sp_R)
    if gRm > 0: c_ops.append(np.sqrt(gRm) * sm_R)
    if gzL > 0: c_ops.append(np.sqrt(gzL) * sz_L)
    if gzR > 0: c_ops.append(np.sqrt(gzR) * sz_R)
    return c_ops

def steady_state_from_params(params: dict, method: str = "direct"):
    """Compute the steady-state density matrix for given parameters."""
    H = build_hamiltonian(params)
    c_ops = build_c_ops(params)
    rho_ss = steadystate(H, c_ops, method=method)
    return rho_ss

def evolve_step(H, c_ops, rho, dt):
    """
    Evolve rho for one time step dt under fixed H and c_ops.
    
    Parameters:
    -----------
    H : Qobj
        Hamiltonian operator
    c_ops : list of Qobj
        List of collapse operators
    rho : Qobj
        Initial density matrix (or ket state)
    dt : float
        Time step duration
    method : str
        Solver method ("adams", "bdf", etc.)
    
    Returns:
    --------
    Qobj
        Evolved state at time dt
    """
    tlist = np.array([0.0, dt])
    result = mesolve(H, rho, tlist, c_ops=c_ops)
    return result.states[-1]