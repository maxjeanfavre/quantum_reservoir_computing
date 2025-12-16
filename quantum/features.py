# Warning filter set in quantum/__init__.py, but set here too for safety
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='numpy._core.getlimits')

import numpy as np
from qutip import expect
from .quantum_system import (
    sx_L, sy_L, sz_L, sx_R, sy_R, sz_R,
    sxL_sxR, sxL_syR, sxL_szR,
    syL_sxR, syL_syR, syL_szR,
    szL_sxR, szL_syR, szL_szR,
)

def rho_to_features(rho):
    """
    Extract the 15 independant real parameters of a 2-qubit density matrix:

        r00, r01, r10,
        alpha_R, alpha_I,
        beta_R,  beta_I,
        v_R, v_I,
        x_R, x_I,
        y_R, y_I,
        z_R, z_I

    Consistent naming: *_R = real part, *_I = imaginary part.
    """

    # Convert Qobj to a standard NumPy array
    M = rho.full()

    # --- Diagonal entries ---
    r00 = float(np.real(M[0, 0]))
    r01 = float(np.real(M[1, 1]))
    r10 = float(np.real(M[2, 2]))
    r11 = float(np.real(M[3, 3]))

    # --- Off-diagonal entries ---
    v    = M[0, 1]
    x    = M[0, 2]
    beta = M[0, 3]
    alpha = M[1, 2]
    y    = M[1, 3]
    z    = M[2, 3]

    v_R,    v_I    = np.real(v),    np.imag(v)
    x_R,    x_I    = np.real(x),    np.imag(x)
    beta_R, beta_I = np.real(beta), np.imag(beta)
    alpha_R,alpha_I= np.real(alpha),np.imag(alpha)
    y_R,    y_I    = np.real(y),    np.imag(y)
    z_R,    z_I    = np.real(z),    np.imag(z)

    return np.array([
        r00, r01, r10, #r11,
        alpha_R, alpha_I,
        beta_R,  beta_I,
        v_R, v_I,
        x_R, x_I,
        y_R, y_I,
        z_R, z_I
    ], dtype=float)

def rho_to_populations(rho):
    """
    Extract the 3 independant real populations of a 2-qubit density matrix:
        r00, r01, r10,
    """

    # Convert Qobj to a standard NumPy array
    M = rho.full()

    # --- Diagonal entries ---
    r00 = float(np.real(M[0, 0]))
    r01 = float(np.real(M[1, 1]))
    r10 = float(np.real(M[2, 2]))
    
    return np.array([
        r00, r01, r10, #r11,
    ], dtype=float)


def rho_to_pauli_features(rho):
    return np.array([
        expect(sz_L, rho),
        expect(sz_R, rho),
        expect(sx_L, rho),
        expect(sx_R, rho),
        expect(szL_szR, rho),
        expect(sxL_sxR, rho),
    ], dtype=float)


