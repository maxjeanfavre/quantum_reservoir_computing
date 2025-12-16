# Warning filter set in quantum/__init__.py, but set here too for safety
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='numpy._core.getlimits')

import numpy as np

def drive_encoding_1d(u_t):
    """
    Encode scalar input u_t into system params by modulating drive fields.
    
    Parameters:
    -----------
    u_t : float
        Scalar input at time t
    
    Returns:
    --------
    dict
        Parameter dictionary for quantum system
    """
    return {
        "eps_L": 1.0,
        "eps_R": 1.0,
        "U": 0.02,
        "g_res": 0.01,
        "g_off": 0.01,
        "f_L": 0.1 * u_t,
        "f_R": 0.1 * u_t,
        "gamma_L_plus": 0.01,
        "gamma_L_minus": 0.05,
        "gamma_R_plus": 0.05,
        "gamma_R_minus": 0.01,
        "gamma_z_L": 0.01,
        "gamma_z_R": 0.01,
    }

def bare_encoding_1d(u_t):
    """
    Encode scalar input u_t by modulating energy levels.
    """
    return {
        "eps_L": 1.0 * (u_t),
        "eps_R": 0.5 * (-u_t),
        "U": 0.0,
        "g_res": 0.01,
        "g_off": 0.0,
        "f_L": 0.0,
        "f_R": 0.0,
        "gamma_L_plus": 0.01,
        "gamma_L_minus": 0.05,
        "gamma_R_plus": 0.05,
        "gamma_R_minus": 0.01,
        "gamma_z_L": 0.01,
        "gamma_z_R": 0.01,
    }

def bare_asymmetric_encoding_1d(u_t):
    """
    Encode scalar input u_t by modulating energy levels asymmetrically.
    """
    return {
        "eps_L": 1.0 * (1.0 + u_t),
        "eps_R": 1.0 * (1.0 - u_t),  # Opposite modulation
        "U": 0.0,
        "g_res": 0.01,
        "g_off": 0.0,
        "f_L": 0.0,
        "f_R": 0.0,
        "gamma_L_plus": 0.01,
        "gamma_L_minus": 0.05,
        "gamma_R_plus": 0.05,
        "gamma_R_minus": 0.01,
        "gamma_z_L": 0.01,
        "gamma_z_R": 0.01,
    }

def gamma_encoding_1d(u_t):
    """
    Encode scalar input u_t by modulating dissipation rates.
    """
    return {
        "eps_L": 1.0,
        "eps_R": 1.0,
        "U": 0.0,
        "g_res": 0.01,
        "g_off": 0.0,
        "f_L": 0.0,
        "f_R": 0.0,
        "gamma_L_plus": 0.01,
        "gamma_L_minus": 0.05 * (1.5 + u_t),
        "gamma_R_plus": 0.05 * (1.5 + u_t),
        "gamma_R_minus": 0.01,
        "gamma_z_L": 0.01,
        "gamma_z_R": 0.01,
    }

def dephase_encoding_1d(u_t):
    """
    Encode scalar input u_t by modulating dephasing rates.
    """
    return {
        "eps_L": 1.0,
        "eps_R": 1.0,
        "U": 0.0,
        "g_res": 0.01,
        "g_off": 0.0,
        "f_L": 0.0,
        "f_R": 0.0,
        "gamma_L_plus": 0.01,
        "gamma_L_minus": 0.05,
        "gamma_R_plus": 0.05,
        "gamma_R_minus": 0.01,
        "gamma_z_L": 0.01 + 0.01 * u_t,
        "gamma_z_R": 0.01 + 0.01 * u_t,
    }

def rich_encoding_1d(u_t):
    """
    Encode scalar input u_t by modulating multiple parameters simultaneously.
    """
    return {
        "eps_L": 0.1 * (1.0 + u_t),
        "eps_R": 0.1 * (1.0 - u_t),
        "U": 0.02,
        "g_res": 0.01 * (1.0 + u_t),
        "g_off": 0.01 * (1.0 - u_t),
        "f_L": 0.1 * u_t,
        "f_R": 0.05 * u_t,
        "gamma_L_plus": 0.01,
        "gamma_L_minus": 0.05 * (1.5 + u_t),
        "gamma_R_plus": 0.05 * (1.5 - u_t),
        "gamma_R_minus": 0.01,
        "gamma_z_L": 0.01 + 0.01 * u_t,
        "gamma_z_R": 0.01 - 0.01 * u_t,
    }

