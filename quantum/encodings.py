# Warning filter set in quantum/__init__.py, but set here too for safety
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='numpy._core.getlimits')

import numpy as np

def bare_encoding_2d(u):
    """Simple encoding of 2D input (x,y) into Lindblad parameters."""
    x, y = u
    return {
        "eps_L": 1.0 * (1 + x),
        "eps_R": 1.0 * (1 + y),
        "U": 0.02,
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

def simple_bare_gamma_encoding_2d(u):
    """Simple encoding of 2D input (x,y) into Lindblad parameters."""
    x, y = u
    return {
        "eps_L": 1.0 * x,
        "eps_R": 1.0 * y,
        "U": 0.0,
        "g_res": 0.01,
        "g_off": 0.0,
        "f_L": 0.0,
        "f_R": 0.0,
        "gamma_L_plus":   0.02 * (1.5 + x),   # emission left
        "gamma_L_minus":  0.05 * (1.5 + x),   # absorption left
        "gamma_R_plus":   0.05 * (1.5 + y),   # emission right
        "gamma_R_minus":  0.02 * (1.5 + y),   # absorption right
        "gamma_z_L": 0.01,
        "gamma_z_R": 0.01,
    }

def drive_encoding_2d(u):
    """Simple encoding of 2D input (x,y) into Lindblad parameters."""
    x, y = u
    return {
        "eps_L": 0.1,
        "eps_R": 0.1,
        "U": 0.02,
        "g_res": 0.01,
        "g_off": 0.01,
        "f_L": 0.1*x,
        "f_R": 0.1*y,
        "gamma_L_plus": 0.01,
        "gamma_L_minus": 0.05,
        "gamma_R_plus": 0.05,
        "gamma_R_minus": 0.01,
        "gamma_z_L": 0.01,
        "gamma_z_R": 0.01,
    }

# Drive with entanglement
def drive1_encoding_2d(u):
    """Simple encoding of 2D input (x,y) into Lindblad parameters."""
    x, y = u
    return {
        "eps_L": 1.0,
        "eps_R": 1.0,
        "U": 0.01,
        "g_res": 3.3e-3,
        "g_off": 0.01,      
        "f_L": 0.1*x,
        "f_R": 0.1*y,
        "gamma_L_plus":  7.0e-5,   
        "gamma_L_minus": 1.0e-2,  
        "gamma_R_plus":  1.0e-2, 
        "gamma_R_minus": 2.0e-3,  
        "gamma_z_L": 0.0001,
        "gamma_z_R": 0.0001,
    }
# Drive without entanglement
def drive2_encoding_2d(u):
    """Simple encoding of 2D input (x,y) into Lindblad parameters."""
    x, y = u
    return {
        "eps_L": 1.0,
        "eps_R": 1.0,
        "U": 0.01,
        "g_res": 1.0e-3,
        "g_off": 0.01,      
        "f_L": 0.1*x,
        "f_R": 0.1*y,
        "gamma_L_plus":  7.0e-5,   
        "gamma_L_minus": 1.0e-2,  
        "gamma_R_plus":  1.0e-2, 
        "gamma_R_minus": 5.0e-3,  
        "gamma_z_L": 0.0001,
        "gamma_z_R": 0.0001,
    }


def drive_gamma_encoding_2d(d):
    def encoding(u):
        x, y = u
        return {
            "eps_L": 0.1,
            "eps_R": 0.1,
            "U": 0.02,
            "g_res": 0.01,
            "g_off": 0.01,
            "f_L": 0.1*x,
            "f_R": 0.1*y,
            "gamma_L_plus": 0.01,
            "gamma_L_minus": 0.05,
            "gamma_R_plus": 0.05,
            "gamma_R_minus": 0.01,
            "gamma_z_L": d,
            "gamma_z_R": d,
        }
    return encoding



def couplings_encoding_2d(u):
    """Simple encoding of 2D input (x,y) into Lindblad parameters."""
    x, y = u
    return {
        "eps_L": 0.1,
        "eps_R": 0.1,
        "U": 0.02,
        "g_res": 0.01*x,
        "g_off": 0.01*y,
        "f_L": 0.1,
        "f_R": 0.1,
        "gamma_L_plus": 0.01,
        "gamma_L_minus": 0.05,
        "gamma_R_plus": 0.05,
        "gamma_R_minus": 0.01,
        "gamma_z_L": 0.01,
        "gamma_z_R": 0.01,
    }

def gamma_encoding_2d(u):
    """Simple encoding of 2D input (x,y) into Lindblad parameters."""
    x, y = u
    return {
        "eps_L": 0.1,
        "eps_R": 0.1,
        "U": 0.02,
        "g_res": 0.01,
        "g_off": 0.01,
        "f_L": 0.1,
        "f_R": 0.1,
        "gamma_L_plus": 0.01,
        "gamma_L_minus": 0.05 + 0.05*x,
        "gamma_R_plus": 0.05,
        "gamma_R_minus": 0.01 + 0.01*y,
        "gamma_z_L": 0.01,
        "gamma_z_R": 0.01,
    }

# With entanglement
def gamma1_encoding_2d(u):
    """Simple encoding of 2D input (x,y) into Lindblad parameters."""
    x, y = u
    return {
        "eps_L": 1.0,
        "eps_R": 1.0,
        "U": 0.01,
        "g_res": 3.3e-3,
        "g_off": 0.01,      
        "f_L": 0.05,
        "f_R": 0.05,
        "gamma_L_plus":  (7.0 + x)*10**(-5),   
        "gamma_L_minus": 1.0e-2,  
        "gamma_R_plus":  1.0e-2, 
        "gamma_R_minus": (2.0 + y)*10**(-3),  
        "gamma_z_L": 0.0001,
        "gamma_z_R": 0.0001,
    }

# Without entanglement
def gamma2_encoding_2d(u):
    """Simple encoding of 2D input (x,y) into Lindblad parameters."""
    x, y = u
    return {
        "eps_L": 1.0,
        "eps_R": 1.0,
        "U": 0.01,
        "g_res": 1.e-4,
        "g_off": 0.01,      
        "f_L": 0.05,
        "f_R": 0.05,
        "gamma_L_plus":  (7.0 + x)*10**(-5),   
        "gamma_L_minus": 1.0e-2,  
        "gamma_R_plus":  1.0e-2, 
        "gamma_R_minus": (2.0 + y)*10**(-3),  
        "gamma_z_L": 0.0001,
        "gamma_z_R": 0.0001,
    }

"""
def gamma2_encoding_2d(u):
    x, y = u
    return {
        "eps_L": 0.1,
        "eps_R": 0.1,
        "U": 0.02,
        "g_res": 0.01,
        "g_off": 0.01,

        "f_L": 0.1,
        "f_R": 0.1,

        # Cross-dependence: left depends on y, right depends on x
        "gamma_L_plus":   0.02 + 0.02*y,   # emission left
        "gamma_L_minus":  0.05 + 0.04*y,   # absorption left
        "gamma_R_plus":   0.02 + 0.02*x,   # emission right
        "gamma_R_minus":  0.05 + 0.04*x,   # absorption right

        # No pure dephasing
        "gamma_z_L": 0.01,
        "gamma_z_R": 0.01,
    }
"""

def dephase_encoding_2d(u):
    """Simple encoding of 2D input (x,y) into Lindblad parameters."""
    x, y = u
    return {
        "eps_L": 0.1,
        "eps_R": 0.1,
        "U": 0.02,
        "g_res": 0.01,
        "g_off": 0.01,
        "f_L": 0.1,
        "f_R": 0.1,
        "gamma_L_plus": 0.01,
        "gamma_L_minus": 0.05,
        "gamma_R_plus": 0.05,
        "gamma_R_minus": 0.01,
        "gamma_z_L": 0.01 + 0.01*x,
        "gamma_z_R": 0.01 + 0.01*y,
    }

def rich1_encoding_2d(u):
    """Simple encoding of 2D input (x,y) into Lindblad parameters."""
    x, y = u
    return {
        "eps_L": 0.1*x,
        "eps_R": 0.1*y,
        "U": 0.02,
        "g_res": 0.01*x,
        "g_off": 0.01*y,
        "f_L": 0.1,
        "f_R": 0.1,
        "gamma_L_plus":   0.02 * (1.5 + y),   # emission left
        "gamma_L_minus":  0.05 * (1.5 + y),   # absorption left
        "gamma_R_plus":   0.05 * (1.5 + x),   # emission right
        "gamma_R_minus":  0.02 * (1.5 + x),   # absorption right
        "gamma_z_L": 0.01 + 0.01*x,
        "gamma_z_R": 0.01 + 0.01*y,
    }

def rich2_encoding_2d(u):
    """Simple encoding of 2D input (x,y) into Lindblad parameters."""
    x, y = u
    return {
        "eps_L": 0.1*x,
        "eps_R": 0.1*y,
        "U": 0.02,
        "g_res": 0.01,
        "g_off": 0.01,
        "f_L": 0.1,
        "f_R": 0.1,
        "gamma_L_plus":   0.02 * (1.5 + y),   # emission left
        "gamma_L_minus":  0.05 * (1.5 + y),   # absorption left
        "gamma_R_plus":   0.05 * (1.5 + x),   # emission right
        "gamma_R_minus":  0.02 * (1.5 + x),   # absorption right
        "gamma_z_L": 0.01,
        "gamma_z_R": 0.01,
    }

def rich3_encoding_2d(u):
    """Simple encoding of 2D input (x,y) into Lindblad parameters."""
    x, y = u
    return {
        "eps_L": 0.1*x,
        "eps_R": 0.1*y,
        "U": 0.02,
        "g_res": 0.01*x,
        "g_off": 0.01*y,
        "f_L": 0.1,
        "f_R": 0.1,
        "gamma_L_plus":   0.02 * (1.5 + y),   # emission left
        "gamma_L_minus":  0.05 * (1.5 + y),   # absorption left
        "gamma_R_plus":   0.05 * (1.5 + x),   # emission right
        "gamma_R_minus":  0.02 * (1.5 + x),   # absorption right
        "gamma_z_L": 0.01,
        "gamma_z_R": 0.01,
    }

def rich4_encoding_2d(u):
    """Simple encoding of 2D input (x,y) into Lindblad parameters."""
    x, y = u
    return {
        "eps_L": 0.1*x,
        "eps_R": 0.1*y,
        "U": 0.02,
        "g_res": 0.01*x,
        "g_off": 0.01*y,
        "f_L": 0.1*x,
        "f_R": 0.1*y,
        "gamma_L_plus":   0.02 * (1.5 + y),   # emission left
        "gamma_L_minus":  0.05 * (1.5 + y),   # absorption left
        "gamma_R_plus":   0.05 * (1.5 + x),   # emission right
        "gamma_R_minus":  0.02 * (1.5 + x),   # absorption right
        "gamma_z_L": 0.01 + 0.01*x,
        "gamma_z_R": 0.01 + 0.01*y,
    }

def bare_drive_encoding_2d(u):
    """Simple encoding of 2D input (x,y) into Lindblad parameters."""
    x, y = u
    return {
        "eps_L": 0.1*x,
        "eps_R": 0.1*y,
        "U": 0.02,
        "g_res": 0.01,
        "g_off": 0.01,
        "f_L": 0.1*x,
        "f_R": 0.1*y,
        "gamma_L_plus": 0.01,
        "gamma_L_minus": 0.05,
        "gamma_R_plus": 0.05,
        "gamma_R_minus": 0.01,
        "gamma_z_L": 0.01,
        "gamma_z_R": 0.01,
    }
