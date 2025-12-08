"""
Functions for calculating boron carbon speciation.
"""

import scipy.optimize as opt
import numpy as np
from .helpers import maxShape
from .uncertainties import negative_log10_preserve_type, _has_uncertainties, uncertainty_propagation_decorator, _zero_finder_with_uncertainties

from .boron import calc_remaining_B_species

from typing import Dict, Tuple, Callable, Union, List, Any
from .dataclasses import KValues, CBsystData

# Function types
# Zero-finders: 2-5, 10-15
# Algebraic: 1, 6-9

# Wrappers for zero-finders and uncertainty handling

def _zero_wrapper(ps: np.ndarray, fn: Callable, bounds: Tuple[float, float] = (10 ** -14, 10 ** -1)) -> float:
    """
    Simplified wrapper to handle zero finders with uncertainty propagation.
    
    This is a cleaner version that uses the zero_finder_decorator infrastructure
    but works with np.apply_along_axis. Handles both deterministic and uncertain
    parameters for root finding operations.
    
    Args:
        ps: Array of parameters to pass to the function.
        fn: Function for which to find the root.
        bounds: Bounds for the root finding algorithm (lower, upper).
        
    Returns:
        Root of the function within the specified bounds.
        
    Raises:
        ValueError: If root finding fails with standard methods.
    """
    # Check if any parameters have uncertainties
    has_uncertainties = any(_has_uncertainties(p) for p in ps if p is not None)
    
    if not has_uncertainties:
        # No uncertainties - use original implementation
        return opt.fsolve(fn, 1e-5, args=tuple(ps))[0]
    else:
        # Has uncertainties - use the enhanced finite difference method
        return _zero_finder_with_uncertainties(ps, fn, bounds)

# Solving logic

def solve_with_broadcasting(params, solver_fn):
    """Generic solver using numpy broadcasting."""
    # Broadcast all parameters
    broadcasted = np.broadcast_arrays(*[np.asarray(p) for p in params])
    
    # Stack for apply_along_axis
    param_stack = np.stack(broadcasted)
    
    # Apply solver
    result = np.apply_along_axis(
        lambda p: _zero_wrapper(p, solver_fn), 
        0, param_stack
    )
    
    return result.item() if result.ndim == 0 else result

def solve_with_vectorization(params, solver_fn):
    """
    Vectorized solver using numpy.vectorize for better performance.
    
    This approach is cleaner than the broadcasting/stacking method and
    handles both scalar and array inputs more efficiently.
    
    Args:
        params: Sequence of parameters (scalars or arrays) to broadcast.
        solver_fn: Function that takes individual parameter values.
        
    Returns:
        Result array or scalar depending on input shapes.
    """
    # Create vectorized wrapper - don't specify otypes to allow uncertainty objects
    vectorized_solver = np.vectorize(
        lambda *args: _zero_wrapper(list(args), solver_fn)
    )
    
    # Apply with automatic broadcasting
    result = vectorized_solver(*params)
    
    # Return scalar if input was scalar
    return result.item() if result.ndim == 0 else result

solve_function = solve_with_vectorization

# Calculation Functions

# Zeebe & Wolf-Gladrow, Appendix B
# 1. CO2 and pH given
def CO2_pH(CO2: Union[float, np.ndarray], pH: Union[float, np.ndarray], Ks: Union[KValues,dict]) -> Union[float, np.ndarray]:
    """
    Calculate dissolved inorganic carbon (DIC) from CO2 and pH.
    
    Based on carbonate system equilibria from Zeebe & Wolf-Gladrow, Appendix B.
    Uses the relationship between CO2, pH, and the carbonate equilibrium constants
    to determine total dissolved inorganic carbon.
    
    Args:
        CO2: Dissolved CO2 concentration in μmol/kg.
        pH: pH value (total scale).
        Ks: Equilibrium constants data structure.
        
    Returns:
        DIC: Dissolved inorganic carbon concentration in μmol/kg.
        
    Example:
        >>> CO2_pH(10.0, 8.1, Ks)
        2000.5
    """
    h = 10.0**-pH
    return CO2 * (1 + Ks.K1 / h + Ks.K1 * Ks.K2 / h ** 2)

def CO2_H(CO2: Union[float, np.ndarray], H: Union[float, np.ndarray], Ks: Union[KValues,dict]) -> Union[float, np.ndarray]:
    """
    Calculate dissolved inorganic carbon (DIC) from CO2 and H.
    
    Based on carbonate system equilibria from Zeebe & Wolf-Gladrow, Appendix B.
    Uses the relationship between CO2, H, and the carbonate equilibrium constants
    to determine total dissolved inorganic carbon.
    
    Args:
        CO2: Dissolved CO2 concentration in μmol/kg.
        H: Hydrogen ion concentration in mol/kg (total scale).
        Ks: Equilibrium constants data structure.
        
    Returns:
        DIC: Dissolved inorganic carbon concentration in μmol/kg.
        
    Example:
        >>> CO2_H(10.0, 10**-8.1, Ks)
        2000.5
    """
    return CO2 * (1 + Ks.K1 / H + Ks.K1 * Ks.K2 / H ** 2)

# 2. CO2 and HCO3 given
def CO2_HCO3(CO2: Union[float, np.ndarray], HCO3: Union[float, np.ndarray], Ks: Union[KValues,dict]) -> Union[float, np.ndarray]:
    """
    Calculate hydrogen ion concentration from CO2 and bicarbonate.
    
    Solves the carbonate system equilibrium to find [H+] when CO2 and HCO3-
    concentrations are known. Uses numerical root finding to solve the
    non-linear equilibrium equations.
    
    Args:
        CO2: Dissolved CO2 concentration in μmol/kg.
        HCO3: Bicarbonate ion concentration in μmol/kg.
        Ks: Equilibrium constants data structure.
        
    Returns:
        H: Hydrogen ion concentration in mol/kg (total scale).
        
    Note:
        This function uses the _zero_wrapper for robust root finding
        that handles both deterministic and uncertain parameters.
    """
    
    return solve_function(
        (CO2, HCO3, Ks.K1, Ks.K2), 
        zero_CO2_HCO3
    )
    # # Don't strip uncertainties - let _zero_wrapper handle them
    # par = cast_array(CO2, HCO3, Ks.K1, Ks.K2)  # cast parameters into array
    # shape = maxShape(CO2, HCO3, Ks.K1, Ks.K2)  # get shape of output

    # result = np.apply_along_axis(_zero_wrapper, 0, par, fn=zero_CO2_HCO3).reshape(shape)
    
    # # If result is a single-element array, extract the scalar
    # if result.size == 1:
    #     return result.item()
    # return result


def zero_CO2_HCO3(h: float, CO2: float, HCO3: float, K1: float, K2: float) -> float:
    """
    Root finding function for CO2-HCO3 carbonate system equilibrium.
    
    This function defines the equation to be solved when finding [H+] from
    CO2 and HCO3- concentrations. Returns zero when the carbonate system
    is in equilibrium.
    
    Args:
        h: Hydrogen ion concentration in mol/kg (trial value).
        CO2: Dissolved CO2 concentration in μmol/kg.
        HCO3: Bicarbonate ion concentration in μmol/kg.
        K1: First carbonic acid dissociation constant.
        K2: Second carbonic acid dissociation constant.
        
    Returns:
        Residual of the equilibrium equation (should be zero at equilibrium).
        
    Note:
        Roots: two negative, one positive - the positive root is used.
    """
    LH = CO2 * (h ** 2 + K1 * h + K1 * K2)
    RH = HCO3 * (h ** 2 + h ** 3 / K1 + K2 * h)
    return LH - RH


# 3. CO2 and CO3
def CO2_CO3(CO2: Union[float, np.ndarray], CO3: Union[float, np.ndarray], Ks: Union[KValues,dict]) -> Union[float, np.ndarray]:
    """
    Calculate hydrogen ion concentration from CO2 and carbonate.
    
    Solves the carbonate system equilibrium to find [H+] when CO2 and CO3^2-
    concentrations are known. Uses numerical root finding to solve the
    non-linear equilibrium equations.
    
    Args:
        CO2: Dissolved CO2 concentration in μmol/kg.
        CO3: Carbonate ion concentration in μmol/kg.
        Ks: Equilibrium constants data structure.
        
    Returns:
        H: Hydrogen ion concentration in mol/kg (total scale).
        
    Note:
        This function uses the _zero_wrapper for robust root finding
        that handles both deterministic and uncertain parameters.
    """
    
    return solve_function(
        (CO2, CO3, Ks.K1, Ks.K2), 
        zero_CO2_CO3
    )
    
    # par = cast_array(CO2, CO3, Ks.K1, Ks.K2)  # cast parameters into array
    # shape = maxShape(CO2, CO3, Ks.K1, Ks.K2)  # get shape of output

    # result = np.apply_along_axis(_zero_wrapper, 0, par, fn=zero_CO2_CO3).reshape(shape)
    
    # # If result is a single-element array, extract the scalar
    # if result.size == 1:
    #     return result.item()
    # return result


def zero_CO2_CO3(h: float, CO2: float, CO3: float, K1: float, K2: float) -> float:
    """
    Root finding function for CO2-CO3 carbonate system equilibrium.
    
    This function defines the equation to be solved when finding [H+] from
    CO2 and CO3^2- concentrations. Returns zero when the carbonate system
    is in equilibrium.
    
    Args:
        h: Hydrogen ion concentration in mol/kg (trial value).
        CO2: Dissolved CO2 concentration in μmol/kg.
        CO3: Carbonate ion concentration in μmol/kg.
        K1: First carbonic acid dissociation constant.
        K2: Second carbonic acid dissociation constant.
        
    Returns:
        Residual of the equilibrium equation (should be zero at equilibrium).
        
    Note:
        Roots: one positive, three negative. The positive root is used.
    """
    LH = CO2 * (h ** 2 + K1 * h + K1 * K2)
    RH = CO3 * (h ** 2 + h ** 3 / K2 + h ** 4 / (K1 * K2))
    return LH - RH


# 4. CO2 and TA
@uncertainty_propagation_decorator
def CO2_TA(CO2: Union[float, np.ndarray], TA: Union[float, np.ndarray], BT: Union[float, np.ndarray], 
           PT: Union[float, np.ndarray], SiT: Union[float, np.ndarray], ST: Union[float, np.ndarray], 
           FT: Union[float, np.ndarray], Ks: Union[KValues,dict]) -> Union[float, np.ndarray]:
    """
    Calculate pH from CO2 and total alkalinity using iterative Newton-Raphson method.
    
    Implements the CO2SYS algorithm for solving the carbonate system when CO2
    and total alkalinity are known. Uses iterative convergence to find pH that
    satisfies the alkalinity balance including contributions from carbonate,
    borate, phosphate, silicate, and other acid-base systems.
    
    Args:
        CO2: Dissolved CO2 concentration in μmol/kg.
        TA: Total alkalinity in μmol/kg.
        BT: Total boron concentration in μmol/kg.
        PT: Total phosphate concentration in μmol/kg.
        SiT: Total silicate concentration in μmol/kg.
        ST: Total sulfate concentration in μmol/kg.
        FT: Total fluoride concentration in μmol/kg.
        Ks: Equilibrium constants data structure.
        
    Returns:
        pH: pH value on the total scale.
        
    Note:
        Taken from MATLAB CO2SYS. Uses Newton-Raphson iteration with
        adaptive step size control for robust convergence.
    """
    
    fCO2 = CO2 / Ks.K0
    L = maxShape(TA, CO2, BT, PT, SiT, ST, FT, Ks.K1)
    if len(L) == 0:
        L = (1,)
        
    pHguess = 8.0
    pHtol = 0.0000001
    pHx = np.full(L, pHguess)
    deltapH = np.array(pHtol + 1, ndmin=1)
    ln10 = np.log(10)

    while np.any(abs(deltapH) > pHtol):
        H = 10 ** -pHx
        HCO3 = Ks.K0 * Ks.K1 * fCO2 / H
        CO3 = Ks.K0 * Ks.K1 * Ks.K2 * fCO2 / H ** 2
        CAlk = HCO3 + 2 * CO3
        BAlk = BT * Ks.KB / (Ks.KB + H)
        OH = Ks.KW / H
        PhosTop = Ks.KP1 * Ks.KP2 * H + 2 * Ks.KP1 * Ks.KP2 * Ks.KP3 - H ** 3
        PhosBot = (
            H ** 3 + Ks.KP1 * H ** 2 + Ks.KP1 * Ks.KP2 * H + Ks.KP1 * Ks.KP2 * Ks.KP3
        )
        PAlk = PT * PhosTop / PhosBot
        SiAlk = SiT * Ks.KSi / (Ks.KSi + H)
        # positive
        Hfree = H / (1 + ST / Ks.KS)
        HSO4 = ST / (1 + Ks.KS / Hfree)
        HF = FT / (1 + Ks.KF / Hfree)

        Residual = TA - CAlk - BAlk - OH - PAlk - SiAlk + Hfree + HSO4 + HF
        Slope = ln10 * (HCO3 + 4.0 * CO3 + BAlk * H / (Ks.KB + H) + OH + H)
        deltapH = Residual / Slope

        while np.any(abs(deltapH) > 1):
            FF = abs(deltapH) > 1
            deltapH[FF] = deltapH[FF] / 2

        pHx += deltapH

    return pHx if pHx.size > 1 else pHx.item()


# 5. CO2 and DIC
def CO2_DIC(CO2: Union[float, np.ndarray], DIC: Union[float, np.ndarray], Ks: Union[KValues,dict]) -> Union[float, np.ndarray]:
    """
    Calculate hydrogen ion concentration from CO2 and dissolved inorganic carbon.
    
    Solves the carbonate system equilibrium to find [H+] when CO2 and total
    dissolved inorganic carbon concentrations are known. Uses numerical root
    finding to solve the non-linear equilibrium equations.
    
    Args:
        CO2: Dissolved CO2 concentration in μmol/kg.
        DIC: Dissolved inorganic carbon concentration in μmol/kg.
        Ks: Equilibrium constants data structure.
        
    Returns:
        H: Hydrogen ion concentration in mol/kg (total scale).
        
    Note:
        This function uses the _zero_wrapper for robust root finding
        that handles both deterministic and uncertain parameters.
    """
    
    return solve_function(
        (CO2, DIC, Ks.K1, Ks.K2), 
        zero_CO2_DIC
    )
    
    # par = cast_array(CO2, DIC, Ks.K1, Ks.K2)  # cast parameters into array
    # shape = maxShape(CO2, DIC, Ks.K1, Ks.K2)  # get shape of output

    # result = np.apply_along_axis(_zero_wrapper, 0, par, fn=zero_CO2_DIC).reshape(shape)
    
    # # If result is a single-element array, extract the scalar
    # if result.size == 1:
    #     return result.item()
    # return result


def zero_CO2_DIC(h: float, CO2: float, DIC: float, K1: float, K2: float) -> float:
    """
    Root finding function for CO2-DIC carbonate system equilibrium.
    
    This function defines the equation to be solved when finding [H+] from
    CO2 and total dissolved inorganic carbon concentrations. Returns zero
    when the carbonate system is in equilibrium.
    
    Args:
        h: Hydrogen ion concentration in mol/kg (trial value).
        CO2: Dissolved CO2 concentration in μmol/kg.
        DIC: Dissolved inorganic carbon concentration in μmol/kg.
        K1: First carbonic acid dissociation constant.
        K2: Second carbonic acid dissociation constant.
        
    Returns:
        Residual of the equilibrium equation (should be zero at equilibrium).
        
    Note:
        Roots: one positive, one negative. The positive root is used.
    """
    LH = DIC * h ** 2
    RH = CO2 * (h ** 2 + K1 * h + K1 * K2)
    return LH - RH


# 6. pH and HCO3
def pH_HCO3(pH: Union[float, np.ndarray], HCO3: Union[float, np.ndarray], Ks: Union[KValues,dict]) -> Union[float, np.ndarray]:
    """
    Calculate dissolved inorganic carbon from pH and bicarbonate.
    
    Uses carbonate system equilibria to determine total dissolved inorganic
    carbon when pH and bicarbonate concentration are known. This is an
    algebraic solution that doesn't require iteration.
    
    Args:
        pH: pH value (total scale).
        HCO3: Bicarbonate ion concentration in μmol/kg.
        Ks: Equilibrium constants data structure.
        
    Returns:
        DIC: Dissolved inorganic carbon concentration in μmol/kg.
    """
    h = 10.0**-pH
    return HCO3 * (1 + h / Ks.K1 + Ks.K2 / h)

def H_HCO3(H: Union[float, np.ndarray], HCO3: Union[float, np.ndarray], Ks: Union[KValues,dict]) -> Union[float, np.ndarray]:
    """
    Calculate dissolved inorganic carbon from H and bicarbonate.
    
    Uses carbonate system equilibria to determine total dissolved inorganic
    carbon when H and bicarbonate concentration are known. This is an
    algebraic solution that doesn't require iteration.
    
    Args:
        H: Hydrogen ion concentration in mol/kg.
        HCO3: Bicarbonate ion concentration in μmol/kg.
        Ks: Equilibrium constants data structure.
        
    Returns:
        DIC: Dissolved inorganic carbon concentration in μmol/kg.
    """
    return HCO3 * (1 + H / Ks.K1 + Ks.K2 / H)

# 7. pH and CO3
def pH_CO3(pH: Union[float, np.ndarray], CO3: Union[float, np.ndarray], Ks: Union[KValues,dict]) -> Union[float, np.ndarray]:
    """
    Calculate dissolved inorganic carbon from pH and carbonate.
    
    Uses carbonate system equilibria to determine total dissolved inorganic
    carbon when pH and carbonate concentration are known. This is an
    algebraic solution that doesn't require iteration.
    
    Args:
        pH: pH value (total scale).
        CO3: Carbonate ion concentration in μmol/kg.
        Ks: Equilibrium constants data structure.
        
    Returns:
        DIC: Dissolved inorganic carbon concentration in μmol/kg.
    """
    h = 10.0**-pH
    return CO3 * (1 + h / Ks.K2 + h ** 2 / (Ks.K1 * Ks.K2))

def H_CO3(H: Union[float, np.ndarray], CO3: Union[float, np.ndarray], Ks: Union[KValues,dict]) -> Union[float, np.ndarray]:
    """
    Calculate dissolved inorganic carbon from H and carbonate.
    
    Uses carbonate system equilibria to determine total dissolved inorganic
    carbon when H and carbonate concentration are known. This is an
    algebraic solution that doesn't require iteration.
    
    Args:
        H: Hydrogen ion concentration in mol/kg.
        CO3: Carbonate ion concentration in μmol/kg.
        Ks: Equilibrium constants data structure.
        
    Returns:
        DIC: Dissolved inorganic carbon concentration in μmol/kg.
    """
    return CO3 * (1 + H / Ks.K2 + H ** 2 / (Ks.K1 * Ks.K2))

# 8. pH and TA
def pH_TA(pH: Union[float, np.ndarray], TA: Union[float, np.ndarray], BT: Union[float, np.ndarray], 
          PT: Union[float, np.ndarray], SiT: Union[float, np.ndarray], ST: Union[float, np.ndarray], 
          FT: Union[float, np.ndarray], Ks: Union[KValues,dict]) -> Union[float, np.ndarray]:
    """
    Calculate dissolved inorganic carbon from pH and total alkalinity.
    
    Uses the alkalinity balance equation to determine total dissolved inorganic
    carbon when pH and total alkalinity are known. Calculates all alkalinity
    contributions and solves for the carbonate alkalinity component.
    
    Args:
        pH: pH value (total scale).
        TA: Total alkalinity in μmol/kg.
        BT: Total boron concentration in μmol/kg.
        PT: Total phosphate concentration in μmol/kg.
        SiT: Total silicate concentration in μmol/kg.
        ST: Total sulfate concentration in μmol/kg.
        FT: Total fluoride concentration in μmol/kg.
        Ks: Equilibrium constants data structure.
        
    Returns:
        DIC: Dissolved inorganic carbon concentration in μmol/kg.
        
    Note:
        Taken directly from MATLAB CO2SYS.
    """
    H = 10 ** -pH
    # negative alk
    BAlk = BT * Ks.KB / (Ks.KB + H)
    OH = Ks.KW / H
    PhosTop = Ks.KP1 * Ks.KP2 * H + 2 * Ks.KP1 * Ks.KP2 * Ks.KP3 - H ** 3
    PhosBot = H ** 3 + Ks.KP1 * H ** 2 + Ks.KP1 * Ks.KP2 * H + Ks.KP1 * Ks.KP2 * Ks.KP3
    PAlk = PT * PhosTop / PhosBot
    SiAlk = SiT * Ks.KSi / (Ks.KSi + H)
    # positive alk
    Hfree = H / (1 + ST / Ks.KS)
    HSO4 = ST / (1 + Ks.KS / Hfree)
    HF = FT / (1 + Ks.KF / Hfree)
    CAlk = TA - BAlk - OH - PAlk - SiAlk + Hfree + HSO4 + HF

    return CAlk * (H ** 2 + Ks.K1 * H + Ks.K1 * Ks.K2) / (Ks.K1 * (H + 2.0 * Ks.K2))

def H_TA(H: Union[float, np.ndarray], TA: Union[float, np.ndarray], BT: Union[float, np.ndarray], 
         PT: Union[float, np.ndarray], SiT: Union[float, np.ndarray], ST: Union[float, np.ndarray], 
         FT: Union[float, np.ndarray], Ks: Union[KValues,dict]) -> Union[float, np.ndarray]:
    """
    Calculate dissolved inorganic carbon from H and total alkalinity.

    Uses the alkalinity balance equation to determine total dissolved inorganic
    carbon when H and total alkalinity are known. Calculates all alkalinity
    contributions and solves for the carbonate alkalinity component.
    
    Args:
        H: Hydrogen ion concentration in mol/kg.
        TA: Total alkalinity in μmol/kg.
        BT: Total boron concentration in μmol/kg.
        PT: Total phosphate concentration in μmol/kg.
        SiT: Total silicate concentration in μmol/kg.
        ST: Total sulfate concentration in μmol/kg.
        FT: Total fluoride concentration in μmol/kg.
        Ks: Equilibrium constants data structure.
        
    Returns:
        DIC: Dissolved inorganic carbon concentration in μmol/kg.
    """
    # negative alk
    BAlk = BT * Ks.KB / (Ks.KB + H)
    OH = Ks.KW / H
    PhosTop = Ks.KP1 * Ks.KP2 * H + 2 * Ks.KP1 * Ks.KP2 * Ks.KP3 - H ** 3
    PhosBot = H ** 3 + Ks.KP1 * H ** 2 + Ks.KP1 * Ks.KP2 * H + Ks.KP1 * Ks.KP2 * Ks.KP3
    PAlk = PT * PhosTop / PhosBot
    SiAlk = SiT * Ks.KSi / (Ks.KSi + H)
    # positive alk
    Hfree = H / (1 + ST / Ks.KS)
    HSO4 = ST / (1 + Ks.KS / Hfree)
    HF = FT / (1 + Ks.KF / Hfree)
    CAlk = TA - BAlk - OH - PAlk - SiAlk + Hfree + HSO4 + HF

    return CAlk * (H ** 2 + Ks.K1 * H + Ks.K1 * Ks.K2) / (Ks.K1 * (H + 2.0 * Ks.K2))

# 9. pH and DIC
def pH_DIC(pH: Union[float, np.ndarray], DIC: Union[float, np.ndarray], Ks: Union[KValues,dict]) -> Union[float, np.ndarray]:
    """
    Calculate dissolved CO2 from pH and dissolved inorganic carbon.
    
    Uses carbonate system equilibria to determine dissolved CO2 concentration
    when pH and total dissolved inorganic carbon are known. This is an
    algebraic solution that doesn't require iteration.
    
    Args:
        pH: pH value (total scale).
        DIC: Dissolved inorganic carbon concentration in μmol/kg.
        Ks: Equilibrium constants data structure.
        
    Returns:
        CO2: Dissolved CO2 concentration in μmol/kg.
    """
    h = 10.0**-pH
    return DIC / (1 + Ks.K1 / h + Ks.K1 * Ks.K2 / h ** 2)

def H_DIC(H: Union[float, np.ndarray], DIC: Union[float, np.ndarray], Ks: Union[KValues,dict]) -> Union[float, np.ndarray]:
    """
    Calculate dissolved CO2 from H and dissolved inorganic carbon.
    
    Uses carbonate system equilibria to determine dissolved CO2 concentration
    when H and total dissolved inorganic carbon are known. This is an
    algebraic solution that doesn't require iteration.

    Args:
        H: Hydrogen ion concentration in mol/kg.
        DIC: Dissolved inorganic carbon concentration in μmol/kg.
        Ks: Equilibrium constants data structure.

    Returns:
        CO2: Dissolved CO2 concentration in μmol/kg.
    """
    return DIC / (1 + Ks.K1 / H + Ks.K1 * Ks.K2 / H ** 2)

# 10. HCO3 and CO3
def HCO3_CO3(HCO3: Union[float, np.ndarray], CO3: Union[float, np.ndarray], Ks: Union[KValues,dict]) -> Union[float, np.ndarray]:
    """
    Calculate hydrogen ion concentration from bicarbonate and carbonate.
    
    Solves the carbonate system equilibrium to find [H+] when HCO3- and CO3^2-
    concentrations are known. Uses numerical root finding to solve the
    non-linear equilibrium equations.
    
    Args:
        HCO3: Bicarbonate ion concentration in μmol/kg.
        CO3: Carbonate ion concentration in μmol/kg.
        Ks: Equilibrium constants data structure.
        
    Returns:
        H: Hydrogen ion concentration in mol/kg (total scale).
        
    Note:
        This function uses the _zero_wrapper for robust root finding
        that handles both deterministic and uncertain parameters.
    """
    
    return solve_function(
        (HCO3, CO3, Ks.K1, Ks.K2), 
        zero_HCO3_CO3
    )
    
    # par = cast_array(HCO3, CO3, Ks.K1, Ks.K2)  # cast parameters into array
    # shape = maxShape(HCO3, CO3, Ks.K1, Ks.K2)  # get shape of output
    
    # result = np.apply_along_axis(_zero_wrapper, 0, par, fn=zero_HCO3_CO3).reshape(shape)
    
    # # If result is a single-element array, extract the scalar
    # if result.size == 1:
    #     return result.item()
    # return result


def zero_HCO3_CO3(h: float, HCO3: float, CO3: float, K1: float, K2: float) -> float:
    """
    Root finding function for HCO3-CO3 carbonate system equilibrium.
    
    This function defines the equation to be solved when finding [H+] from
    HCO3- and CO3^2- concentrations. Returns zero when the carbonate system
    is in equilibrium.
    
    Args:
        h: Hydrogen ion concentration in mol/kg (trial value).
        HCO3: Bicarbonate ion concentration in μmol/kg.
        CO3: Carbonate ion concentration in μmol/kg.
        K1: First carbonic acid dissociation constant.
        K2: Second carbonic acid dissociation constant.
        
    Returns:
        Residual of the equilibrium equation (should be zero at equilibrium).
        
    Note:
        Roots: one positive, two negative. The positive root is used.
    """
    LH = HCO3 * (h + h ** 2 / K1 + K2)
    RH = CO3 * (h + h ** 2 / K2 + h ** 3 / (K1 * K2))
    return LH - RH


# 11. HCO3 and TA
@uncertainty_propagation_decorator
def HCO3_TA(HCO3: Union[float, np.ndarray], TA: Union[float, np.ndarray], BT: Union[float, np.ndarray], 
            Ks: Union[KValues,dict]) -> Union[float, np.ndarray]:
    """
    Calculate hydrogen ion concentration from bicarbonate and total alkalinity.
    
    Solves the carbonate system equilibrium to find [H+] when HCO3- and total
    alkalinity are known. Uses numerical root finding to solve the non-linear
    alkalinity balance equation. Note that this simplified version only
    considers carbon and boron alkalinity contributions.
    
    Args:
        HCO3: Bicarbonate ion concentration in μmol/kg.
        TA: Total alkalinity in μmol/kg.
        BT: Total boron concentration in μmol/kg.
        Ks: Equilibrium constants data structure.
        
    Returns:
        H: Hydrogen ion concentration in mol/kg (total scale).
        
    Warning:
        Nutrient alkalinity not implemented for this input combination.
        Calculations use only C and B alkalinity.
    """
    
    return solve_function(
        (HCO3, TA, BT, Ks.K1, Ks.K2, Ks.KB, Ks.KW), 
        zero_HCO3_TA
    )
    
    # par = cast_array(
    #     HCO3, TA, BT, Ks.K1, Ks.K2, Ks.KB, Ks.KW
    # )  # cast parameters into array
    # shape = maxShape(HCO3, TA, BT, Ks.K1, Ks.K2, Ks.KB, Ks.KW)  # get shape of output

    # result = np.apply_along_axis(_zero_wrapper, 0, par, fn=zero_HCO3_TA).reshape(shape)
    
    # # If result is a single-element array, extract the scalar
    # if result.size == 1:
    #     return result.item()
    # return result


def zero_HCO3_TA(h: float, HCO3: float, TA: float, BT: float, K1: float, K2: float, KB: float, KW: float) -> float:
    """
    Root finding function for HCO3-TA carbonate system equilibrium.
    
    This function defines the equation to be solved when finding [H+] from
    HCO3- and total alkalinity concentrations. Returns zero when the carbonate
    system alkalinity balance is satisfied (simplified for C and B alkalinity only).
    
    Args:
        h: Hydrogen ion concentration in mol/kg (trial value).
        HCO3: Bicarbonate ion concentration in μmol/kg.
        TA: Total alkalinity in μmol/kg.
        BT: Total boron concentration in μmol/kg.
        K1: First carbonic acid dissociation constant.
        K2: Second carbonic acid dissociation constant.
        KB: Boron dissociation constant.
        KW: Water dissociation constant.
        
    Returns:
        Residual of the alkalinity balance equation (should be zero at equilibrium).
        
    Note:
        Roots: one positive, four negative. The positive root is used.
    """
    LH = TA * (KB + h) * (h ** 3 + K1 * h ** 2 + K1 * K2 * h)
    RH = (
        HCO3
        * (h + h ** 2 / K1 + K2)
        * ((KB + 2 * K2) * K1 * h + 2 * KB * K1 * K2 + K1 * h ** 2)
    ) + (
        (h ** 2 + K1 * h + K1 * K2)
        * (KB * BT * h + KW * KB + KW * h - KB * h ** 2 - h ** 3)
    )
    return LH - RH


# 12. HCO3 and DIC
def HCO3_DIC(HCO3: Union[float, np.ndarray], DIC: Union[float, np.ndarray], Ks: Union[KValues,dict]) -> Union[float, np.ndarray]:
    """
    Calculate hydrogen ion concentration from bicarbonate and dissolved inorganic carbon.
    
    Solves the carbonate system equilibrium to find [H+] when HCO3- and total
    dissolved inorganic carbon concentrations are known. Uses numerical root
    finding to solve the non-linear equilibrium equations.
    
    Args:
        HCO3: Bicarbonate ion concentration in μmol/kg.
        DIC: Dissolved inorganic carbon concentration in μmol/kg.
        Ks: Equilibrium constants data structure.
        
    Returns:
        H: Hydrogen ion concentration in mol/kg (total scale).
        
    Note:
        This function uses the _zero_wrapper for robust root finding
        that handles both deterministic and uncertain parameters.
    """
    return solve_function(
        (HCO3, DIC, Ks.K1, Ks.K2), 
        zero_HCO3_DIC
    )
    
    # par = cast_array(HCO3, DIC, Ks.K1, Ks.K2)  # cast parameters into array
    # shape = maxShape(HCO3, DIC, Ks.K1, Ks.K2)  # get shape of output
    
    # result = np.apply_along_axis(_zero_wrapper, 0, par, fn=zero_HCO3_DIC).reshape(shape)
    
    # # If result is a single-element array, extract the scalar
    # if result.size == 1:
    #     return result.item()
    # return result


def zero_HCO3_DIC(h, HCO3, DIC, K1, K2):
    # Roots: two pos. Use smaller.
    LH = HCO3 * (h + h ** 2 / K1 + K2)
    RH = h * DIC
    return LH - RH


# 13. CO3 and TA
@uncertainty_propagation_decorator
def CO3_TA(CO3, TA, BT, Ks):
    """
    Returns H
    """
    
    return solve_function(
        (CO3, TA, BT, Ks.K1, Ks.K2, Ks.KB, Ks.KW), 
        zero_CO3_TA
    )
    
    # par = cast_array(
    #     CO3, TA, BT, Ks.K1, Ks.K2, Ks.KB, Ks.KW
    # )  # cast parameters into array
    # shape = maxShape(CO3, TA, BT, Ks.K1, Ks.K2, Ks.KB, Ks.KW)  # get shape of output
    
    # result = np.apply_along_axis(_zero_wrapper, 0, par, fn=zero_CO3_TA).reshape(shape)
    
    # # If result is a single-element array, extract the scalar
    # if result.size == 1:
    #     return result.item()
    # return result


def zero_CO3_TA(h, CO3, TA, BT, K1, K2, KB, KW):
    # Roots: three neg, two pos. Use larger pos.
    LH = TA * (KB + h) * (h ** 3 + K1 * h ** 2 + K1 * K2 * h)
    RH = (
        CO3
        * (h + h ** 2 / K2 + h ** 3 / (K1 * K2))
        * (K1 * h ** 2 + K1 * h * (KB + 2 * K2) + 2 * KB * K1 * K2)
    ) + (
        (h ** 2 + K1 * h + K1 * K2)
        * (KB * BT * h + KW * KB + KW * h - KB * h ** 2 - h ** 3)
    )
    return LH - RH


# 14. CO3 and DIC
def CO3_DIC(CO3, DIC, Ks):
    """
    Returns H
    """
    
    return solve_function(
        (CO3, DIC, Ks.K1, Ks.K2), 
        zero_CO3_DIC
    )
    
    # par = cast_array(CO3, DIC, Ks.K1, Ks.K2)  # cast parameters into array
    # shape = maxShape(CO3, DIC, Ks.K1, Ks.K2)  # get shape of output

    # result = np.apply_along_axis(_zero_wrapper, 0, par, fn=zero_CO3_DIC).reshape(shape)
    
    # # If result is a single-element array, extract the scalar
    # if result.size == 1:
    #     return result.item()
    # return result


def zero_CO3_DIC(h, CO3, DIC, K1, K2):
    # Roots: one pos, one neg. Use neg.
    LH = CO3 * (1 + h / K2 + h ** 2 / (K1 * K2))
    RH = DIC
    return LH - RH


# 15. TA and DIC
@uncertainty_propagation_decorator
def TA_DIC(TA: Union[float, np.ndarray], DIC: Union[float, np.ndarray], BT: Union[float, np.ndarray], 
           PT: Union[float, np.ndarray], SiT: Union[float, np.ndarray], ST: Union[float, np.ndarray], 
           FT: Union[float, np.ndarray], Ks: Union[KValues,dict]) -> Union[float, np.ndarray]:
    """
    Calculate pH from total alkalinity and dissolved inorganic carbon using iterative Newton-Raphson method.
    
    This is one of the most important carbonate system calculations, implementing
    the CO2SYS algorithm for solving pH when both total alkalinity and dissolved
    inorganic carbon are known. Uses iterative convergence to find pH that
    satisfies the alkalinity balance including all major acid-base systems.
    
    Args:
        TA: Total alkalinity in μmol/kg.
        DIC: Dissolved inorganic carbon concentration in μmol/kg.
        BT: Total boron concentration in μmol/kg.
        PT: Total phosphate concentration in μmol/kg.
        SiT: Total silicate concentration in μmol/kg.
        ST: Total sulfate concentration in μmol/kg.
        FT: Total fluoride concentration in μmol/kg.
        Ks: Equilibrium constants data structure.
        
    Returns:
        pH: pH value on the total scale.
        
    Note:
        Taken directly from MATLAB CO2SYS. Uses Newton-Raphson iteration with
        adaptive step size control for robust convergence. This is the most
        comprehensive alkalinity solver including all nutrient contributions.
    """
    # determine shape of input
    L = maxShape(TA, DIC, BT, PT, SiT, ST, FT, Ks.K1)
    if len(L) == 0:
        L = (1,)
        
    pHguess = 8.0
    pHtol = 0.00000001
    pHx = np.full(L, pHguess)
    deltapH = np.array(pHtol + 1, ndmin=1)
    ln10 = np.log(10)

    while np.any(abs(deltapH) > pHtol):
        H = 10 ** -pHx
        # negative
        Denom = H ** 2 + Ks.K1 * H + Ks.K1 * Ks.K2
        CAlk = DIC * Ks.K1 * (H + 2 * Ks.K2) / Denom
        BAlk = BT * Ks.KB / (Ks.KB + H)
        OH = Ks.KW / H
        PhosTop = Ks.KP1 * Ks.KP2 * H + 2 * Ks.KP1 * Ks.KP2 * Ks.KP3 - H ** 3
        PhosBot = (
            H ** 3 + Ks.KP1 * H ** 2 + Ks.KP1 * Ks.KP2 * H + Ks.KP1 * Ks.KP2 * Ks.KP3
        )
        PAlk = PT * PhosTop / PhosBot
        SiAlk = SiT * Ks.KSi / (Ks.KSi + H)
        # positive
        Hfree = H / (1 + ST / Ks.KS)
        HSO4 = ST / (1 + Ks.KS / Hfree)
        HF = FT / (1 + Ks.KF / Hfree)

        Residual = TA - CAlk - BAlk - OH - PAlk - SiAlk + Hfree + HSO4 + HF

        Slope = ln10 * (
            DIC * Ks.K1 * H * (H ** 2 + Ks.K1 * Ks.K2 + 4 * H * Ks.K2) / Denom / Denom
            + BAlk * H / (Ks.KB + H)
            + OH
            + H
        )
        deltapH = Residual / Slope

        while np.any(abs(deltapH) > 1):
            FF = abs(deltapH) > 1
            deltapH[FF] = deltapH[FF] / 2

        pHx += deltapH

    return pHx if pHx.size > 1 else pHx.item()


def zero_TA_DIC(h, TA, DIC, BT, K1, K2, KB, KW):
    # Roots: one pos, four neg. Use pos.
    LH = DIC * (KB + h) * (K1 * h ** 2 + 2 * K1 * K2 * h)
    RH = (TA * (KB + h) * h - KB * BT * h - KW * (KB + h) + (KB + h) * h ** 2) * (
        h ** 2 + K1 * h + K1 * K2
    )
    return LH - RH


# 1.1.9
def cCO2(H: Union[float, np.ndarray], DIC: Union[float, np.ndarray], Ks: Union[KValues,dict]) -> Union[float, np.ndarray]:
    """
    Calculate dissolved CO2 concentration from hydrogen ion and DIC.
    
    Uses carbonate system equilibria to calculate dissolved CO2 concentration
    from hydrogen ion concentration and total dissolved inorganic carbon.
    
    Args:
        H: Hydrogen ion concentration in mol/kg (total scale).
        DIC: Dissolved inorganic carbon concentration in μmol/kg.
        Ks: Equilibrium constants data structure.
        
    Returns:
        CO2: Dissolved CO2 concentration in μmol/kg.
    """
    return DIC / (1 + Ks.K1 / H + Ks.K1 * Ks.K2 / H ** 2)


# 1.1.10
def cHCO3(H: Union[float, np.ndarray], DIC: Union[float, np.ndarray], Ks: Union[KValues,dict]) -> Union[float, np.ndarray]:
    """
    Calculate bicarbonate concentration from hydrogen ion and DIC.
    
    Uses carbonate system equilibria to calculate bicarbonate ion concentration
    from hydrogen ion concentration and total dissolved inorganic carbon.
    
    Args:
        H: Hydrogen ion concentration in mol/kg (total scale).
        DIC: Dissolved inorganic carbon concentration in μmol/kg.
        Ks: Equilibrium constants data structure.
        
    Returns:
        HCO3: Bicarbonate ion concentration in μmol/kg.
    """
    return DIC / (1 + H / Ks.K1 + Ks.K2 / H)


# 1.1.11
def cCO3(H: Union[float, np.ndarray], DIC: Union[float, np.ndarray], Ks: Union[KValues,dict]) -> Union[float, np.ndarray]:
    """
    Calculate carbonate concentration from hydrogen ion and DIC.
    
    Uses carbonate system equilibria to calculate carbonate ion concentration
    from hydrogen ion concentration and total dissolved inorganic carbon.
    
    Args:
        H: Hydrogen ion concentration in mol/kg (total scale).
        DIC: Dissolved inorganic carbon concentration in μmol/kg.
        Ks: Equilibrium constants data structure.
        
    Returns:
        CO3: Carbonate ion concentration in μmol/kg.
    """
    return DIC / (1 + H / Ks.K2 + H ** 2 / (Ks.K1 * Ks.K2))


# 1.5.80
def cTA(H: Union[float, np.ndarray], DIC: Union[float, np.ndarray], BT: Union[float, np.ndarray], 
        PT: Union[float, np.ndarray], SiT: Union[float, np.ndarray], ST: Union[float, np.ndarray], 
        FT: Union[float, np.ndarray], Ks: Union[KValues,dict], mode: str = "multi") -> Union[Union[float, np.ndarray], Tuple]:
    """
    Calculate total alkalinity and individual alkalinity components.
    
    Comprehensive calculation of total alkalinity from hydrogen ion concentration
    and all major dissolved species. Includes contributions from carbonate,
    borate, phosphate, silicate, water, sulfate, and fluoride systems.
    
    Args:
        H: Hydrogen ion concentration in mol/kg (total scale).
        DIC: Dissolved inorganic carbon concentration in μmol/kg.
        BT: Total boron concentration in μmol/kg.
        PT: Total phosphate concentration in μmol/kg.
        SiT: Total silicate concentration in μmol/kg.
        ST: Total sulfate concentration in μmol/kg.
        FT: Total fluoride concentration in μmol/kg.
        Ks: Equilibrium constants data structure.
        mode: Output mode - "multi" returns all components, else returns only TA.
        
    Returns:
        If mode == "multi": tuple of (TA, CAlk, BAlk, PAlk, SiAlk, OH, Hfree, HSO4, HF)
        Else: TA only.
        
    Note:
        All alkalinity components in μmol/kg. This is the comprehensive alkalinity
        calculation used throughout the carbonate system calculations.
    """
    # negative
    Denom = H ** 2 + Ks.K1 * H + Ks.K1 * Ks.K2
    CAlk = DIC * Ks.K1 * (H + 2 * Ks.K2) / Denom
    BAlk = BT * Ks.KB / (Ks.KB + H)
    OH = Ks.KW / H
    PhosTop = Ks.KP1 * Ks.KP2 * H + 2 * Ks.KP1 * Ks.KP2 * Ks.KP3 - H ** 3
    PhosBot = H ** 3 + Ks.KP1 * H ** 2 + Ks.KP1 * Ks.KP2 * H + Ks.KP1 * Ks.KP2 * Ks.KP3
    PAlk = PT * PhosTop / PhosBot
    SiAlk = SiT * Ks.KSi / (Ks.KSi + H)
    # positive
    Hfree = H / (1 + ST / Ks.KS)
    HSO4 = ST / (1 + Ks.KS / Hfree)
    HF = FT / (1 + Ks.KF / Hfree)

    TA = CAlk + BAlk + OH + PAlk + SiAlk - Hfree - HSO4 - HF

    if mode == "multi":
        return TA, CAlk, BAlk, PAlk, SiAlk, OH, Hfree, HSO4, HF
    else:
        return TA

# C.4.14
def fCO2_to_CO2(fCO2: Union[float, np.ndarray], Ks: Union[KValues,dict]) -> Union[float, np.ndarray]:
    """
    Calculate dissolved CO2 concentration from CO2 fugacity.
    
    Simple conversion using Henry's law constant for CO2 solubility.
    
    Args:
        fCO2: CO2 fugacity in μatm.
        Ks: Equilibrium constants data structure containing K0.
        
    Returns:
        CO2: Dissolved CO2 concentration in μmol/kg.
    """
    return fCO2 * Ks.K0


# C.4.14
def CO2_to_fCO2(CO2: Union[float, np.ndarray], Ks: Union[KValues,dict]) -> Union[float, np.ndarray]:
    """
    Calculate CO2 fugacity from dissolved CO2 concentration.
    
    Simple conversion using Henry's law constant for CO2 solubility.
    
    Args:
        CO2: Dissolved CO2 concentration in μmol/kg.
        Ks: Equilibrium constants data structure containing K0.
        
    Returns:
        fCO2: CO2 fugacity in μatm.
    """
    return CO2 / Ks.K0


def pCO2_to_fCO2(pCO2: Union[float, np.ndarray], Tc: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate CO2 fugacity from CO2 partial pressure with virial correction.
    
    Applies virial equation of state correction to convert partial pressure
    to fugacity, accounting for non-ideal gas behavior. Based on Weiss (1974)
    formulation used in MATLAB CO2SYS.
    
    Args:
        pCO2: CO2 partial pressure in μatm.
        Tc: Temperature in degrees Celsius.
        
    Returns:
        fCO2: CO2 fugacity in μatm.
        
    Note:
        Assumes pressure is at one atmosphere or close to it. Otherwise,
        the pressure term in the exponent affects the results. Based on
        Weiss, R. F., Marine Chemistry 2:203-215, 1974.
        
        For a mixture of CO2 and air at 1 atm (at low CO2 concentrations).
        Delta and B are in cm³/mol.
    """
    Tk = Tc + 273.15
    P = 1.01325  # in bar
    RT = 83.1451 * Tk

    a0, a1, a2, a3 = (-1636.75, 12.0408, -3.27957e-2, 3.16528e-05)
    b0, b1 = (57.7, -0.118)

    B = a0 + a1 * Tk + a2 * Tk ** 2 + a3 * Tk ** 3
    delta = b0 + b1 * Tk

    return pCO2 * np.exp(P * (B + 2 * delta) / RT)


def fCO2_to_pCO2(fCO2: Union[float, np.ndarray], Tc: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate CO2 partial pressure from CO2 fugacity with virial correction.
    
    Applies virial equation of state correction to convert fugacity to
    partial pressure, accounting for non-ideal gas behavior. Based on
    Weiss (1974) formulation used in MATLAB CO2SYS.
    
    Args:
        fCO2: CO2 fugacity in μatm.
        Tc: Temperature in degrees Celsius.
        
    Returns:
        pCO2: CO2 partial pressure in μatm.
        
    Note:
        Assumes pressure is at one atmosphere or close to it. Otherwise,
        the pressure term in the exponent affects the results. Based on
        Weiss, R. F., Marine Chemistry 2:203-215, 1974.
        
        For a mixture of CO2 and air at 1 atm (at low CO2 concentrations).
        Delta and B are in cm³/mol.
    """
    Tk = Tc + 273.15
    P = 1.01325  # in bar
    RT = 83.1451 * Tk

    a0, a1, a2, a3 = (-1636.75, 12.0408, -3.27957e-2, 3.16528e-05)
    b0, b1 = (57.7, -0.118)

    B = a0 + a1 * Tk + a2 * Tk ** 2 + a3 * Tk ** 3
    delta = b0 + b1 * Tk

    return fCO2 / np.exp(P * (B + 2 * delta) / RT)

def calc_revelle_factor(TA: Union[float, np.ndarray], DIC: Union[float, np.ndarray], BT: Union[float, np.ndarray], 
                       PT: Union[float, np.ndarray], SiT: Union[float, np.ndarray], ST: Union[float, np.ndarray], 
                       FT: Union[float, np.ndarray], Ks: Union[KValues,dict]) -> Union[float, np.ndarray]:
    """
    Calculate the Revelle factor (buffer factor) for the carbonate system.
    
    The Revelle factor quantifies the sensitivity of CO2 fugacity to changes
    in dissolved inorganic carbon at constant alkalinity. It is defined as
    (dfCO2/fCO2) / (dDIC/DIC) and is an important parameter for understanding
    the ocean's capacity to absorb atmospheric CO2.
    
    Args:
        TA: Total alkalinity in μmol/kg.
        DIC: Dissolved inorganic carbon concentration in μmol/kg.
        BT: Total boron concentration in μmol/kg.
        PT: Total phosphate concentration in μmol/kg.
        SiT: Total silicate concentration in μmol/kg.
        ST: Total sulfate concentration in μmol/kg.
        FT: Total fluoride concentration in μmol/kg.
        Ks: Equilibrium constants data structure.
        
    Returns:
        Revelle factor (dimensionless).
        
    Note:
        Calculated using finite differences with dDIC = 1 μmol/kg.
        Higher values indicate lower buffering capacity.
    """
    dDIC = 1e-6  # (1 umol kg-1)

    pH = TA_DIC(TA=TA, DIC=DIC, BT=BT, PT=PT, SiT=SiT, ST=ST, FT=FT, Ks=Ks)
    fCO2 = cCO2(10.0**-pH, DIC, Ks) / Ks.K0

    # Calculate new fCO2 above and below given value
    pH_hi = TA_DIC(TA=TA, DIC=DIC + dDIC, BT=BT, PT=PT, SiT=SiT, ST=ST, FT=FT, Ks=Ks)
    fCO2_hi = cCO2(10.0**-pH_hi, DIC, Ks) / Ks.K0

    pH_lo = TA_DIC(TA=TA, DIC=DIC - dDIC, BT=BT, PT=PT, SiT=SiT, ST=ST, FT=FT, Ks=Ks)
    fCO2_lo = cCO2(10.0**-pH_lo, DIC, Ks) / Ks.K0

    return (fCO2_hi - fCO2_lo) * DIC / (fCO2 * 2 * dDIC)

# C system Utilities

def given(params: CBsystData) -> List[Any]:
    """
    Get list of provided carbon system parameters from dataclass.
    
    Identifies which carbon system parameters have been provided (are not None)
    in the input dataclass. Used to determine which solver to use.
    
    Args:
        params: CBsyst data structure containing carbon parameters.
        
    Returns:
        List of parameter values that are not None.
    """
    valid_inputs = ['CO2', 'HCO3', 'CO3', 'TA', 'DIC', 'pCO2', 'fCO2', 'OmegaC', 'OmegaA']
    return [params.get(p) for p in valid_inputs if params.get(p) is not None]

def n_given(params: CBsystData) -> int:
    """
    Count number of provided carbon system parameters.
    
    Args:
        params: CBsyst data structure containing carbon parameters.
        
    Returns:
        Number of carbon system parameters that are not None.
    """
    return len(given(params))

def use_Omega(params: CBsystData) -> None:
    """
    Convert omega saturation states to carbonate concentration.
    
    If omega values are provided instead of carbonate concentration,
    calculate CO3 from the saturation state and solubility constant.
    
    Args:
        params: CBsyst data structure, modified in place.
    """
    if params.OmegaC is not None:
        params.CO3 = params.OmegaC * params.Ks.KspC / params.Ca
    if params.OmegaA is not None:
        params.CO3 = params.OmegaA * params.Ks.KspA / params.Ca

def calculate_Omegas(params: CBsystData) -> None:
    """
    Calculate omega saturation states from carbonate concentration.
    
    Calculate aragonite and calcite saturation states from carbonate ion
    concentration, calcium concentration, and solubility constants.
    
    Args:
        params: CBsyst data structure, modified in place.
    """
    if params.OmegaA is None: params.OmegaA = params.CO3 * params.Ca * params.S_in / 35 / params.Ks.KspA
    if params.OmegaC is None: params.OmegaC = params.CO3 * params.Ca * params.S_in / 35 / params.Ks.KspC

def convert_CO2(params: CBsystData) -> None:
    """
    Convert pCO2 or fCO2 to dissolved CO2 concentration.
    
    If CO2 is not provided but pCO2 or fCO2 are available,
    convert them to dissolved CO2 using appropriate conversions.
    
    Args:
        params: CBsyst data structure, modified in place.
    """
    if params.CO2 is None:
        if params.fCO2 is not None:
            params.CO2 = fCO2_to_CO2(params.fCO2, params.Ks)
        elif params.pCO2 is not None:
            params.fCO2 = pCO2_to_fCO2(params.pCO2, params.T_in)
            params.CO2 = fCO2_to_CO2(params.fCO2, params.Ks)

# B System Solvers

SOLVER_RULES: Dict[Tuple[str, str], List[Callable[[CBsystData], None]]] = {
    ('CO2', 'pHtot'): [
        lambda p: setattr(p, 'H', 10.0**-p.pHtot),
        lambda p: setattr(p, 'DIC', CO2_H(CO2=p.CO2, H=p.H, Ks=p.Ks)),
    ],
    ('CO2', 'HCO3'): [
        lambda p: setattr(p, 'H', CO2_HCO3(CO2=p.CO2, HCO3=p.HCO3, Ks=p.Ks)),
        lambda p: setattr(p, 'DIC', CO2_H(CO2=p.CO2, H=p.H, Ks=p.Ks)),
    ],
    ('CO2', 'CO3'): [
        lambda p: setattr(p, 'H', CO2_CO3(CO2=p.CO2, CO3=p.CO3, Ks=p.Ks)),
        lambda p: setattr(p, 'DIC', CO2_H(CO2=p.CO2, H=p.H, Ks=p.Ks)),
    ],
    ('CO2', 'TA'): [
        lambda p: setattr(p, 'pHtot', CO2_TA(CO2=p.CO2, TA=p.TA, BT=p.BT, PT=p.PT, SiT=p.SiT, ST=p.ST, FT=p.FT, Ks=p.Ks)),
        lambda p: setattr(p, 'H', 10.0**-p.pHtot),
        lambda p: setattr(p, 'DIC', CO2_H(CO2=p.CO2, H=p.H, Ks=p.Ks)),
    ],
    ('CO2', 'DIC'): [
        lambda p: setattr(p, 'H', CO2_DIC(CO2=p.CO2, DIC=p.DIC, Ks=p.Ks)),
    ],
    ('pHtot', 'HCO3'): [
        lambda p: setattr(p, 'H', 10.0**-p.pHtot),
        lambda p: setattr(p, 'DIC', H_HCO3(H=p.H, HCO3=p.HCO3, Ks=p.Ks)),
    ],
    ('pHtot', 'CO3'): [
        lambda p: setattr(p, 'H', 10.0**-p.pHtot),
        lambda p: setattr(p, 'DIC', H_CO3(H=p.H, CO3=p.CO3, Ks=p.Ks)),
    ],
    ('pHtot', 'TA'): [
        lambda p: setattr(p, 'H', 10.0**-p.pHtot),
        lambda p: setattr(p, 'DIC', H_TA(H=p.H, TA=p.TA, BT=p.BT, PT=p.PT, SiT=p.SiT, ST=p.ST, FT=p.FT, Ks=p.Ks)),
    ],
    ('pHtot', 'DIC'): [
        lambda p: setattr(p, 'H', 10.0**-p.pHtot),
    ],
    ('HCO3', 'CO3'): [
        lambda p: setattr(p, 'H', HCO3_CO3(HCO3=p.HCO3, CO3=p.CO3, Ks=p.Ks)),
        lambda p: setattr(p, 'DIC', H_CO3(H=p.H, CO3=p.CO3, Ks=p.Ks)),
    ],
    ('HCO3', 'TA'): [
        lambda p: setattr(p, 'H', HCO3_TA(HCO3=p.HCO3, TA=p.TA, BT=p.BT, Ks=p.Ks)),
        lambda p: setattr(p, 'DIC', H_HCO3(H=p.H, HCO3=p.HCO3, Ks=p.Ks)),
    ],
    ('HCO3', 'DIC'): [
        lambda p: setattr(p, 'H', HCO3_DIC(HCO3=p.HCO3, DIC=p.DIC, Ks=p.Ks)),
    ],
    ('CO3', 'TA'): [
        lambda p: setattr(p, 'H', CO3_TA(CO3=p.CO3, TA=p.TA, BT=p.BT, Ks=p.Ks)),
        lambda p: setattr(p, 'DIC', H_CO3(H=p.H, CO3=p.CO3, Ks=p.Ks)),
    ],
    ('CO3', 'DIC'): [
        lambda p: setattr(p, 'H', CO3_DIC(CO3=p.CO3, DIC=p.DIC, Ks=p.Ks)),
    ],
    ('TA', 'DIC'): [
        lambda p: setattr(p, 'pHtot', TA_DIC(TA=p.TA, DIC=p.DIC, BT=p.BT, PT=p.PT, SiT=p.SiT, ST=p.ST, FT=p.FT, Ks=p.Ks)),
        lambda p: setattr(p, 'H', 10.0**-p.pHtot),
    ],
}

def calculate_remaining_C_species(params: CBsystData) -> None:
    """
    Calculate all remaining carbon species after solving the main system.
    
    Once H and DIC are known from the primary solver, calculate all other
    carbon species and alkalinity components that were not provided as input.
    
    Args:
        params: CBsyst data structure, modified in place.
    """
    # populate missing carbon parameters
    if params.CO2 is None: params.CO2 = cCO2(params.H, params.DIC, params.Ks)
    if params.fCO2 is None: params.fCO2 = CO2_to_fCO2(params.CO2, params.Ks)
    if params.pCO2 is None: params.pCO2 = fCO2_to_pCO2(params.fCO2, params.T_in)
    if params.HCO3 is None: params.HCO3 = cHCO3(params.H, params.DIC, params.Ks)
    if params.CO3 is None: params.CO3 = cCO3(params.H, params.DIC, params.Ks)
    if params.pHtot is None: params.pHtot = negative_log10_preserve_type(params.H)

    TA_COMPONENTS = ['TA', 'CAlk', 'BAlk', 'PAlk', 'SiAlk', 'OH', 'Hfree', 'HSO4', 'HF']
    for par, val in zip(TA_COMPONENTS, cTA(
                H=params.H, DIC=params.DIC, BT=params.BT, PT=params.PT, 
                SiT=params.SiT, ST=params.ST, FT=params.FT, 
                Ks=params.Ks, mode="multi"
            )):
        if params[par] is None: params[par] = val

def solve_C_system(params: CBsystData) -> None:
    """
    Solve the complete carbonate system from provided parameters.
    
    Main carbonate system solver that:
    1. Converts CO2-related parameters as needed
    2. Converts omega values to carbonate if needed
    3. Identifies the parameter pair and selects appropriate solver
    4. Solves for H and DIC
    5. Calculates all remaining carbon species
    6. Calculates omega saturation states
    
    Args:
        params: CBsyst data structure, modified in place.
        
    Raises:
        ValueError: If wrong number of parameters provided or unsupported combination.
    """

    convert_CO2(params)
    use_Omega(params)

    # identify pair of provided params
    carbon_params = ['CO2', 'pHtot', 'HCO3', 'CO3', 'TA', 'DIC']
    provided = tuple([p for p in carbon_params if params.get(p) is not None])
    # params.inputs += provided 

    solver = SOLVER_RULES.get(provided)
    if solver is None:
        n_provided = len(provided)
        if n_provided < 2:
            msg = f"Not enough carbon parameters provided: {provided}"
        elif n_provided > 2:
            msg = f"Too many carbon parameters provided: {provided}"
        else:
            msg = f"No carbon system solver for parameter combination: {provided}"
        raise ValueError(msg)
    
    # solve for H and DIC
    for function in solver:
        function(params)

    calculate_remaining_C_species(params)
    # calc_remaining_B_species(params)
    calculate_Omegas(params)
