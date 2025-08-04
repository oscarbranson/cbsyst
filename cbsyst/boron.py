"""
Functions for calculating boron speciation.
"""

import numpy as np
from typing import Union, Any, Dict, List, Tuple, Callable
from .dataclasses import KValues, CBsystData
from .uncertainties import negative_log10_preserve_type
# Basic Calculation Functions

def chiB_calc(
    H: Union[float, np.ndarray], 
    Ks: Union[KValues, Dict]) -> Union[float, np.ndarray]:
    """
    Calculate the fraction of total boron present as borate (BO4-).
    
    Args:
        H: Hydrogen ion concentration (mol/kg) on the same scale as the KB value (total scale).
        Ks: Object containing equilibrium constants, must have KB attribute.
        
    Returns:
        float or array: Fraction of total boron as borate (dimensionless).
        
    Notes:
        chiB = [BO4-] / BT = 1 / (1 + KB/H)
        where KB is the boron dissociation constant.
    """
    return 1 / (1 + Ks.KB / H)

# B conc fns
def BT_BO3(
    BT: Union[float, np.ndarray], 
    BO3: Union[float, np.ndarray], 
    Ks: Union[KValues, Dict]
) -> Union[float, np.ndarray]:
    """
    Calculate hydrogen ion concentration from total boron and boric acid.
    
    Args:
        BT: Total boron concentration (mol/kg).
        BO3: Boric acid (B(OH)3) concentration (mol/kg).
        Ks: Object containing equilibrium constants, must have KB attribute.
        
    Returns:
        float or array: Hydrogen ion concentration (mol/kg) on total scale.
        
    Notes:
        Derived from the boron equilibrium: B(OH)3 + H2O ⇌ B(OH)4- + H+
        H = KB / (BT/BO3 - 1)
    """
    return Ks.KB / (BT / BO3 - 1)


def BT_BO4(
    BT: Union[float, np.ndarray], 
    BO4: Union[float, np.ndarray], 
    Ks: Union[KValues, Dict]
) -> Union[float, np.ndarray]:
    """
    Calculate hydrogen ion concentration from total boron and borate.
    
    Args:
        BT: Total boron concentration (mol/kg).
        BO4: Borate (B(OH)4-) concentration (mol/kg).
        Ks: Object containing equilibrium constants, must have KB attribute.
        
    Returns:
        float or array: Hydrogen ion concentration (mol/kg) on total scale.
        
    Notes:
        Derived from the boron equilibrium: B(OH)3 + H2O ⇌ B(OH)4- + H+
        H = KB * (BT/BO4 - 1)
    """
    return Ks.KB * (BT / BO4 - 1)


def pH_BO3(
    pH: Union[float, np.ndarray], 
    BO3: Union[float, np.ndarray], 
    Ks: Union[KValues, Dict]
) -> Union[float, np.ndarray]:
    """
    Calculate total boron concentration from pH and boric acid concentration.
    
    Args:
        pH: pH on total scale.
        BO3: Boric acid (B(OH)3) concentration (mol/kg).
        Ks: Object containing equilibrium constants, must have KB attribute.
        
    Returns:
        float or array: Total boron concentration (mol/kg).
        
    Notes:
        BT = BO3 * (1 + KB/H) where H = 10^(-pH)
    """
    H = 10.0**-pH
    return BO3 * (1 + Ks.KB / H)


def pH_BO4(
    pH: Union[float, np.ndarray], 
    BO4: Union[float, np.ndarray], 
    Ks: Union[KValues, Dict]
) -> Union[float, np.ndarray]:
    """
    Calculate total boron concentration from pH and borate concentration.
    
    Args:
        pH: pH on total scale.
        BO4: Borate (B(OH)4-) concentration (mol/kg).
        Ks: Object containing equilibrium constants, must have KB attribute.
        
    Returns:
        float or array: Total boron concentration (mol/kg).
        
    Notes:
        BT = BO4 * (1 + H/KB) where H = 10^(-pH)
    """
    H = 10.0**-pH
    return BO4 * (1 + H / Ks.KB)


def cBO4(
    BT: Union[float, np.ndarray], 
    H: Union[float, np.ndarray], 
    Ks: Union[KValues, Dict]
) -> Union[float, np.ndarray]:
    """
    Calculate borate concentration from total boron and hydrogen ion concentration.
    
    Args:
        BT: Total boron concentration (mol/kg).
        H: Hydrogen ion concentration (mol/kg) on total scale.
        Ks: Object containing equilibrium constants, must have KB attribute.
        
    Returns:
        float or array: Borate (B(OH)4-) concentration (mol/kg).
        
    Notes:
        BO4 = BT / (1 + H/KB)
    """
    return BT / (1 + H / Ks.KB)


def cBO3(
    BT: Union[float, np.ndarray], 
    H: Union[float, np.ndarray], 
    Ks: Union[KValues, Dict]
) -> Union[float, np.ndarray]:
    """
    Calculate boric acid concentration from total boron and hydrogen ion concentration.
    
    Args:
        BT: Total boron concentration (mol/kg).
        H: Hydrogen ion concentration (mol/kg) on total scale.
        Ks: Object containing equilibrium constants, must have KB attribute.
        
    Returns:
        float or array: Boric acid (B(OH)3) concentration (mol/kg).
        
    Notes:
        BO3 = BT / (1 + KB/H)
    """
    return BT / (1 + Ks.KB / H)

# B system Utilities

def given(params: CBsystData) -> List[Any]:
    """
    Check which boron parameters are given in the parameters.
    
    Args:
        params: Parameter object with boron system attributes.
        
    Returns:
        list: List of non-None boron parameter values from BT, BO3, BO4.
    """
    valid_inputs = ['BT', 'BO3', 'BO4']
    return [params.get(p) for p in valid_inputs if params.get(p) is not None]

def n_given(params: CBsystData) -> int:
    """
    Count the number of boron parameters provided.
    
    Args:
        params: Parameter object with boron system attributes.
        
    Returns:
        int: Number of non-None boron parameters (BT, BO3, BO4).
    """
    return len(given(params))

# B System Solvers

SOLVER_RULES: Dict[Tuple[str, str], List[Callable[[CBsystData], None]]] = {
    ('pHtot', 'BT'): [
        lambda p: setattr(p, 'H', 10.0**-p.pHtot)
        ],
    ('BT', 'BO3'): [
        lambda p: setattr(p, 'H', BT_BO3(BT=p.BT, BO=p.BO3, Ks=p.Ks))
        ],
    ('BT', 'BO4'): [
        lambda p: setattr(p, 'H', BT_BO4()(BT=p.BT, BO4=p.BO4, Ks=p.Ks))
        ],
    ('BO3', 'BO4'): [
        lambda p: setattr(p, 'BT', p.BO3 + p.BO4),
        lambda p: setattr(p, 'H', BT_BO4()(BT=p.BT, BO4=p.BO4, Ks=p.Ks))
    ],
    ('pHtot', 'BO3'): [
        lambda p: setattr(p, 'H', 10.0**-p.pHtot),
        lambda p: setattr(p, 'BT', pH_BO3(pH=p.pHtot, BO3=p.BO3, Ks=p.Ks))
    ],
    ('pHtot', 'BO4'): [
        lambda p: setattr(p, 'H', 10.0**-p.pHtot),
        lambda p: setattr(p, 'BT', pH_BO4(pH=p.pHtot, BO4=p.BO4, Ks=p.Ks))
    ],
}

def calc_remaining_B_species(params: CBsystData) -> None:
    """
    Calculate remaining boron species after solving for H and BT.
    
    Args:
        params: Parameter object with H, BT, Ks attributes and dataclass fields.
        
    Notes:
        Calculates BO3 and BO4 if they are None and part of the dataclass.
        Calculates pHtot if it is None.
    """
    if 'BO3' in params.__dataclass_fields__:
        if params.BO3 is None: params.BO3 = cBO3(params.BT, params.H, params.Ks)
        if params.BO4 is None: params.BO4 = cBO4(params.BT, params.H, params.Ks)
    if params.pHtot is None: params.pHtot = negative_log10_preserve_type(params.H)

def solve_B_system(params: CBsystData) -> None:
    """
    Solve the boron system given any two boron parameters.
    
    Args:
        params: Parameter object with boron system attributes.
        
    Raises:
        ValueError: If no solver is found for the provided parameter combination.
        
    Notes:
        Uses appropriate solver function based on which parameters are provided.
        Valid combinations: (pHtot, BT), (BT, BO3), (BT, BO4), (BO3, BO4), 
                          (pHtot, BO3), (pHtot, BO4)
    """
    boron_params = ['pHtot', 'BT', 'BO3', 'BO4']
    provided = tuple([p for p in boron_params if params.get(p) is not None])

    if provided in SOLVER_RULES:
        for func in SOLVER_RULES[provided]:
            func(params)
    else:
        raise ValueError(f"No solver found for provided parameters: {provided}")
