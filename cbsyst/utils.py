from typing import Union
from .uncertainties import remove_negatives
from .dataclasses import CarbonSystemParams, CBsystData

def remove_negative_concentrations(params: CarbonSystemParams) -> None:
    """
    Remove negative values from concentration parameters.
    
    Applies the remove_negatives function to all concentration parameters
    that should be non-negative by definition (concentrations, pressures, etc.).
    This is typically used after numerical calculations to clean up results
    that may have small negative values due to numerical precision issues.
    
    Args:
        params: CBsyst data structure containing concentration parameters.
        
    Note:
        This function modifies the params object in-place. The parameters
        cleaned include: DIC, CO2, HCO3, CO3, BT, fCO2, pCO2, PT, SiT.
        These are all parameters that must be non-negative by physical
        definition.
    """
    for param_name in ["DIC", "CO2", "HCO3", "CO3", "BT", "fCO2", "pCO2", "PT", "SiT"]:
        setattr(params, param_name, remove_negatives(getattr(params, param_name)))

def has_output_condition(params: Union[CBsystData, CarbonSystemParams]) -> bool:
    """
    Check if output conditions differ from input conditions.
    
    Determines whether the calculation requires correction from input
    conditions to different output conditions for temperature, salinity,
    or pressure.
    
    Args:
        params: CBsyst data structure containing environmental conditions.
        
    Returns:
        True if any output condition (T_out, S_out, P_out) is specified
        and differs from input conditions, False otherwise.
        
    Note:
        This is used to determine whether temperature/pressure corrections
        need to be applied to equilibrium constants and whether pH scale
        conversions need to account for different ionic strengths.
    """
    return params.T_out is not None or params.S_out is not None or params.P_out is not None

