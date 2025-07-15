from .uncertainties import remove_negatives
from .dataclasses import CarbonSystemParams

def remove_negative_concentrations(params: CarbonSystemParams) -> None:
    """Remove negative values from concentration parameters"""
    for param_name in ["DIC", "CO2", "HCO3", "CO3", "BT", "fCO2", "pCO2", "PT", "SiT"]:
        setattr(params, param_name, remove_negatives(getattr(params, param_name)))

def has_output_condition(params):
    return params.T_out is not None or params.S_out is not None or params.P_out is not None

