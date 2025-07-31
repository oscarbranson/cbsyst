import numpy as np
from .dataclasses import CBsystData

CONCENTRATION_PARAMS = ["DIC", "TA", "CO2", "HCO3", "CO3", "BT", "BO3", "BO4", "PT", "SiT"]
GAS_PARAMS = ['pCO2', 'fCO2']
ALKALINITY_PARAMS = ["CAlk", "BAlk", "PAlk", "SiAlk", "OH", "HSO4", "HF", "Hfree"]

UNIT_MULTIPLIERS = {
    "mol": 1.0,
    "mmol": 1.0e3,
    "umol": 1.0e6,
    "nmol": 1.0e9,
    "pmol": 1.0e12,
    "fmol": 1.0e15,
}
    
def convert_to_molar(params: CBsystData) -> None:
    """Convert input units to molar"""
    multiplier = UNIT_MULTIPLIERS.get(params.unit, params.unit)
    
    if multiplier != 1:
        for param in CONCENTRATION_PARAMS:
            if param not in params.__dataclass_fields__:
                continue
            value = getattr(params, param)
            if value is not None:
                setattr(params, param, np.divide(value, multiplier))
    
    for param in GAS_PARAMS:
        if param not in params.__dataclass_fields__:
            continue        
        value = getattr(params, param)
        if value is not None:
            setattr(params, param, np.divide(value, 1e6))

def convert_from_molar(params: CBsystData) -> None:
    """Convert results back to input units"""
    multiplier = UNIT_MULTIPLIERS.get(params.unit, params.unit)

    if multiplier != 1:
        for param in CONCENTRATION_PARAMS + ALKALINITY_PARAMS:
            if param not in params.__dataclass_fields__:
                continue
            value = getattr(params, param)
            if value is not None:
                params[param] *= multiplier
    
    for param in GAS_PARAMS:
        if param not in params.__dataclass_fields__:
            continue
        value = getattr(params, param)
        if value is not None:
            params[param] = np.multiply(value, 1e6)