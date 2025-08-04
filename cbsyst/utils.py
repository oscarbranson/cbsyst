from typing import Union
import numpy as np
from .uncertainties import remove_negatives
from .dataclasses import CarbonSystemParams, CBsystData
from . import constants

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

def swdens(TempC: Union[float, np.ndarray], Sal: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate seawater density from temperature and salinity.
    
    Calculates seawater density using the equation of state from Dickson, 
    Sabine and Christian (2007). The calculation follows Chapter 5, Section 4.2
    of the Guide to Best Practices for Ocean CO2 Measurements.
    
    Args:
        TempC: Temperature in degrees Celsius.
        Sal: Salinity in practical salinity units (PSU).
        
    Returns:
        Seawater density in kg/L.
        
    Note:
        Temperature is first converted to IPTS-68 scale before calculation.
        The density equation is valid for typical oceanic temperature and
        salinity ranges.
        
    Reference:
        Dickson, A.G., Sabine, C.L. and Christian, J.R. (Eds.) 2007. Guide to 
        best practices for ocean CO2 measurements. PICES Special Publication 3, 
        191 pp. http://cdiac.ornl.gov/oceans/Handbook_2007.html
    """
    # convert temperature to IPTS-68
    T68 = (TempC + 0.0002) / 0.99975
    pSMOW = (
        999.842594
        + 6.793952e-2 * T68
        + -9.095290e-3 * T68 ** 2
        + 1.001685e-4 * T68 ** 3
        + -1.120083e-6 * T68 ** 4
        + 6.536332e-9 * T68 ** 5
    )
    A = (
        8.24493e-1
        + -4.0899e-3 * T68
        + 7.6438e-5 * T68 ** 2
        + -8.2467e-7 * T68 ** 3
        + 5.3875e-9 * T68 ** 4
    )
    B = -5.72466e-3 + 1.0227e-4 * T68 + -1.6546e-6 * T68 ** 2
    C = 4.8314e-4
    return (pSMOW + A * Sal + B * Sal ** 1.5 + C * Sal ** 2) / 1000

def calc_pH_scales(pHtot, pHfree, pHsws, pHNBS, Sal, TempC=None, p_bar=None, ST=None, FT=None, Ks=None, TempK=None):
    """
    Calculate pH on all scales, given one.
    """

    # check if any pH scale is given.
    npH = sum([x is not None for x in [pHfree, pHsws, pHtot, pHNBS]])

    if npH == 0:
        raise ValueError("At least one pH scale must be provided (pHfree, pHsws, pHtot, pHNBS).")
    
    if npH > 1:
        raise ValueError("Only one pH scale can be provided at a time (pHfree, pHsws, pHtot, pHNBS).")
    
    if TempK is None:
        if TempC is None:
            raise ValueError("Temperature must be provided either as TempC or TempK.")
        TempK = TempC + 273.15
    
    if ST is None:
        ST = constants.calc_ST(Sal)
    if FT is None:
        FT = constants.calc_FT(Sal)
    if p_bar is None:
        P_bar = 0.0
    if Ks is None:
        Ks = constants.calc_Ks(temp_c=TempC, sal=Sal, p_bar=p_bar)

    # pH scale conversions
    FREEtoTOT = -np.log10((1 + ST / Ks.KS))
    SWStoTOT = -np.log10((1 + ST / Ks.KS) / (1 + ST / Ks.KS + FT / Ks.KF))
    fH = constants.calc_fH(TempK, Sal)

    if pHtot is not None:
        return {
            "pHfree": pHtot - FREEtoTOT,
            "pHsws": pHtot - SWStoTOT,
            "pHNBS": pHtot - SWStoTOT - np.log10(fH),
        }
    elif pHsws is not None:
        return {
            "pHfree": pHsws + SWStoTOT - FREEtoTOT,
            "pHtot": pHsws + SWStoTOT,
            "pHNBS": pHsws - np.log10(fH),
        }
    elif pHfree is not None:
        return {
            "pHsws": pHfree + FREEtoTOT - SWStoTOT,
            "pHtot": pHfree + FREEtoTOT,
            "pHNBS": pHfree + FREEtoTOT - SWStoTOT - np.log10(fH),
        }
    elif pHNBS is not None:
        return {
            "pHsws": pHNBS + np.log10(fH),
            "pHtot": pHNBS + np.log10(fH) + SWStoTOT,
            "pHfree": pHNBS + np.log10(fH) + SWStoTOT - FREEtoTOT,
        }


def pH_scale_converter(pH, scale, TempC, Sal, Press_bar=None, ST=None, FT=None):
    """
    Returns pH on all scales.
    """
    pH_scales = ["Total", "FREE", "SWS", "NBS"]
    if scale not in pH_scales:
        raise ValueError("scale must be one of Total, NBS, SWS or FREE.")

    inp = [None, None, None, None]
    inp[np.argwhere(scale == np.array(pH_scales))[0, 0]] = pH

    return calc_pH_scales(*inp, TempC=TempC, Sal=Sal, p_bar=Press_bar)
