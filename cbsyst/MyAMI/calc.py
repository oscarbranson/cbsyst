""""
User-facing function for calculating K correction factors using MyAMI
"""

import numpy as np
import pandas as pd
from .helpers import shape_matcher
from .pitzer import calculate_gKs

def calc_Fcorr(Sal=35., TempC=25., Na=None, K=None, Mg=None, Ca=None, Sr=None, Cl=None, BOH4=None, HCO3=None, CO3=None, SO4=None):
    """
    Calculate K correction factors as a fn of temp and salinity that can be applied to empirical Ks

    Parameters
    ----------
    TempC : array-like
        Temperature in Celcius
    Sal : array-like
        Salinity in PSU
    Na, K, Mg, Ca, Sr : array-like
        Cation concentrations in mol/kg. If None, mean ocean water values are used.
    Cl, BOH4, HCO3, CO3, SO4 : array-like
        Anion concentrations in mol/kg. If None, mean ocean water values are used.


    Returns
    -------
    dict 
        Correction factors (Fcorr) to be applied to empirical K values,
        where K_corr = K_cond * F_corr.
    """

    # ensure all inputs are the same shape
    TempC, Sal, Na, K, Mg, Ca, Sr, Cl, BOH4, HCO3, CO3, SO4 = shape_matcher(TempC, Sal, Na, K, Mg, Ca, Sr, Cl, BOH4, HCO3, CO3, SO4)

    # Calculate gK's for modern (mod) and experimental (x) seawater composition
    (
        gKspC_mod,
        gK1_mod,
        gK2_mod,
        gKW_mod,
        gKB_mod,
        gKspA_mod,
        gK0_mod,
        gKS_mod,
    ) = calculate_gKs(TempC, Sal)
    
    (
        gKspC_X, 
        gK1_X, 
        gK2_X, 
        gKW_X, 
        gKB_X, 
        gKspA_X, 
        gK0_X, 
        gKS_X) = calculate_gKs(TempC, Sal, Na=Na, K=K, Mg=Mg, Ca=Ca, Sr=Sr, Cl=Cl, BOH4=BOH4, HCO3=HCO3, CO3=CO3, SO4=SO4)

    # Calculate conditional K's predicted for seawater composition X
    F_dict = {
        "K0": gK0_X / gK0_mod,
        "K1": gK1_X / gK1_mod,
        "K2": gK2_X / gK2_mod,
        "KB": gKB_X / gKB_mod,
        "KW": gKW_X / gKW_mod,
        "KspC": gKspC_X / gKspC_mod,
        "KspA": gKspA_X / gKspA_mod,
        "KS": gKS_X / gKS_mod,
    }
    
    return F_dict

