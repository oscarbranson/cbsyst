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
        gKSO4_mod,
    ) = calculate_gKs(TempC, Sal)
    
    (
        gKspC_X, 
        gK1_X, 
        gK2_X, 
        gKW_X, 
        gKB_X, 
        gKspA_X, 
        gK0_X, 
        gKSO4_X) = calculate_gKs(TempC, Sal, Na=Na, K=K, Mg=Mg, Ca=Ca, Sr=Sr, Cl=Cl, BOH4=BOH4, HCO3=HCO3, CO3=CO3, SO4=SO4)

    # Calculate conditional K's predicted for seawater composition X
    F_dict = {
        "K0": gK0_X / gK0_mod,
        "K1": gK1_X / gK1_mod,
        "K2": gK2_X / gK2_mod,
        "KB": gKB_X / gKB_mod,
        "KW": gKW_X / gKW_mod,
        "KspC": gKspC_X / gKspC_mod,
        "KspA": gKspA_X / gKspA_mod,
        "KSO4": gKSO4_X / gKSO4_mod,
    }
    
    return F_dict

def generate_Fcorr_LUT(n=21):
    """
    Generate a Look Up Table (LUT) of Fcorr factors using MyAMI
    on an even grid of TempC, Sal, Mg and Ca.

    The Fcorr factor is used to correct an empirical K value at
    a given TempC and Sal for Mg and Ca concentration, and should be
    applied as:

    Kcorr = Kempirical * Fcorr

    Parameter ranges are:
        TempC: 0 - 40 Celcius
        Sal: 30-40 PSU
        Mg, Ca: 0, 0.06 mol/kg

    Parameters
    ----------
    n : int
        The number of grid points to calculate for each parameter.
    
    Returns
    -------
    pandas.DataFrame
    """

    TempC = np.linspace(0, 40, n)
    Sal = np.linspace(30, 40, n)
    Mg = np.linspace(0, 0.06, n)
    Ca = np.linspace(0, 0.06, n)

    # grid inputs
    gTempC, gSal, gMg, gCa = np.meshgrid(TempC, Sal, Mg, Ca)

    # calculate Fcorr
    Fcorr = calc_Fcorr(Sal=gSal, TempC=gTempC, Mg=gMg, Ca=gCa)

    # assign axes
    Fcorr['TempC'] = TempC
    Fcorr['Sal'] = Sal
    Fcorr['Mg'] = Mg
    Fcorr['Ca'] = Ca

    return Fcorr
