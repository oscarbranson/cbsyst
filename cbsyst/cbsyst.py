"""
Functions for calculating the carbon and boron chemistry of seawater.
"""

import numpy as np
import pandas as pd
from cbsyst.carbon import calc_C_species, calc_revelle_factor, pCO2_to_fCO2, fCO2_to_CO2
from cbsyst.boron import calc_B_species
from cbsyst.boron_isotopes import d11_to_A11, A11_to_d11, get_alphaB, calc_B_isotopes
from cbsyst.helpers import Bunch, ch, cp, NnotNone, calc_TF, calc_TS, calc_TB, calc_pH_scales, calc_Ks

# C Speciation
# ------------
def Csys(
    pHtot=None, DIC=None, TA=None,
    CO2=None, HCO3=None, CO3=None,
    pCO2=None, fCO2=None,
    BT=None,
    Ca=None, Mg=None,
    T_in=25.0, T_out=None, 
    S_in=35.0, S_out=None,
    P_in=None, P_out=None,
    TP=0.0, TSi=0.0,
    TS=None, TF=None,
    pHsws=None, pHfree=None, pHNBS=None,
    unit="umol", Ks=None,
    pdict=None,
):
    """
    Calculate the carbon chemistry of seawater from a minimal parameter set.

    Constants calculated by MyAMI model (Hain et al, 2015; doi:10.1002/2014GB004986).
    Speciation calculations from Zeebe & Wolf-Gladrow (2001; ISBN:9780444509468) Appendix B

    Inputs must either be single values, arrays of equal length or a mixture of both.
    If you use arrays of unequal length, it won't work.

    Error propagation:
    If inputs are ufloat or uarray (from uncertainties package) errors will
    be propagated through all calculations, but:

    **WARNING** Error propagation NOT IMPLEMENTED for carbon system calculations
    with zero-finders (i.e. when pH is not given; cases 2-5 and 10-15).

    Concentration Units
    +++++++++++++++++++
    * Ca and Mg must be in molar units.
    * All other units must be the same, and can be specified in the 'unit' variable. Defaults to umolar.

    Parameters
    ----------
    pH, DIC, CO2, HCO3, CO3, TA : array-like
        Carbon system parameters. Two of these must be provided.
    BT : array-like
        Total B at Salinity = 35, used in Alkalinity calculations.
    Ca, Mg : arra-like
        The [Ca] and [Mg] of the seawater, in mol / kg.
        Used in calculating MyAMI constants.
    T_in, S_in : array-like
        Temperature in Celcius and Salinity in PSU that the
        measurements were conducted under.
        Used in calculating constants.
    P_in : array-like
        Pressure in Bar that the measurements were conducted under.
        Used in pressure-correcting constants.
    T_out, S_out : array-like
        Temperature in Celcius and Salinity in PSU of the desired
        output conditions.
        Used in calculating constants.
    P_in : array-like
        Pressure in Bar of the desired output conditions.
        Used in pressure-correcting constants.
    unit : str
        Concentration units of C and B parameters (all must be in
        the same units).
        Can be 'mol', 'mmol', 'umol', 'nmol', 'pmol' or 'fmol'.
        Used in calculating Alkalinity. Default is 'umol'.
    Ks : dict
        A dictionary of constants. Must contain keys
        'K1', 'K2', 'KB' and 'KW'.
        If None, Ks are calculated using MyAMI model.
    pdict : dict
        Optionally, you can provide some or all parameters as a dict,
        with keys the same as the parameter names above. Any parameters
        included in the dict will overwrite manually specified
        parameters. This is particularly useful if you're including
        this in other code.

    Returns
    -------
    dict(/Bunch) containing all calculated parameters.
    """

    # Bunch inputs
    ps = Bunch(locals())
    if isinstance(pdict, dict):
        ps.update(pdict)

    # convert unit to multiplier
    udict = {
        "mol": 1.0,
        "mmol": 1.0e3,
        "umol": 1.0e6,
        "nmol": 1.0e9,
        "pmol": 1.0e12,
        "fmol": 1.0e15,
    }
    if isinstance(ps.unit, str):
        ps.unit = udict[ps.unit]

    if ps.unit != 1:
        upar = ["DIC", "TA", "CO2", "HCO3", "CO3", "BT", "fCO2", "pCO2", "TP", "TSi"]
        for p in upar:
            if ps[p] is not None:
                ps[p] = np.divide(ps[p], ps.unit)  # convert to molar
    
    # Conserved seawater chemistry
    if ps.TS is None:
        ps.TS = calc_TS(ps.S_in)
    if ps.TF is None:
        ps.TF = calc_TF(ps.S_in)
    if ps.BT is None:
        ps.BT = calc_TB(ps.S_in)

    # Remove negative values 
    for p in ["DIC", "TA", "CO2", "HCO3", "CO3", "BT", "fCO2", "pCO2", "TP", "TSi"]:
        if ps[p] is not None:
            if isinstance(ps[p], (np.ndarray, pd.core.series.Series)):
                ps[p][ps[p] < 0] = np.nan
            elif ps[p] < 0:
                ps[p] = np.nan

    
    # Calculate Ks at input conditions
    ps.Ks = calc_Ks(T=ps.T_in, S=ps.S_in, P=ps.P_in, Mg=ps.Mg, Ca=ps.Ca, TS=ps.TS, TF=ps.TF, Ks=ps.Ks)

    # Calculate pH scales at input conditions (does nothing if no pH given)
    ps.update(
        calc_pH_scales(
            pHtot=ps.pHtot,
            pHfree=ps.pHfree,
            pHsws=ps.pHsws,
            pHNBS=ps.pHNBS,
            TS=ps.TS,
            TF=ps.TF,
            TempK=ps.T_in + 273.15,
            Sal=ps.S_in,
            Ks=ps.Ks
        )
    )

    # calculate C system at input conditions
    ps.update(calc_C_species(**ps))
    
    ps["revelle_factor"] = calc_revelle_factor(
        TA=ps.TA,
        DIC=ps.DIC,
        BT=ps.BT,
        TP=ps.TP,
        TSi=ps.TSi,
        TS=ps.TS,
        TF=ps.TF,
        Ks=ps.Ks,
    )

    # clean up output
    outputs = [
        "BT", "CO2", "CO3", "Ca", "DIC", "H", "HCO3", 
        "Mg", "S_in", "T_in", "TA", "CAlk", "PAlk", 
        "SiAlk", "OH"]
    for k in outputs:
        if not isinstance(ps[k], np.ndarray):
            # convert all outputs to (min) 1D numpy arrays.
            ps[k] = np.array(ps[k], ndmin=1)
    if ps.unit != 1:
        for p in upar + ["CAlk", "BAlk", "PAlk", "SiAlk", "OH", "HSO4", "HF", "Hfree"]:
            ps[p] *= ps.unit  # convert back to input units

    # Calculate Output Conditions
    # ===========================
    if ps.T_out is not None or ps.S_out is not None or ps.P_out is not None:
        if ps.T_out is None:
            ps.T_out = ps.T_in
        if ps.S_out is None:
            ps.S_out = ps.S_in
        if ps.P_out is None:
            ps.P_out = ps.P_in
        # assumes conserved alkalinity and DIC
        out_cond = Csys(
            TA=ps.TA,
            DIC=ps.DIC,
            T_in=ps.T_out,
            S_in=ps.S_out,
            P_in=ps.P_out,
            unit=ps.unit,
        )

        # rename parameters in output conditions
        outputs = [
            "BAlk", "BT", "CAlk", "CO2", "CO3", "DIC", "H", "HCO3", 
            "HF", "HSO4", "Hfree", "Ks", "OH", "PAlk", "SiAlk", "TA", "TF",
            "TP", "TS", "TSi", "fCO2", "pCO2", "pHfree", "pHsws", 
            "pHtot", "pHNBS", "revelle_factor",
        ]

        ps.update({k + "_in": ps[k] for k in outputs})
        ps.update({k: out_cond[k] for k in outputs})

    # remove some superfluous outputs
    rem = ["pdict"]
    for r in rem:
        if r in ps:
            del ps[r]

    return ps


# B Speciation
# ------------
def Bsys(
    pHtot=None,
    BT=None,
    BO3=None,
    BO4=None,
    ABT=None,
    ABO3=None,
    ABO4=None,
    dBT=None,
    dBO3=None,
    dBO4=None,
    alphaB=None,
    T_in=25.0,
    S_in=35.0,
    P_in=None,
    Ca=None,
    Mg=None,
    TS=None,
    TF=None,
    pHsws=None,
    pHfree=None,
    pHNBS=None,
    Ks=None,
    pdict=None,
):
    """
    Calculate the boron chemistry of seawater from a minimal parameter set.

    Constants calculated by MyAMI model (Hain et al, 2015; doi:10.1002/2014GB004986).
    Speciation calculations from Zeebe & Wolf-Gladrow (2001; ISBN:9780444509468).

    Inputs must either be single values, arrays of equal length or a mixture of both.
    If you use arrays of unequal length, it won't work.

    Error propagation:
    If inputs are ufloat or uarray (from uncertainties package) errors will
    be propagated through all calculations.

    Concentration Units
    +++++++++++++++++++
    * All concentrations must be in the same units. Returned in the same units as inputs.

    Parameters
    ----------
    pH, BT, BO3, BO4 : array-like
        Boron system parameters. Two of these must be provided.
    dBT, dBO3, dBO4, ABT, ABO3, ABO4 : array-like
        delta (d) or fractional abundance (A) values for the Boron
        isotope system. One of these must be provided.
    alphaB : array-like
        The alpha value for BO3-BO4 isotope fractionation.
    T, S : array-like
        Temperature in Celcius and Salinity in PSU.
        Used in calculating MyAMI constants.
    P : array-like
        Pressure in Bar.
        Used in calculating MyAMI constants.
    Ca, Mg : arra-like
        The [Ca] and [Mg] of the seawater, in mol / kg.
        Used in calculating MyAMI constants.
    Ks : dict
        A dictionary of constants. Must contain keys
        'K1', 'K2', 'KB' and 'KW'.
        If None, Ks are calculated using MyAMI model.
    pdict : dict
        Optionally, you can provide some or all parameters as a dict,
        with keys the same as the parameter names above. Any parameters
        included in the dict will overwrite manually specified
        parameters. This is particularly useful if you're including
        this in other code.

    Returns
    -------
    dict(/Bunch) containing all calculated parameters.
    """
    # input checks
    if NnotNone(BT, BO3, BO4) < 1:
        raise ValueError("Must provide at least one of BT, BO3 or BO4")
    if NnotNone(dBT, dBO3, dBO4, ABT, ABO3, ABO4) < 1:
        raise ValueError("Must provide one of dBT, dBO3, dBO4, ABT, ABO3 or ABO4")

    # Bunch inputs
    ps = Bunch(locals())
    if isinstance(pdict, dict):
        ps.update(pdict)

    # Conserved seawater chemistry
    if ps.TS is None:
        ps.TS = calc_TS(ps.S_in)
    if ps.TF is None:
        ps.TF = calc_TF(ps.S_in)
    
    # Remove negative values 
    for p in ["BT", "BO3", "BO4", "TS", "TF"]:
        if ps[p] is not None:
            if isinstance(ps[p], (np.ndarray, pd.core.series.Series)):
                ps[p][ps[p] < 0] = np.nan
            elif ps[p] < 0:
                ps[p] = np.nan

    # Calculate Ks
    ps.Ks = calc_Ks(T=ps.T_in, S=ps.S_in, P=ps.P_in, Mg=ps.Mg, Ca=ps.Ca, TS=ps.TS, TF=ps.TF, Ks=ps.Ks)

    # Calculate pH scales (does nothing if no pH given)
    ps.update(
        calc_pH_scales(
            pHtot=ps.pHtot,
            pHfree=ps.pHfree,
            pHsws=ps.pHsws,
            pHNBS=ps.pHNBS,
            TS=ps.TS,
            TF=ps.TF,
            TempK=ps.T_in + 273.15,
            Sal=ps.S_in,
            Ks=ps.Ks,
        )
    )
    
    # calcualte pH if not provided.
    if ps.pHtot is None:
        if ps.dBT is None and ps.ABT is None:
            ps.dBT = 39.61
        if ps.dBT is not None:
            ps.ABT = d11_to_A11(ps.dBT)
        if ps.dBO3 is not None:
            ps.ABO3 = d11_to_A11(ps.dBO3)
        if ps.dBO4 is not None:
            ps.ABO4 = d11_to_A11(ps.dBO4)
        if ps.alphaB is None:
            ps.alphaB = get_alphaB()
            
        ps.update(calc_B_isotopes(**ps))

    ps.update(calc_B_species(**ps))

    # If pH not calced yet, calculate on all scales (does nothing if all pH scales already calculated)
    ps.update(
        calc_pH_scales(
            pHtot=ps.pHtot,
            pHfree=ps.pHfree,
            pHsws=ps.pHsws,
            pHNBS=ps.pHNBS,
            TS=ps.TS,
            TF=ps.TF,
            TempK=ps.T_in + 273.15,
            Sal=ps.S_in,
            Ks=ps.Ks,
        )
    )

    # If any isotope parameter specified, calculate the isotope systen.
    if NnotNone(ps.ABT, ps.ABO3, ps.ABO4, ps.dBT, ps.dBO3, ps.dBO4) != 0:
        ps.update(ABsys(pdict=ps))

    for k in ["BT", "H", "BO3", "BO4", "Ca", "Mg", "S_in", "T_in"]:
        # convert all outputs to (min) 1D numpy arrays.
        if not isinstance(ps[k], np.ndarray):
            # convert all outputs to (min) 1D numpy arrays.
            ps[k] = np.array(ps[k], ndmin=1)

    # remove some superfluous outputs
    rem = ["pdict"]
    for r in rem:
        if r in ps:
            del ps[r]

    return ps


# B Isotopes
# ----------
def ABsys(
    pHtot=None,
    ABT=None,
    ABO3=None,
    ABO4=None,
    dBT=None,
    dBO3=None,
    dBO4=None,
    alphaB=None,
    T_in=25.0,
    S_in=35.0,
    P_in=None,
    Ca=None,
    Mg=None,
    TS=None,
    TF=None,
    pHsws=None,
    pHfree=None,
    pHNBS=None,
    Ks=None,
    pdict=None,
):
    """
    Calculate the boron isotope chemistry of seawater from a minimal parameter set.

    Constants calculated by MyAMI model (Hain et al, 2015; doi:10.1002/2014GB004986).
    Speciation calculations from Zeebe & Wolf-Gladrow (2001; ISBN:9780444509468).

    Inputs must either be single values, arrays of equal length or a mixture of both.
    If you use arrays of unequal length, it won't work.

    Error propagation:
    If inputs are ufloat or uarray (from uncertainties package) errors will
    be propagated through all calculations.

    Concentration Units
    +++++++++++++++++++
    * 'A' is fractional abundance (11B / BT)
    * 'd' are delta values
    Either specified, both returned.

    Parameters
    ----------
    pH, ABT, ABO3, ABO4, dBT, dBO3, dBO4 : array-like
        Boron isotope system parameters. Two of pH, {ABT, dBT},
        {ABO4, dBO4}, {ABO3, dBO3} must be provided.
    alphaB : array-like
        Alpha value describing B fractionation (1.0XXX).
        If missing, it's calculated using the temperature
        sensitive formulation of Honisch et al (2008)
    T, S : array-like
        Temperature in Celcius and Salinity in PSU.
        Used in calculating MyAMI constants.
    P : array-like
        Pressure in Bar.
        Used in calculating MyAMI constants.
    Ca, Mg : arra-like
        The [Ca] and [Mg] of the seawater, in mol / kg.
        Used in calculating MyAMI constants.
    Ks : dict
        A dictionary of constants. Must contain keys
        'K1', 'K2', 'KB' and 'KW'.
        If None, Ks are calculated using MyAMI model.
    pdict : dict
        Optionally, you can provide some or all parameters as a dict,
        with keys the same as the parameter names above. Any parameters
        included in the dict will overwrite manually specified
        parameters. This is particularly useful if you're including
        this in other code.

    Returns
    -------
    dict(/Bunch) containing all calculated parameters.
    """

    # Bunch inputs
    ps = Bunch(locals())
    if isinstance(pdict, dict):
        ps.update(pdict)

    # Conserved seawater chemistry
    if ps.TS is None:
        ps.TS = calc_TS(ps.S_in)
    if ps.TF is None:
        ps.TF = calc_TF(ps.S_in)

    # Calculate Ks
    ps.Ks = calc_Ks(T=ps.T_in, S=ps.S_in, P=ps.P_in, Mg=ps.Mg, Ca=ps.Ca, TS=ps.TS, TF=ps.TF, Ks=ps.Ks)

    # Calculate pH scales (does nothing if no pH given)
    ps.update(
        calc_pH_scales(
            pHtot=ps.pHtot,
            pHfree=ps.pHfree,
            pHsws=ps.pHsws,
            pHNBS=ps.pHNBS,
            TS=ps.TS,
            TF=ps.TF,
            TempK=ps.T_in + 273.15,
            Sal=ps.S_in,
            Ks=ps.Ks,
        )
    )

    # if deltas provided, calculate corresponding As
    if ps.dBT is not None:
        ps.ABT = d11_to_A11(ps.dBT)
    if ps.dBO3 is not None:
        ps.ABO3 = d11_to_A11(ps.dBO3)
    if ps.dBO4 is not None:
        ps.ABO4 = d11_to_A11(ps.dBO4)

    # calculate alpha
    if alphaB is None:
        ps.alphaB = get_alphaB()
    else:
        ps.alphaB = alphaB

    ps.update(calc_B_isotopes(**ps))

    if ps.dBT is None:
        ps.dBT = A11_to_d11(ps.ABT)
    if ps.dBO3 is None:
        ps.dBO3 = A11_to_d11(ps.ABO3)
    if ps.dBO4 is None:
        ps.dBO4 = A11_to_d11(ps.ABO4)

    for k in [
        "ABO3",
        "ABO4",
        "ABT",
        "Ca",
        "H",
        "Mg",
        "S_in",
        "T_in",
        "alphaB",
        "dBO3",
        "dBO4",
        "dBT",
        "pHtot",
    ]:
        if not isinstance(ps[k], np.ndarray):
            # convert all outputs to (min) 1D numpy arrays.
            ps[k] = np.array(ps[k], ndmin=1)

    # remove some superfluous outputs
    rem = ["pdict"]
    for r in rem:
        if r in ps:
            del ps[r]

    return ps


# Whole C-B-Isotope System
# ------------------------
def CBsys(
    pHtot=None,
    DIC=None,
    CO2=None,
    HCO3=None,
    CO3=None,
    TA=None,
    fCO2=None,
    pCO2=None,
    BT=None,
    BO3=None,
    BO4=None,
    ABT=None,
    ABO3=None,
    ABO4=None,
    dBT=None,
    dBO3=None,
    dBO4=None,
    alphaB=None,
    T_in=25.0,
    S_in=35.0,
    P_in=None,
    T_out=None,
    S_out=None,
    P_out=None,
    Ca=None,
    Mg=None,
    TP=0.0,
    TSi=0.0,
    TS=None,
    TF=None,
    pHsws=None,
    pHfree=None,
    pHNBS=None,
    Ks=None,
    pdict=None,
    unit="umol",
):
    """
    Calculate carbon, boron and boron isotope chemistry of seawater from a minimal parameter set.

    Constants calculated by MyAMI model (Hain et al, 2015; doi:10.1002/2014GB004986).
    Speciation calculations from Zeebe & Wolf-Gladrow (2001; ISBN:9780444509468) Appendix B

    Inputs must either be single values, arrays of equal length or a mixture of both.
    If you use arrays of unequal length, it won't work.

    Note: Special Case! If pH is not known, you must provide either:
      - Two of [DIC, CO2, HCO3, CO3], and one of [BT, BO3, BO4]
      - One of [DIC, CO2, HCO3, CO3], and TA and BT
      - Two of [BT, BO3, BO4] and one of [DIC, CO2, HCO3, CO3]

    Isotopes will only be calculated if one of [ABT, ABO3, ABO4, dBT, dBO3, dBO4]
    is provided.

    Error propagation:
    If inputs are ufloat or uarray (from uncertainties package) errors will
    be propagated through all calculations, but:

    **WARNING** Error propagation NOT IMPLEMENTED for carbon system calculations
    with zero-finders (i.e. when pH is not given; cases 2-5 and 10-15).

    Concentration Units
    +++++++++++++++++++
    * Ca and Mg must be in molar units.
    * All other units must be the same, and can be specified in the 'unit' variable. Defaults to umolar.
    * Isotopes can be in A (11B / BT) or d (delta). Either specified, both returned.

    Parameters
    ----------
    pH, DIC, CO2, HCO3, CO3, TA : array-like
        Carbon system parameters. Two of these must be provided.
        If TA is specified, a B species must also be specified.
    pH, BT, BO3, BO4 : array-like
        Boron system parameters. Two of these must be provided.
    pH, ABT, ABO3, ABO4, dBT, dBO3, dBO4 : array-like
        Boron isotope system parameters. pH and one other
        parameter must be provided.
    alphaB : array-like
        Alpha value describing B fractionation (1.0XXX).
        If missing, it's calculated using the temperature
        sensitive formulation of Honisch et al (2008)
    T, S : array-like
        Temperature in Celcius and Salinity in PSU.
        Used in calculating MyAMI constants.
    P : array-like
        Pressure in Bar.
        Used in calculating MyAMI constants.
    unit : str
        Concentration units of C and B parameters (all must be in
        the same units).
        Can be 'mol', 'mmol', 'umol', 'nmol', 'pmol' or 'fmol'.
        Used in calculating Alkalinity. Default is 'umol'.
    Ca, Mg : arra-like
        The [Ca] and [Mg] of the seawater, * in mol / kg *.
        Used in calculating MyAMI constants.
    Ks : dict
        A dictionary of constants. Must contain keys
        'K1', 'K2', 'KB' and 'KW'.
        If None, Ks are calculated using MyAMI model.
    pdict : dict
        Optionally, you can provide some or all parameters as a dict,
        with keys the same as the parameter names above. Any parameters
        included in the dict will overwrite manually specified
        parameters. This is particularly useful if you're including
        this in other code.

    Returns
    -------
    dict(/Bunch) containing all calculated parameters.
    """
    # Bunch inputs
    ps = Bunch(locals())
    if isinstance(pdict, dict):
        ps.update(pdict)

    # convert unit to multiplier
    udict = {
        "mol": 1.0,
        "mmol": 1.0e3,
        "umol": 1.0e6,
        "nmol": 1.0e9,
        "pmol": 1.0e12,
        "fmol": 1.0e15,
    }
    if isinstance(ps.unit, str):
        ps.unit = udict[ps.unit]
    elif isinstance(ps.unit, (int, float)):
        ps.unit = unit

    upar = [
        "DIC",
        "CO2",
        "HCO3",
        "CO3",
        "TA",
        "fCO2",
        "pCO2",
        "BT",
        "BO3",
        "BO4",
        "TP",
        "TSi",
    ]
    for p in upar:
        if ps[p] is not None:
            ps[p] = np.divide(ps[p], ps.unit)  # convert to molar

    # Conserved seawater chemistry
    if ps.TS is None:
        ps.TS = calc_TS(ps.S_in)
    if ps.TF is None:
        ps.TF = calc_TF(ps.S_in)

    # Remove negative values 
    for p in ["DIC", "TA", "CO2", "HCO3", "CO3", "BT", "BO3", "BO4", "fCO2", "pCO2", "TP", "TSi"]:
        if ps[p] is not None:
            if isinstance(ps[p], (np.ndarray, pd.core.series.Series)):
                ps[p][ps[p] < 0] = np.nan
            elif ps[p] < 0:
                ps[p] = np.nan
    
    # Calculate Ks
    ps.Ks = calc_Ks(T=ps.T_in, S=ps.S_in, P=ps.P_in, Mg=ps.Mg, Ca=ps.Ca, TS=ps.TS, TF=ps.TF, Ks=ps.Ks)

    # calculate alpha
    if alphaB is None:
        ps.alphaB = get_alphaB()
    else:
        ps.alphaB = alphaB
        
    # convert any B isotopes to A notation
    if ps.dBT is None and ps.ABT is None:
        ps.dBT = 39.61
    if ps.dBT is not None:
        ps.ABT = d11_to_A11(ps.dBT)
    if ps.dBO3 is not None:
        ps.ABO3 = d11_to_A11(ps.dBO3)
    if ps.dBO4 is not None:
        ps.ABO4 = d11_to_A11(ps.dBO4)
        
    nBiso = NnotNone(ps.ABT) + NnotNone(ps.ABO4, ps.ABO3)
    
    # Calculate all pH scales (does nothing if no pH given)
    ps.update(
        calc_pH_scales(
            pHtot=ps.pHtot,
            pHfree=ps.pHfree,
            pHsws=ps.pHsws,
            pHNBS=ps.pHNBS,
            TS=ps.TS,
            TF=ps.TF,
            TempK=ps.T_in + 273.15,
            Sal=ps.S_in,
            Ks=ps.Ks,
        )
    )
    
    # if fCO2 is given but CO2 is not, calculate CO2
    if ps.CO2 is None:
        if ps.fCO2 is not None:
            ps.CO2 = fCO2_to_CO2(ps.fCO2, ps.Ks)
        elif ps.pCO2 is not None:
            ps.CO2 = fCO2_to_CO2(pCO2_to_fCO2(ps.pCO2, ps.T_in), ps.Ks)
    
    # if no B info provided, assume modern conc.
    nBspec = NnotNone(ps.BT, ps.BO3, ps.BO4)
    if nBspec == 0:
        ps.BT = calc_TB(ps.S_in)

    # count number of not None C parameters
    nCspec = NnotNone(ps.DIC, ps.CO2, ps.HCO3, ps.CO3)  # used below
        
    # if pH or two B species are given:
    if ps.pHtot is not None or nBspec == 2:
        ps.update(calc_B_species(**ps))
        ps.update(calc_C_species(**ps))
        ps.update(calc_B_isotopes(**ps))
    # if pH can be calculated from B isotopes
    elif nBiso == 2:
        ps.update(calc_B_isotopes(**ps))
        ps.update(calc_B_species(**ps))
        ps.update(calc_C_species(**ps))
    # if ther eare two carbon species, or one carbon species + TA and BT
    elif (nCspec == 2) | ((nCspec == 1) & (NnotNone(ps.TA, ps.BT) == 2)):
        ps.update(calc_C_species(**ps))
        ps.update(calc_B_species(**ps))
        ps.update(calc_B_isotopes(**ps))
    
    else:  # if neither condition is met, throw an error
        raise ValueError(
            (
                "Impossible! You haven't provided enough information.\n"
                + "If you don't know pH, you must provide either:\n"
                + "  - Two of [DIC, CO2, HCO3, CO3] and BT\n"
                + "  - One of [DIC, CO2, HCO3, CO3], and TA and BT\n"
                + "  - Two of [BT, BO3, BO4] and one of [DIC, CO2, HCO3, CO3]"
                + "  - Two of [dBT, dBO3, dBO4] and one of [DIC, CO2, HCO3, CO3]"                
            )
        )

    # convert isotopes to delta notation
    if ps.dBT is None:
        ps.dBT = A11_to_d11(ps.ABT)
    if ps.dBO3 is None:
        ps.dBO3 = A11_to_d11(ps.ABO3)
    if ps.dBO4 is None:
        ps.dBO4 = A11_to_d11(ps.ABO4)
    
    ps["revelle_factor"] = calc_revelle_factor(
        TA=ps.TA,
        DIC=ps.DIC,
        BT=ps.BT,
        TP=ps.TP,
        TSi=ps.TSi,
        TS=ps.TS,
        TF=ps.TF,
        Ks=ps.Ks,
    )

    # clean up output
    outputs = [
        "BAlk",
        "BT",
        "CAlk",
        "CO2",
        "CO3",
        "DIC",
        "H",
        "HCO3",
        "HF",
        "HSO4",
        "Hfree",
        "Ks",
        "OH",
        "PAlk",
        "SiAlk",
        "TA",
        "TF",
        "TP",
        "TS",
        "TSi",
        "fCO2",
        "pCO2",
        "pHfree",
        "pHsws",
        "pHtot",
        "pHNBS",
        "BO3",
        "BO4",
        "ABO3",
        "ABO4",
        "dBO3",
        "dBO4",
    ]
    for k in outputs:
        if k == 'Ks':
            continue
        if not isinstance(ps[k], np.ndarray):
            # convert all outputs to (min) 1D numpy arrays.
            ps[k] = np.array(ps[k], ndmin=1)

    # Handle Units
    for p in upar + ["CAlk", "BAlk", "PAlk", "SiAlk", "OH", "HSO4", "HF", "Hfree"]:
        ps[p] *= ps.unit  # convert back to input units

    # Calculate Output Conditions
    # ===========================
    
    # Recursive approach to calculate output params.
    # if output conditions specified, calculate outputs.
    if ps.T_out is not None or ps.S_out is not None or ps.P_out is not None:
        if ps.T_out is None:
            ps.T_out = ps.T_in
        if ps.S_out is None:
            ps.S_out = ps.S_in
        if ps.P_out is None:
            ps.P_out = ps.P_in
        # assumes conserved alkalinity, DIC and BT
        out_cond = CBsys(
            TA=ps.TA,
            DIC=ps.DIC,
            BT=ps.BT,
            dBT=ps.dBT,
            T_in=ps.T_out,
            S_in=ps.S_out,
            P_in=ps.P_out,
            unit=ps.unit,
        )
        # rename parameters in output conditions
        ps.update({k + "_in": ps[k] for k in outputs})
        ps.update({k: out_cond[k] for k in outputs})

        # remove some superfluous outputs
    rem = ["pdict", "unit"]
    for r in rem:
        if r in ps:
            del ps[r]
    return ps
