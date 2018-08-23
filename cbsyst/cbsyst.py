import numpy as np
from cbsyst.helpers import Bunch, maxL
from cbsyst.MyAMI_V2 import MyAMI_K_calc, MyAMI_K_calc_multi
from cbsyst.carbon_fns import *
from cbsyst.boron_fns import *
from cbsyst.helpers import ch, cp, NnotNone, calc_TF, calc_TS, calc_TB, calc_pH_scales
from cbsyst.non_MyAMI_constants import *


# Helper functions
# ----------------
def calc_Ks(T, S, P, Mg, Ca, TS, TF, Ks=None):
    """
    Helper function to calculate Ks.

    If Ks is a dict, those Ks are used
    transparrently (i.e. no pressure modification).
    """
    if isinstance(Ks, dict):
        Ks = Bunch(Ks)
    else:
        if maxL(Mg, Ca) == 1:
            if Mg is None:
                Mg = 0.0528171
            if Ca is None:
                Ca = 0.0102821
            Ks = MyAMI_K_calc(TempC=T, Sal=S, P=P,
                              Mg=Mg, Ca=Ca)
        else:
            # if only Ca or Mg provided, fill in other with modern
            if Mg is None:
                Mg = 0.0528171
            if Ca is None:
                Ca = 0.0102821
            # calculate Ca and Mg specific Ks
            Ks = MyAMI_K_calc_multi(TempC=T, Sal=S, P=P,
                                    Ca=Ca, Mg=Mg)

        # non-MyAMI Constants
        Ks.update(calc_KPs(T, S, P))
        Ks.update(calc_KF(T, S, P))
        Ks.update(calc_KSi(T, S, P))

        # pH conversions to total scale.
        #   - KP1, KP2, KP3 are all on SWS
        #   - KSi is on SWS
        #   - MyAMI KW is on SWS... DOES THIS MATTER?

        SWStoTOT = (1 + TS / Ks.KSO4) / (1 + TS / Ks.KSO4 + TF / Ks.KF)
        # FREEtoTOT = 1 + 'T_' + mode]S / Ks.KSO4
        conv = ['KP1', 'KP2', 'KP3', 'KSi', 'KW']
        for c in conv:
            Ks[c] *= SWStoTOT

    return Ks


# C Speciation
# ------------
def Csys(pHtot=None, DIC=None, CO2=None,
         HCO3=None, CO3=None, TA=None,
         fCO2=None, pCO2=None,
         BT=None, Ca=None, Mg=None,
         T_in=25., S_in=35., P_in=None,
         T_out=None, S_out=None, P_out=None,
         TP=0., TSi=0.,
         pHsws=None, pHfree=None, pHNBS=None,
         Ks=None, pdict=None, unit='umol'):
    """
    Calculate the carbon chemistry of seawater from a minimal parameter set.

    Constants calculated by MyAMI model (Hain et al, 2015; doi:10.1002/2014GB004986).
    Speciation calculations from Zeebe & Wolf-Gladrow (2001; ISBN:9780444509468) Appendix B

    pH is Total scale.

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
    udict = {'mol': 1.,
             'mmol': 1.e3,
             'umol': 1.e6,
             'nmol': 1.e9,
             'pmol': 1.e12,
             'fmol': 1.e15}
    if isinstance(ps.unit, str):
        ps.unit = udict[ps.unit]
    # elif isinstance(ps.unit, (int, float)):
    #     ps.unit = ps.unit

    if ps.unit != 1:
        upar = ['DIC', 'TA', 'CO2', 'HCO3', 'CO3',
                'BT', 'fCO2', 'pCO2', 'TP', 'TSi']
        for p in upar:
            if ps[p] is not None:
                ps[p] = np.divide(ps[p], ps.unit)  # convert to molar

    # Conserved seawater chemistry
    if 'TS' not in ps:
        ps.TS = calc_TS(ps.S_in)
    if 'TF' not in ps:
        ps.TF = calc_TF(ps.S_in)
    if ps.BT is None:
        ps.BT = calc_TB(ps.S_in)
    # elif isinstance(BT, (int, float)):
    #     ps.BT = ps.BT * ps.S_in / 35.

    # Calculate Ks at input conditions
    ps.Ks = calc_Ks(ps.T_in, ps.S_in, ps.P_in,
                    ps.Mg, ps.Ca, ps.TS, ps.TF, ps.Ks)

    # Calculate pH scales at input conditions (does nothing if no pH given)
    ps.update(calc_pH_scales(ps.pHtot, ps.pHfree, ps.pHsws, ps.pHNBS,
                             ps.TS, ps.TF, ps.T_in + 273.15, ps.S_in, ps.Ks))

    # calculate C system at input conditions
    ps.update(calc_C_species(pHtot=ps.pHtot, DIC=ps.DIC, CO2=ps.CO2,
                             HCO3=ps.HCO3, CO3=ps.CO3, TA=ps.TA,
                             fCO2=ps.fCO2, pCO2=ps.pCO2,
                             T_in=ps.T_in, BT=ps.BT, TP=ps.TP, TSi=ps.TSi,
                             TS=ps.TS, TF=ps.TF, Ks=ps.Ks))

    # clean up output
    for k in ['BT', 'CO2', 'CO3', 'Ca', 'DIC', 'H',
              'HCO3', 'Mg', 'S_in', 'T_in', 'TA',
              'CAlk', 'PAlk', 'SiAlk', 'OH']:
        if not isinstance(ps[k], np.ndarray):
            # convert all outputs to (min) 1D numpy arrays.
            ps[k] = np.array(ps[k], ndmin=1)
    if ps.unit != 1:
        for p in upar + ['CAlk', 'BAlk', 'PAlk', 'SiAlk',
                         'OH', 'HSO4', 'HF', 'Hfree']:
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
        out_cond = Csys(TA=ps.TA, DIC=ps.DIC, 
                        T_in=ps.T_out,
                        S_in=ps.S_out,
                        P_in=ps.P_out,
                        unit=ps.unit)
        # Calculate pH scales at output conditions (does nothing if no pH given)
        out_cond.update(calc_pH_scales(out_cond.pHtot, out_cond.pHfree, out_cond.pHsws, out_cond.pHNBS,
                                       out_cond.TS, out_cond.TF, out_cond.T_in + 273.15, out_cond.S_in, out_cond.Ks))

        # rename parameters in output conditions
        outputs = ['BAlk', 'BT', 'CAlk', 'CO2', 'CO3',
                   'DIC', 'H', 'HCO3', 'HF',
                   'HSO4', 'Hfree', 'Ks', 'OH',
                   'PAlk', 'SiAlk', 'TA', 'TF', 'TP',
                   'TS', 'TSi', 'fCO2', 'pCO2',
                   'pHfree', 'pHsws', 'pHtot', 'pHNBS']

        ps.update({k + '_out': out_cond[k] for k in outputs})

    # remove some superfluous outputs
    rem = ['pdict']
    for r in rem:
        if r in ps:
            del ps[r]

    return ps


# B Speciation
# ------------
def Bsys(pHtot=None, BT=None, BO3=None, BO4=None,
         ABT=None, ABO3=None, ABO4=None,
         dBT=None, dBO3=None, dBO4=None,
         alphaB=None,
         T_in=25., S_in=35., P_in=None,
         T_out=None, S_out=None, P_out=None,
         Ca=None, Mg=None,
         pHsws=None, pHfree=None, pHNBS=None,
         Ks=None, pdict=None):
    """
    Calculate the boron chemistry of seawater from a minimal parameter set.

    Constants calculated by MyAMI model (Hain et al, 2015; doi:10.1002/2014GB004986).
    Speciation calculations from Zeebe & Wolf-Gladrow (2001; ISBN:9780444509468).

    pH is Total scale.

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
    if 'TS' not in ps:
        ps.TS = calc_TS(ps.S_in)
    if 'TF' not in ps:
        ps.TF = calc_TF(ps.S_in)

    # Calculate Ks
    ps.Ks = calc_Ks(ps.T_in, ps.S_in, ps.P_in,
                    ps.Mg, ps.Ca, ps.TS, ps.TF, ps.Ks)

    # Calculate pH scales (does nothing if none pH given)
    ps.update(calc_pH_scales(ps.pHtot, ps.pHfree, ps.pHsws, ps.pHNBS,
                             ps.TS, ps.TF, ps.T_in + 273.15, ps.S_in, ps.Ks))

    ps.update(calc_B_species(pHtot=ps.pHtot, BT=ps.BT, BO3=ps.BO3, BO4=ps.BO4, Ks=ps.Ks))

    # If pH not calced yet, calculate on all scales
    if ps.pHtot is None:
        ps.pHtot = np.array(cp(ps.H), ndmin=1)
        # Calculate other pH scales
        ps.update(calc_pH_scales(ps.pHtot, ps.pHfree, ps.pHsws, ps.pHNBS,
                                 ps.TS, ps.TF, ps.T_in + 273.15, ps.S_in, ps.Ks))

    # If any isotope parameter specified, calculate the isotope systen.
    if NnotNone(ps.ABT, ps.ABO3, ps.ABO4, ps.dBT, ps.dBO3, ps.dBO4) != 0:
        ps.update(ABsys(pdict=ps))

    for k in ['BT', 'H', 'BO3', 'BO4',
              'Ca', 'Mg', 'S_in', 'T_in']:
        # convert all outputs to (min) 1D numpy arrays.
        if not isinstance(ps[k], np.ndarray):
            # convert all outputs to (min) 1D numpy arrays.
            ps[k] = np.array(ps[k], ndmin=1)

    # remove some superfluous outputs
    rem = ['pdict']
    for r in rem:
        if r in ps:
            del ps[r]

    return ps


# B Isotopes
# ----------
def ABsys(pHtot=None,
          ABT=None, ABO3=None, ABO4=None,
          dBT=None, dBO3=None, dBO4=None,
          alphaB=None,
          T_in=25., S_in=35., P_in=None,
          Ca=None, Mg=None,
          pHsws=None, pHfree=None, pHNBS=None,
          Ks=None, pdict=None):
    """
    Calculate the boron isotope chemistry of seawater from a minimal parameter set.

    Constants calculated by MyAMI model (Hain et al, 2015; doi:10.1002/2014GB004986).
    Speciation calculations from Zeebe & Wolf-Gladrow (2001; ISBN:9780444509468).

    pH is Total scale.

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
    if 'TS' not in ps:
        ps.TS = calc_TS(ps.S_in)
    if 'TF' not in ps:
        ps.TF = calc_TF(ps.S_in)

    # Calculate Ks
    ps.Ks = calc_Ks(ps.T_in, ps.S_in, ps.P_in,
                    ps.Mg, ps.Ca, ps.TS, ps.TF, ps.Ks)

    # Calculate pH scales (does nothing if none pH given)
    ps.update(calc_pH_scales(ps.pHtot, ps.pHfree, ps.pHsws, ps.pHNBS,
                             ps.TS, ps.TF, ps.T_in + 273.15, ps.S_in, ps.Ks))

    # if deltas provided, calculate corresponding As
    if ps.dBT is not None:
        ps.ABT = d11_2_A11(ps.dBT)
    if ps.dBO3 is not None:
        ps.ABO3 = d11_2_A11(ps.dBO3)
    if ps.dBO4 is not None:
        ps.ABO4 = d11_2_A11(ps.dBO4)

    # calculate alpha
    ps.alphaB = alphaB_calc(ps.T)

    if ps.pHtot is not None and ps.ABT is not None:
        ps.H = ch(ps.pHtot)
    elif ps.pHtot is not None and ps.ABO3 is not None:
        ps.ABT = pH_ABO3(ps.pHtot, ps.ABO3, ps.Ks, ps.alphaB)
    elif ps.pHtot is not None and ps.ABO4 is not None:
        ps.ABT = pH_ABO3(ps.pHtot, ps.ABO4, ps.Ks, ps.alphaB)
    else:
        raise ValueError('pH must be determined to calculate isotopes.')

    if ps.ABO3 is None:
        ps.ABO3 = cABO3(ps.H, ps.ABT, ps.Ks, ps.alphaB)
    if ps.ABO4 is None:
        ps.ABO4 = cABO4(ps.H, ps.ABT, ps.Ks, ps.alphaB)

    if ps.dBT is None:
        ps.dBT = A11_2_d11(ps.ABT)
    if ps.dBO3 is None:
        ps.dBO3 = A11_2_d11(ps.ABO3)
    if ps.dBO4 is None:
        ps.dBO4 = A11_2_d11(ps.ABO4)

    for k in ['ABO3', 'ABO4', 'ABT', 'Ca',
              'H', 'Mg', 'S', 'T', 'alphaB',
              'dBO3', 'dBO4', 'dBT', 'pHtot']:
        if not isinstance(ps[k], np.ndarray):
            # convert all outputs to (min) 1D numpy arrays.
            ps[k] = np.array(ps[k], ndmin=1)

    # remove some superfluous outputs
    rem = ['pdict']
    for r in rem:
        if r in ps:
            del ps[r]

    return ps


# Whole C-B-Isotope System
# ------------------------
def CBsys(pHtot=None, DIC=None, CO2=None, HCO3=None, CO3=None, TA=None, fCO2=None, pCO2=None,
          BT=None, BO3=None, BO4=None,
          ABT=None, ABO3=None, ABO4=None, dBT=None, dBO3=None, dBO4=None,
          alphaB=None,
          T_in=25., S_in=35., P_in=None,
          T_out=None, S_out=None, P_out=None,
          Ca=None, Mg=None, TP=0., TSi=0.,
          pHsws=None, pHfree=None, pHNBS=None,
          Ks=None, pdict=None, unit='umol'):
    """
    Calculate carbon, boron and boron isotope chemistry of seawater from a minimal parameter set.

    Constants calculated by MyAMI model (Hain et al, 2015; doi:10.1002/2014GB004986).
    Speciation calculations from Zeebe & Wolf-Gladrow (2001; ISBN:9780444509468) Appendix B

    pH is Total scale.

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
    udict = {'mol': 1.,
             'mmol': 1.e3,
             'umol': 1.e6,
             'nmol': 1.e9,
             'pmol': 1.e12,
             'fmol': 1.e15}
    if isinstance(ps.unit, str):
        ps.unit = udict[ps.unit]
    elif isinstance(ps.unit, (int, float)):
        ps.unit = unit

    upar = ['DIC', 'CO2', 'HCO3', 'CO3', 'TA', 'fCO2', 'pCO2',
            'BT', 'BO3', 'BO4', 'TP', 'TSi']
    for p in upar:
        if ps[p] is not None:
            ps[p] = np.divide(ps[p], ps.unit)  # convert to molar

    # reassign unit, convert back at end
    orig_unit = ps.unit
    ps.unit = 1.

    # Conserved seawater chemistry
    if 'TS' not in ps:
        ps.TS = calc_TS(ps.S_in)
    if 'TF' not in ps:
        ps.TF = calc_TF(ps.S_in)

    # Calculate Ks
    ps.Ks = calc_Ks(ps.T_in, ps.S_in, ps.P_in,
                    ps.Mg, ps.Ca, ps.TS, ps.TF, ps.Ks)

    # Calculate pH scales (does nothing if none pH given)
    ps.update(calc_pH_scales(ps.pHtot, ps.pHfree, ps.pHsws, ps.pHNBS,
                             ps.TS, ps.TF, ps.T_in + 273.15, ps.S_in, ps.Ks))

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
    elif isinstance(BT, (int, float)):
        ps.BT = ps.BT * ps.S_in / 35.
    # count number of not None C parameters
    nCspec = NnotNone(ps.DIC, ps.CO2, ps.HCO3, ps.CO3)  # used below

    # if pH is given, it's easy
    if ps.pHtot is not None or nBspec == 2:
        ps.update(calc_B_species(pHtot=ps.pHtot, BT=ps.BT, BO3=ps.BO3, BO4=ps.BO4, Ks=ps.Ks))
        ps.update(calc_C_species(pHtot=ps.pHtot, DIC=ps.DIC, CO2=ps.CO2,
                                 HCO3=ps.HCO3, CO3=ps.CO3, TA=ps.TA,
                                 fCO2=ps.fCO2, pCO2=ps.pCO2,
                                 T_in=ps.T_in, BT=ps.BT, TP=ps.TP, TSi=ps.TSi,
                                 TS=ps.TS, TF=ps.TF, Ks=ps.Ks))
    # if not, this section works out the order that things should be calculated in.
    # Special case: if pH is missing, must have:
    #   a) two C or one C and both TA and BT
    #   b) two B (above)
    #   c) one pH-dependent B, one pH-dependent C... But that's cray...
    #      (c not implemented!)
    elif ((nCspec == 2) | ((nCspec == 1) & (NnotNone(ps.TA, ps.BT) == 2))):  # case A
        ps.update(calc_C_species(pHtot=ps.pHtot, DIC=ps.DIC, CO2=ps.CO2,
                                 HCO3=ps.HCO3, CO3=ps.CO3, TA=ps.TA,
                                 fCO2=ps.fCO2, pCO2=ps.pCO2,
                                 T_in=ps.T_in, BT=ps.BT, TP=ps.TP, TSi=ps.TSi,
                                 TS=ps.TS, TF=ps.TF, Ks=ps.Ks))
        ps.update(calc_B_species(pHtot=ps.pHtot, BT=ps.BT, BO3=ps.BO3, BO4=ps.BO4, Ks=ps.Ks))
    # elif nBspec == 2:  # case B -- moved up
    #     ps.update(calc_B_species(pHtot=ps.pHtot, BT=ps.BT, BO3=ps.BO3, BO4=ps.BO4, Ks=ps.Ks))
    #     ps.update(calc_C_species(pHtot=ps.pHtot, DIC=ps.DIC, CO2=ps.CO2,
    #                              HCO3=ps.HCO3, CO3=ps.CO3, TA=ps.TA,
    #                              fCO2=ps.fCO2, pCO2=ps.pCO2,
    #                              T_in=ps.T_in, BT=ps.BT, TP=ps.TP, TSi=ps.TSi,
    #                              TS=ps.TS, TF=ps.TF, Ks=ps.Ks))  # then C
    else:  # if neither condition is met, throw an error
        raise ValueError(("Impossible! You haven't provided enough parameters.\n" +
                          "If you don't know pH, you must provide either:\n" +
                          "  - Two of [DIC, CO2, HCO3, CO3], and one of [BT, BO3, BO4]\n" +
                          "  - One of [DIC, CO2, HCO3, CO3], and TA and BT\n" +
                          "  - Two of [BT, BO3, BO4] and one of [DIC, CO2, HCO3, CO3]"))

    for p in upar + ['CAlk', 'BAlk', 'PAlk', 'SiAlk', 'OH', 'HSO4', 'HF', 'Hfree']:
        ps[p] *= orig_unit  # convert back to input units

    # Recursive approach to calculate output params.
    # if output conditions specified, calculate outputs.
    if ps.T_out is not None or ps.S_out is not None or ps.P_out is not None:
        if ps.T_out is None:
            ps.T_out = ps.T_in
        if ps.S_out is None:
            ps.S_out = ps.S_in
        if ps.P_out is None:
            ps.P_out = ps.P_in
        # assumes conserved alkalinity
        out_cond = CBsys(TA=ps.TA, DIC=ps.DIC, BT=ps.BT, T_in=ps.T_out,
                         S_in=ps.S_out, P_in=ps.P_out, unit=ps.unit)
        # Calculate pH scales (does nothing if no pH given)
        out_cond.update(calc_pH_scales(out_cond.pHtot, out_cond.pHfree, out_cond.pHsws, out_cond.pHNBS,
                                       out_cond.TS, out_cond.TF, out_cond.T_in + 273.15, out_cond.S_in, out_cond.Ks))
        # rename parameters in output conditions
        outputs = ['BAlk', 'BT', 'CAlk', 'CO2', 'CO3',
                   'DIC', 'H', 'HCO3', 'HF',
                   'HSO4', 'Hfree', 'Ks', 'OH',
                   'PAlk', 'SiAlk', 'TA', 'TF', 'TP',
                   'TS', 'TSi', 'fCO2', 'pCO2',
                   'pHfree', 'pHsws', 'pHtot', 'pHNBS', 'BO3', 'BO4',
                   'ABO3', 'ABO4', 'dBO3', 'dBO4']

        ps.update({k + '_out': out_cond[k] for k in outputs})

        # remove some superfluous outputs
    rem = ['pdict', 'unit']
    for r in rem:
        if r in ps:
            del ps[r]
    return ps
